#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from mhlc_data_prep.parallel import configure_parallel_context, contiguous_range, worker_dir

PARALLEL = configure_parallel_context()

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel
from mhlc_neuron_probe.io_utils import model_tag, output_dirs, read_jsonl, write_csv, write_json, write_jsonl
from mhlc_neuron_probe.paper_baselines import format_float, resolve_table3_row


DEFAULT_GENERATED_EVAL_PATH = (
    "../mhlc_data/eval_outputs/when2call/"
    "Qwen3-VL-4B-Instruct/when2call_test_generated_4class.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the neuron Resolution Head on When2Call Table-3 style metrics.")
    parser.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data-root", default="../mhlc_data")
    parser.add_argument("--head-checkpoint-path", default=None)
    parser.add_argument("--neuron-path", default=None)
    parser.add_argument("--generated-eval-path", default=DEFAULT_GENERATED_EVAL_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--model-family", default="qwen3_vl", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    parser.add_argument("--thinking-mode", default="off", choices=["auto", "on", "off"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--prefer-unsloth-mirror", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--use-gradient-checkpointing", default="unsloth")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--max-seq-len", type=int, default=32000)
    parser.add_argument("--max-pixels", type=int, default=200000)
    parser.add_argument("--head-input-mode", default=None)
    parser.add_argument("--max-head-input-tokens", type=int, default=None)
    parser.add_argument("--reuse-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def _chunked(items: Sequence[Any], batch_size: int):
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _checkpoint_config(path: Path) -> dict[str, Any]:
    import torch

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {}
    cfg = ckpt.get("config")
    if isinstance(cfg, dict):
        return dict(cfg)
    cfg = ckpt.get("cfg")
    if isinstance(cfg, dict):
        return dict(cfg)
    return {}


def _messages_for_row(when2call: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    tools_json = row.get("tools_json") or when2call.tools_to_text(row.get("tools", []))
    return [
        {"role": "system", "content": row.get("system_prompt") or when2call.build_system_prompt(tools_json)},
        {"role": "user", "content": str(row["question"])},
        {"role": "assistant", "content": str(row["completion"])},
    ]


def _row_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("uuid") or row.get("sample_id") or f"row_{index}")


def _reusable_record(
    record: dict[str, Any],
    *,
    head_path: Path,
    neuron_path: Path,
    head_input_mode: str,
    max_head_input_tokens: int | None,
) -> bool:
    if not isinstance(record.get("head_probs"), list):
        return False
    if record.get("head_checkpoint_path") != rel(head_path):
        return False
    if record.get("neuron_path") != rel(neuron_path):
        return False
    if str(record.get("head_input_mode")) != str(head_input_mode):
        return False
    if record.get("max_head_input_tokens") != max_head_input_tokens:
        return False
    return True


def _make_prediction_record(
    *,
    when2call: Any,
    row: dict[str, Any],
    key: str,
    head_logits: list[float],
    head_probs: list[float],
    decision_threshold: float,
    behavior_to_id: dict[str, int],
    head_path: Path,
    neuron_path: Path,
    head_input_mode: str,
    max_head_input_tokens: int | None,
) -> dict[str, Any]:
    gold_name = when2call.normalize_class_name(row.get("gold_label") or row.get("correct_answer"))
    behavior_scores = when2call.head_probs_to_behavior_scores(head_probs)
    pred_name = when2call.predict_behavior_from_probs(head_probs, float(decision_threshold))
    pred_id = behavior_to_id[pred_name]
    gold_id = behavior_to_id[gold_name]
    pred_score = float(behavior_scores[pred_name])
    gold_score = float(behavior_scores[gold_name])
    sorted_scores = sorted(behavior_scores.values(), reverse=True)
    margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else float(sorted_scores[0])
    return {
        "uuid": key,
        "question": row.get("question"),
        "tools": row.get("tools") or row.get("tools_json"),
        "gold_label": gold_name,
        "generated_completion": row.get("completion"),
        "head_pred_label": pred_name,
        "head_logits": [float(x) for x in head_logits],
        "head_probs": [float(x) for x in head_probs],
        "behavior_scores": behavior_scores,
        "head_pred_score": pred_score,
        "head_gold_behavior_score": gold_score,
        "head_margin": margin,
        "head_is_correct": int(pred_id == gold_id),
        "decision_threshold": float(decision_threshold),
        "head_checkpoint_path": rel(head_path),
        "neuron_path": rel(neuron_path),
        "head_input_mode": str(head_input_mode),
        "max_head_input_tokens": max_head_input_tokens,
    }


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _write_probability_plots(
    *,
    when2call: Any,
    plots_dir: Path,
    metrics: dict[str, Any],
    behavior_class_names: list[str],
    pred_score_correct: list[float],
    pred_score_wrong: list[float],
    gold_score_correct: list[float],
    gold_score_wrong: list[float],
    per_pred_class_scores: dict[str, dict[str, list[float]]],
    prob_one_vs_rest: dict[str, list[float]],
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skip plots: {exc}", flush=True)
        return

    when2call._plot_hist(
        pred_score_correct,
        pred_score_wrong,
        "correct",
        "wrong",
        "Predicted behavior score by correctness",
        plots_dir / "predicted_behavior_score_by_correctness.png",
    )
    when2call._plot_hist(
        gold_score_correct,
        gold_score_wrong,
        "correct",
        "wrong",
        "Gold behavior score by correctness",
        plots_dir / "gold_behavior_score_by_correctness.png",
    )
    for cls_name in behavior_class_names:
        plt.figure(figsize=(7, 4.5))
        plt.hist(prob_one_vs_rest[cls_name], bins=30)
        plt.xlabel(f"score({cls_name})")
        plt.ylabel("Count")
        plt.title(f"Behavior score distribution for {cls_name}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"prob_{cls_name}_one_vs_rest.png", dpi=160)
        plt.close()
        when2call._plot_hist(
            per_pred_class_scores[cls_name]["correct"],
            per_pred_class_scores[cls_name]["wrong"],
            "predicted correctly",
            "predicted wrongly",
            f"{cls_name}: predicted-class score, correct vs wrong",
            plots_dir / f"prob_{cls_name}_predicted_correct_vs_wrong.png",
        )
    when2call._plot_confusion(
        metrics["head_on_generated_completion"]["confusion"],
        behavior_class_names,
        "Neuron Resolution Head confusion matrix",
        plots_dir / "head_confusion.png",
    )


def _comparison_rows(model_path: str, ours_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    paper = resolve_table3_row(model_path)
    return [
        {"system": "Backbone Choice", **paper["backbone_choice"]},
        {"system": "Backbone + Resolution Head（MHLC）", **paper["mhlc"]},
        {
            "system": "Backbone + Resolution Head（Ours）",
            "f1": 100.0 * float(ours_metrics.get("macro_f1", 0.0) or 0.0),
            "acc": 100.0 * float(ours_metrics.get("accuracy", 0.0) or 0.0),
        },
    ]


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\nWhen2Call Table 3 style comparison", flush=True)
    print("| System | F1 ↑ | Acc ↑ |", flush=True)
    print("|---|---:|---:|", flush=True)
    for row in rows:
        print(
            "| {system} | {f1} | {acc} |".format(
                system=row["system"],
                f1=format_float(row.get("f1"), 1),
                acc=format_float(row.get("acc"), 1),
            ),
            flush=True,
        )


def _completion_marker(output_dir: Path) -> Path:
    return output_dir / ".resolution_when2call_eight_gpu_complete.json"


def _merge_parallel_eval(
    *,
    args: argparse.Namespace,
    when2call: Any,
    output_dir: Path,
    work_root: Path,
    rows: list[dict[str, Any]],
    row_stats: dict[str, Any],
    head_class_names: list[str],
    behavior_class_names: list[str],
    behavior_to_id: dict[str, int],
    decision_threshold: float,
) -> None:
    records_by_key: dict[str, dict[str, Any]] = {}
    for rank in range(PARALLEL.world_size):
        path = work_root / f"rank_{rank:02d}" / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing eight-GPU Resolution worker output: {rel(path)}")
        for record in read_jsonl(path):
            records_by_key[str(record.get("uuid") or "")] = record
    records = [records_by_key[_row_key(row, index)] for index, row in enumerate(rows)]
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    gold_ids: list[int] = []
    pred_ids: list[int] = []
    pred_score_correct: list[float] = []
    pred_score_wrong: list[float] = []
    gold_score_correct: list[float] = []
    gold_score_wrong: list[float] = []
    prob_one_vs_rest = {name: [] for name in behavior_class_names}
    per_pred_class_scores = {name: {"correct": [], "wrong": []} for name in behavior_class_names}
    for record in records:
        gold_name = when2call.normalize_class_name(record["gold_label"])
        pred_name = when2call.normalize_class_name(record["head_pred_label"])
        gold_id = behavior_to_id[gold_name]
        pred_id = behavior_to_id[pred_name]
        pred_score = float(record["head_pred_score"])
        gold_score = float(record["head_gold_behavior_score"])
        is_correct = int(pred_id == gold_id)
        gold_ids.append(gold_id)
        pred_ids.append(pred_id)
        if is_correct:
            pred_score_correct.append(pred_score)
            gold_score_correct.append(gold_score)
            per_pred_class_scores[pred_name]["correct"].append(pred_score)
        else:
            pred_score_wrong.append(pred_score)
            gold_score_wrong.append(gold_score)
            per_pred_class_scores[pred_name]["wrong"].append(pred_score)
        for cls_name in behavior_class_names:
            prob_one_vs_rest[cls_name].append(float(record["behavior_scores"][cls_name]))

    head_metrics = when2call.confusion_and_metrics(gold_ids, pred_ids, behavior_class_names)
    metrics = {
        "num_rows_scored": len(rows), "head_class_names": head_class_names, "behavior_class_names": behavior_class_names,
        "row_filter_stats": row_stats, "decision_threshold": float(decision_threshold), "head_on_generated_completion": head_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", records)
    prob_summary = {
        "predicted_behavior_score_correct_mean": _safe_mean(pred_score_correct),
        "predicted_behavior_score_wrong_mean": _safe_mean(pred_score_wrong),
        "gold_behavior_score_correct_mean": _safe_mean(gold_score_correct),
        "gold_behavior_score_wrong_mean": _safe_mean(gold_score_wrong), "per_predicted_class": {},
    }
    for cls_name in behavior_class_names:
        correct_vals = per_pred_class_scores[cls_name]["correct"]
        wrong_vals = per_pred_class_scores[cls_name]["wrong"]
        prob_summary["per_predicted_class"][cls_name] = {
            "num_correct_predictions": len(correct_vals), "num_wrong_predictions": len(wrong_vals),
            "mean_prob_when_correct": _safe_mean(correct_vals), "mean_prob_when_wrong": _safe_mean(wrong_vals),
        }
    write_json(output_dir / "probability_summary.json", prob_summary)
    _write_probability_plots(
        when2call=when2call, plots_dir=plots_dir, metrics=metrics, behavior_class_names=behavior_class_names,
        pred_score_correct=pred_score_correct, pred_score_wrong=pred_score_wrong, gold_score_correct=gold_score_correct,
        gold_score_wrong=gold_score_wrong, per_pred_class_scores=per_pred_class_scores, prob_one_vs_rest=prob_one_vs_rest,
    )
    comparison_rows = _comparison_rows(args.model_path, head_metrics)
    write_json(output_dir / "paper_table3_comparison.json", {"rows": comparison_rows, "paper_source": "Multi-Head Latent Control Table 3"})
    write_csv(output_dir / "paper_table3_comparison.csv", comparison_rows)
    summary = {
        "benchmark": "when2call", "model_path": args.model_path, "num_rows": len(rows), "metrics": metrics,
        "probability_summary": prob_summary, "paper_table3_comparison": comparison_rows, "parallel_workers": PARALLEL.world_size,
    }
    write_json(output_dir / "summary.json", summary)
    _completion_marker(output_dir).write_text(
        json.dumps({"world_size": PARALLEL.world_size, "rows": len(records)}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    _print_table(comparison_rows)
    print(f"[done] saved Resolution When2Call eval to {rel(output_dir)}", flush=True)


def main() -> None:
    args = parse_args()
    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    tag = model_tag(args.model_path)
    dirs = output_dirs(data_root, tag, "resolution")
    head_path = dirs["trained"] / "neuron_head_final.pt" if args.head_checkpoint_path is None else resolve_from_code_root(args.head_checkpoint_path)
    neuron_path = dirs["neurons"] / "selected_neurons.jsonl" if args.neuron_path is None else resolve_from_code_root(args.neuron_path)
    generated_eval_path = resolve_from_code_root(args.generated_eval_path)
    output_dir = data_root / "eval_outputs" / "neuron_heads" / tag / "resolution_when2call" if args.output_dir is None else resolve_from_code_root(args.output_dir)
    base_output_dir = output_dir
    if PARALLEL.enabled and _completion_marker(base_output_dir).exists() and not args.clean:
        if PARALLEL.is_main:
            print(f"[skip] completed eight-GPU Resolution eval: {rel(base_output_dir)}", flush=True)
        return
    if PARALLEL.enabled and base_output_dir.exists() and not args.clean:
        raise FileExistsError(
            f"Existing output is not an eight-GPU completed run: {rel(base_output_dir)}. "
            "Use --clean to start a new eight-GPU run without mixing artifacts."
        )
    work_root = worker_dir(base_output_dir, "resolution_when2call", PARALLEL).parent
    if args.clean and (not PARALLEL.enabled or PARALLEL.is_main):
        clean_path(output_dir, [data_root], "resolution when2call eval output")
        if PARALLEL.enabled:
            clean_path(work_root, [data_root], "resolution eval worker cache")
    PARALLEL.barrier()
    if PARALLEL.enabled:
        output_dir = worker_dir(base_output_dir, "resolution_when2call", PARALLEL)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not head_path.exists():
        raise FileNotFoundError(f"Missing neuron resolution head checkpoint: {rel(head_path)}")
    if not neuron_path.exists():
        raise FileNotFoundError(f"Missing selected neurons: {rel(neuron_path)}")
    if not generated_eval_path.exists():
        raise FileNotFoundError(
            f"Missing When2Call generated eval completions: {rel(generated_eval_path)}\n"
            "Run: python src/06_generate_when2call_eval_completions.py --model-path ../Qwen/Qwen3-VL-4B-Instruct"
        )

    from mhlc_neuron_probe.eval_utils import NeuronHeadScorer

    when2call = load_upstream_module(
        "when2call/eval/eval_when2call_head_only_4class_3sigmoid.py",
        "mhlc_upstream_when2call_head_eval_for_neurons",
    )
    behavior_class_names = [when2call.normalize_class_name(x) for x in when2call.BEHAVIOR_CLASS_NAMES]
    head_class_names = [when2call.normalize_class_name(x) for x in when2call.HEAD_CLASS_NAMES]
    behavior_to_id = {name: idx for idx, name in enumerate(behavior_class_names)}

    ckpt_cfg = _checkpoint_config(head_path)
    head_input_mode = args.head_input_mode or ckpt_cfg.get("head_input_mode") or "completion_text_only"
    max_head_input_tokens = args.max_head_input_tokens
    if max_head_input_tokens is None and ckpt_cfg.get("max_head_input_tokens") is not None:
        max_head_input_tokens = int(ckpt_cfg["max_head_input_tokens"])
    if args.decision_threshold is not None:
        decision_threshold = float(args.decision_threshold)
    elif ckpt_cfg.get("decision_threshold") is not None:
        decision_threshold = float(ckpt_cfg["decision_threshold"])
    else:
        decision_threshold = 0.1

    rows, row_stats = when2call.load_generated_rows(
        str(generated_eval_path),
        args.max_eval_rows,
        behavior_class_names,
    )
    if not rows:
        raise ValueError("No supported eval rows remain after filtering.")
    all_rows = rows
    if PARALLEL.enabled:
        start, end = contiguous_range(len(rows), PARALLEL)
        rows = rows[start:end]
        print(f"[parallel] rank={PARALLEL.rank}/{PARALLEL.world_size} rows={start}:{end}", flush=True)

    config_used = {
        "model_path": args.model_path,
        "head_checkpoint_path": rel(head_path),
        "neuron_path": rel(neuron_path),
        "generated_eval_path": rel(generated_eval_path),
        "output_dir": rel(output_dir),
        "max_eval_rows": args.max_eval_rows,
        "batch_size": int(args.batch_size),
        "decision_threshold": float(decision_threshold),
        "head_input_mode": str(head_input_mode),
        "max_head_input_tokens": max_head_input_tokens,
        "reuse_scores": bool(args.reuse_scores),
        "model_family": args.model_family,
        "thinking_mode": args.thinking_mode,
        "attn_implementation": args.attn_implementation,
    }
    write_json(output_dir / "config_used.json", config_used)
    write_json(output_dir / "row_filter_stats.json", row_stats)

    row_pairs = [(_row_key(row, idx), row) for idx, row in enumerate(rows)]
    predictions_path = output_dir / "predictions.jsonl"
    records_by_key: dict[str, dict[str, Any]] = {}
    if bool(args.reuse_scores) and predictions_path.exists():
        expected_keys = {key for key, _row in row_pairs}
        for record in read_jsonl(predictions_path):
            key = str(record.get("uuid") or "")
            if key not in expected_keys:
                continue
            if not _reusable_record(
                record,
                head_path=head_path,
                neuron_path=neuron_path,
                head_input_mode=str(head_input_mode),
                max_head_input_tokens=max_head_input_tokens,
            ):
                continue
            row = next(item for item_key, item in row_pairs if item_key == key)
            records_by_key[key] = _make_prediction_record(
                when2call=when2call,
                row=row,
                key=key,
                head_logits=[float(x) for x in record.get("head_logits", [])],
                head_probs=[float(x) for x in record["head_probs"]],
                decision_threshold=float(decision_threshold),
                behavior_to_id=behavior_to_id,
                head_path=head_path,
                neuron_path=neuron_path,
                head_input_mode=str(head_input_mode),
                max_head_input_tokens=max_head_input_tokens,
            )
        if records_by_key:
            print(f"[reuse] resolution scores: {len(records_by_key)}/{len(row_pairs)} from {rel(predictions_path)}", flush=True)

    missing_pairs = [(key, row) for key, row in row_pairs if key not in records_by_key]
    if missing_pairs:
        with NeuronHeadScorer(
            task="resolution",
            model_path=args.model_path,
            head_checkpoint_path=head_path,
            neuron_path=neuron_path,
            model_family=args.model_family,
            thinking_mode=args.thinking_mode,
            trust_remote_code=bool(args.trust_remote_code),
            attn_implementation=args.attn_implementation,
            prefer_unsloth_mirror=bool(args.prefer_unsloth_mirror),
            load_in_4bit=bool(args.load_in_4bit),
            load_in_8bit=bool(args.load_in_8bit),
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            dtype=args.dtype,
            max_seq_len=int(args.max_seq_len),
            max_pixels=int(args.max_pixels),
            head_input_mode=str(head_input_mode),
            max_head_input_tokens=max_head_input_tokens,
        ) as scorer:
            for chunk in tqdm(list(_chunked(missing_pairs, int(args.batch_size))), desc="score neuron resolution", dynamic_ncols=True):
                messages_batch = [_messages_for_row(when2call, row) for _key, row in chunk]
                score_batch = scorer.score_messages(messages_batch)
                probs = score_batch.probs
                logits = score_batch.logits
                for i, (key, row) in enumerate(chunk):
                    records_by_key[key] = _make_prediction_record(
                        when2call=when2call,
                        row=row,
                        key=key,
                        head_logits=[float(x) for x in logits[i].tolist()],
                        head_probs=[float(x) for x in probs[i].tolist()],
                        decision_threshold=float(decision_threshold),
                        behavior_to_id=behavior_to_id,
                        head_path=head_path,
                        neuron_path=neuron_path,
                        head_input_mode=str(head_input_mode),
                        max_head_input_tokens=max_head_input_tokens,
                    )
                ordered_partial = [records_by_key[key] for key, _row in row_pairs if key in records_by_key]
                write_jsonl(predictions_path, ordered_partial)
                print(f"[write] {rel(predictions_path)} rows={len(records_by_key)}/{len(row_pairs)}", flush=True)
    else:
        print(f"[skip] resolution scores already complete: {rel(predictions_path)}", flush=True)

    records = [records_by_key[key] for key, _row in row_pairs]

    if PARALLEL.enabled:
        PARALLEL.barrier()
        if not PARALLEL.is_main:
            PARALLEL.barrier()
            return
        _merge_parallel_eval(
            args=args,
            when2call=when2call,
            output_dir=base_output_dir,
            work_root=work_root,
            rows=all_rows,
            row_stats=row_stats,
            head_class_names=head_class_names,
            behavior_class_names=behavior_class_names,
            behavior_to_id=behavior_to_id,
            decision_threshold=decision_threshold,
        )
        PARALLEL.barrier()
        return

    gold_ids: list[int] = []
    pred_ids: list[int] = []
    pred_score_correct: list[float] = []
    pred_score_wrong: list[float] = []
    gold_score_correct: list[float] = []
    gold_score_wrong: list[float] = []
    prob_one_vs_rest: dict[str, list[float]] = {name: [] for name in behavior_class_names}
    per_pred_class_scores: dict[str, dict[str, list[float]]] = {
        name: {"correct": [], "wrong": []} for name in behavior_class_names
    }
    for record in records:
        gold_name = when2call.normalize_class_name(record["gold_label"])
        pred_name = when2call.normalize_class_name(record["head_pred_label"])
        gold_id = behavior_to_id[gold_name]
        pred_id = behavior_to_id[pred_name]
        pred_score = float(record["head_pred_score"])
        gold_score = float(record["head_gold_behavior_score"])
        is_correct = int(pred_id == gold_id)
        gold_ids.append(gold_id)
        pred_ids.append(pred_id)
        if is_correct:
            pred_score_correct.append(pred_score)
            gold_score_correct.append(gold_score)
            per_pred_class_scores[pred_name]["correct"].append(pred_score)
        else:
            pred_score_wrong.append(pred_score)
            gold_score_wrong.append(gold_score)
            per_pred_class_scores[pred_name]["wrong"].append(pred_score)
        for cls_name in behavior_class_names:
            prob_one_vs_rest[cls_name].append(float(record["behavior_scores"][cls_name]))

    head_metrics = when2call.confusion_and_metrics(gold_ids, pred_ids, behavior_class_names)
    metrics = {
        "num_rows_scored": len(rows),
        "head_class_names": head_class_names,
        "behavior_class_names": behavior_class_names,
        "row_filter_stats": row_stats,
        "decision_threshold": float(decision_threshold),
        "head_on_generated_completion": head_metrics,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", records)

    prob_summary = {
        "predicted_behavior_score_correct_mean": _safe_mean(pred_score_correct),
        "predicted_behavior_score_wrong_mean": _safe_mean(pred_score_wrong),
        "gold_behavior_score_correct_mean": _safe_mean(gold_score_correct),
        "gold_behavior_score_wrong_mean": _safe_mean(gold_score_wrong),
        "per_predicted_class": {},
    }
    for cls_name in behavior_class_names:
        correct_vals = per_pred_class_scores[cls_name]["correct"]
        wrong_vals = per_pred_class_scores[cls_name]["wrong"]
        prob_summary["per_predicted_class"][cls_name] = {
            "num_correct_predictions": len(correct_vals),
            "num_wrong_predictions": len(wrong_vals),
            "mean_prob_when_correct": _safe_mean(correct_vals),
            "mean_prob_when_wrong": _safe_mean(wrong_vals),
        }
    write_json(output_dir / "probability_summary.json", prob_summary)

    _write_probability_plots(
        when2call=when2call,
        plots_dir=plots_dir,
        metrics=metrics,
        behavior_class_names=behavior_class_names,
        pred_score_correct=pred_score_correct,
        pred_score_wrong=pred_score_wrong,
        gold_score_correct=gold_score_correct,
        gold_score_wrong=gold_score_wrong,
        per_pred_class_scores=per_pred_class_scores,
        prob_one_vs_rest=prob_one_vs_rest,
    )

    comparison_rows = _comparison_rows(args.model_path, head_metrics)
    write_json(
        output_dir / "paper_table3_comparison.json",
        {"rows": comparison_rows, "paper_source": "Multi-Head Latent Control Table 3"},
    )
    write_csv(output_dir / "paper_table3_comparison.csv", comparison_rows)

    summary = {
        "benchmark": "when2call",
        "model_path": args.model_path,
        "num_rows": len(rows),
        "metrics": metrics,
        "probability_summary": prob_summary,
        "paper_table3_comparison": comparison_rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    _print_table(comparison_rows)
    print(f"[done] saved Resolution When2Call eval to {rel(output_dir)}", flush=True)


if __name__ == "__main__":
    main()
