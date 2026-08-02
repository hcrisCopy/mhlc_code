#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from tqdm.auto import tqdm

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel
from mhlc_neuron_probe.io_utils import model_tag, output_dirs, read_jsonl, write_csv, write_json, write_jsonl
from mhlc_neuron_probe.paper_baselines import format_float, format_int, resolve_table4_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the neuron Capability Head on TriviaQA Table-4 style tool decisions.")
    parser.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--data-root", default="../mhlc_data")
    parser.add_argument("--head-checkpoint-path", default=None)
    parser.add_argument("--neuron-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-path", default="../mhlc_data/data/benchmarks/triviaqa/dataset")
    parser.add_argument("--allow-hf-fallback", action="store_true")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95")
    parser.add_argument("--report-threshold", type=float, default=None)
    parser.add_argument("--call-tool-if-missing-final-answer", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--head-input-mode", default="completion_text_only")
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-max-model-len", type=int, default=32000)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--reuse-generations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _parse_thresholds(text: str) -> list[float]:
    values = []
    for part in str(text or "").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise ValueError("No thresholds were provided.")
    return sorted(set(values))


def _chunked(items: Sequence[Any], batch_size: int):
    for start in range(0, len(items), max(1, int(batch_size))):
        yield items[start:start + max(1, int(batch_size))]


def _complete_by_idx(path: Path, examples: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    from mhlc_neuron_probe.eval_utils import load_jsonl_by_sample_idx

    rows = load_jsonl_by_sample_idx(path)
    expected = {int(ex.get("sample_idx", i)) for i, ex in enumerate(examples)}
    return {idx: row for idx, row in rows.items() if idx in expected}


def _ensure_generation_rows(
    *,
    trivia: Any,
    examples: list[dict[str, Any]],
    orchestrator: Any,
    prompt_variant: str,
    path: Path,
    batch_size: int,
    reuse: bool,
    debug: bool,
) -> list[dict[str, Any]]:
    from mhlc_neuron_probe.eval_utils import ordered_rows_from_index

    rows_by_idx = _complete_by_idx(path, examples) if reuse else {}
    if len(rows_by_idx) == len(examples):
        print(f"[skip] {prompt_variant} generations already complete: {rel(path)}", flush=True)
        return ordered_rows_from_index(rows_by_idx, examples)

    missing = [ex for ex in examples if int(ex.get("sample_idx", 0)) not in rows_by_idx]
    for chunk in tqdm(list(_chunked(missing, batch_size)), desc=f"{prompt_variant} resume", dynamic_ncols=True):
        new_rows = trivia.run_generation_pass(
            examples=list(chunk),
            orchestrator=orchestrator,
            prompt_variant=prompt_variant,
            batch_size=int(batch_size),
            debug=bool(debug),
        )
        for row in new_rows:
            rows_by_idx[int(row["sample_idx"])] = row
        write_jsonl(path, ordered_rows_from_index(rows_by_idx, examples))
        print(f"[write] {rel(path)} rows={len(rows_by_idx)}/{len(examples)}", flush=True)
    return ordered_rows_from_index(rows_by_idx, examples)


def _messages_for_aux(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        messages = [{"role": "user", "content": [{"type": "text", "text": str(row.get("prompt_text") or row.get("question") or "")}]}]
    return list(messages) + [{"role": "assistant", "content": str(row.get("raw_response") or "")}]


def _score_standard_rows(
    *,
    trivia: Any,
    examples: list[dict[str, Any]],
    standard_rows_raw: list[dict[str, Any]],
    scorer: NeuronHeadScorer,
    path: Path,
    batch_size: int,
    reuse: bool,
) -> list[dict[str, Any]]:
    from mhlc_neuron_probe.eval_utils import ordered_rows_from_index

    scored_by_idx = _complete_by_idx(path, examples) if reuse else {}
    if len(scored_by_idx) == len(examples):
        print(f"[skip] neuron-head scores already complete: {rel(path)}", flush=True)
        return ordered_rows_from_index(scored_by_idx, examples)

    raw_by_idx = {int(row["sample_idx"]): row for row in standard_rows_raw}
    missing_examples = [ex for ex in examples if int(ex.get("sample_idx", 0)) not in scored_by_idx]
    for chunk in tqdm(list(_chunked(missing_examples, batch_size)), desc="score neuron capability", dynamic_ncols=True):
        raw_chunk = [raw_by_idx[int(ex["sample_idx"])] for ex in chunk]
        messages_batch = [_messages_for_aux(row) for row in raw_chunk]
        score_batch = scorer.score_messages(messages_batch)
        probs = score_batch.probs.squeeze(-1).tolist()
        new_rows: list[dict[str, Any]] = []
        for row, prob in zip(raw_chunk, probs):
            evaluated = trivia.evaluate_saved_row("triviaqa", row, judge_runtime=None, judge_sampling=None)
            info = trivia.response_final_answer_status("triviaqa", row.get("raw_response", ""))
            merged = {
                **evaluated,
                "base_has_final_answer": bool(info.get("has_final_answer", False)),
                "base_final_answer_reason": str(info.get("reason", "unknown")),
                "aux_enabled_for_run": True,
                "aux_prob_correct": float(prob),
                "aux_pred": int(float(prob) >= 0.5),
                "base_correct": 0 if evaluated.get("judge_label") is None else int(evaluated.get("judge_label", 0)),
            }
            scored_by_idx[int(row["sample_idx"])] = merged
            new_rows.append(merged)
        write_jsonl(path, ordered_rows_from_index(scored_by_idx, examples))
        print(f"[write] {rel(path)} rows={len(scored_by_idx)}/{len(examples)}", flush=True)
    return ordered_rows_from_index(scored_by_idx, examples)


def _precision_pct(needed_calls: int, calls: int) -> float:
    return 100.0 * float(needed_calls) / max(int(calls), 1)


def _table_row_from_threshold(no_tool_score: float, chosen: dict[str, Any]) -> dict[str, Any]:
    calls = int(chosen.get("num_aux_tool_calls", 0) or 0)
    needed = int(chosen.get("num_aux_potentially_necessary_tool_calls", 0) or 0)
    return {
        "system": "Backbone + Capability Head（Ours）",
        "no_tool": float(no_tool_score),
        "score": float(chosen.get("score_ours", 0.0) or 0.0),
        "calls": calls,
        "precision_pct": _precision_pct(needed, calls),
        "missed": int(chosen.get("num_aux_missed_incorrect_without_tool", 0) or 0),
        "threshold": float(chosen.get("threshold", 0.0) or 0.0),
    }


def _paper_table_rows(model_path: str, ours: dict[str, Any]) -> list[dict[str, Any]]:
    paper = resolve_table4_row(model_path)
    return [
        {"system": "Backbone Choice", **paper["backbone_choice"], "threshold": None},
        {"system": "Backbone + Capability Head（MHLC）", **paper["mhlc"], "threshold": None},
        ours,
    ]


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\nTriviaQA Table 4 style comparison", flush=True)
    print("| System | No-Tool ↑ | Score ↑ | Calls | Precision (%) ↑ | Missed ↓ |", flush=True)
    print("|---|---:|---:|---:|---:|---:|", flush=True)
    for row in rows:
        print(
            "| {system} | {no_tool} | {score} | {calls} | {precision} | {missed} |".format(
                system=row["system"],
                no_tool=format_float(row.get("no_tool"), 3),
                score=format_float(row.get("score"), 3),
                calls=format_int(row.get("calls")),
                precision=format_float(row.get("precision_pct"), 1),
                missed=format_int(row.get("missed")),
            ),
            flush=True,
        )


def _plot_threshold_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [float(row["threshold"]) for row in rows]
    scores = [float(row["score_ours"]) for row in rows]
    calls = [int(row["num_aux_tool_calls"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(thresholds, scores, marker="o", label="Score")
    ax1.set_xlabel("Capability threshold")
    ax1.set_ylabel("Score")
    ax2 = ax1.twinx()
    ax2.plot(thresholds, calls, marker="s", color="tab:orange", label="Calls")
    ax2.set_ylabel("Tool calls")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    tag = model_tag(args.model_path)
    dirs = output_dirs(data_root, tag, "capability")
    head_path = dirs["trained"] / "neuron_head_final.pt" if args.head_checkpoint_path is None else resolve_from_code_root(args.head_checkpoint_path)
    neuron_path = dirs["neurons"] / "selected_neurons.jsonl" if args.neuron_path is None else resolve_from_code_root(args.neuron_path)
    output_dir = data_root / "eval_outputs" / "neuron_heads" / tag / "capability_triviaqa" if args.output_dir is None else resolve_from_code_root(args.output_dir)
    dataset_path = resolve_from_code_root(args.dataset_path)

    if args.clean:
        clean_path(output_dir, [data_root], "capability triviaqa eval output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists() and not args.allow_hf_fallback:
        raise FileNotFoundError(
            f"Missing local TriviaQA benchmark dataset: {rel(dataset_path)}\n"
            "Run: python src/01_download_data.py --group benchmarks --benchmarks triviaqa"
        )
    if not head_path.exists():
        raise FileNotFoundError(f"Missing neuron capability head checkpoint: {rel(head_path)}")
    if not neuron_path.exists():
        raise FileNotFoundError(f"Missing selected neurons: {rel(neuron_path)}")

    from mhlc_neuron_probe.eval_utils import NeuronHeadScorer

    trivia = load_upstream_module(
        "multi_agenT_bench/triviaqa_web_overuse_eval.py",
        "mhlc_upstream_triviaqa_web_overuse_eval_for_neurons",
    )
    thresholds = _parse_thresholds(args.thresholds)
    dataset_cfg = {
        "data_mode": "hf" if args.allow_hf_fallback and not dataset_path.exists() else "disk",
        "dataset_name": "mandarjoshi/trivia_qa",
        "dataset_config_name": "rc",
        "split": "validation",
        "data_path": rel(dataset_path),
        "max_samples": int(args.max_samples),
    }
    write_json(
        output_dir / "run_config.json",
        {
            "model_path": args.model_path,
            "head_checkpoint_path": rel(head_path),
            "neuron_path": rel(neuron_path),
            "dataset_cfg": dataset_cfg,
            "thresholds": thresholds,
            "report_threshold": args.report_threshold,
            "call_tool_if_missing_final_answer": bool(args.call_tool_if_missing_final_answer),
        },
    )

    examples = trivia.load_examples_for_benchmark("triviaqa", dataset_cfg)
    write_json(output_dir / "dataset_summary.json", {"benchmark": "triviaqa", "num_examples": len(examples)})

    standard_raw_path = output_dir / "standard_no_tool_rows_raw_generation.jsonl"
    tool_raw_path = output_dir / "tool_enabled_rows_raw_generation.jsonl"
    gen_args = SimpleNamespace(
        model_name_or_path=args.model_path,
        model_family=args.model_family,
        thinking_mode=args.thinking_mode,
        aux_head_ckpt="",
        debug=bool(args.debug),
    )
    trivia.DEFAULT_VLLM_RUNTIME["dtype"] = args.vllm_dtype
    trivia.DEFAULT_VLLM_RUNTIME["max_model_len"] = int(args.vllm_max_model_len)
    trivia.DEFAULT_VLLM_RUNTIME["gpu_memory_utilization"] = float(args.vllm_gpu_memory_utilization)
    trivia.DEFAULT_VLLM_RUNTIME["max_num_seqs"] = int(args.vllm_max_num_seqs)
    generation_bundle = trivia.build_model_bundle_single(gen_args, vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization))
    generation_orchestrator = trivia.MultiAgentOrchestrator(
        benchmark="triviaqa",
        model_bundles={"model1": generation_bundle},
        debug_mode=bool(args.debug),
        debug_max_chars=2000,
    )
    try:
        standard_rows_raw = _ensure_generation_rows(
            trivia=trivia,
            examples=examples,
            orchestrator=generation_orchestrator,
            prompt_variant="standard_no_tool",
            path=standard_raw_path,
            batch_size=int(args.generation_batch_size),
            reuse=bool(args.reuse_generations),
            debug=bool(args.debug),
        )
        tool_rows_raw = _ensure_generation_rows(
            trivia=trivia,
            examples=examples,
            orchestrator=generation_orchestrator,
            prompt_variant="tool_enabled",
            path=tool_raw_path,
            batch_size=int(args.generation_batch_size),
            reuse=bool(args.reuse_generations),
            debug=bool(args.debug),
        )
    finally:
        generation_orchestrator.unload_all(drop_processors=False)
        trivia.release_memory()

    trivia.save_rows_to_parquet(output_dir / "standard_no_tool_rows_raw_generation.parquet", standard_rows_raw, debug=bool(args.debug))
    trivia.save_rows_to_parquet(output_dir / "tool_enabled_rows_raw_generation.parquet", tool_rows_raw, debug=bool(args.debug))

    scored_path = output_dir / "standard_no_tool_rows_scored_with_neuron_head.jsonl"
    with NeuronHeadScorer(
        task="capability",
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
        head_input_mode=args.head_input_mode,
    ) as scorer:
        standard_rows = _score_standard_rows(
            trivia=trivia,
            examples=examples,
            standard_rows_raw=standard_rows_raw,
            scorer=scorer,
            path=scored_path,
            batch_size=int(args.score_batch_size),
            reuse=bool(args.reuse_scores),
        )

    tool_rows = trivia.evaluate_tool_enabled_rows(base_rows=standard_rows, tool_rows=tool_rows_raw)
    write_jsonl(output_dir / "tool_enabled_rows.jsonl", tool_rows)
    trivia.save_rows_to_parquet(output_dir / "standard_no_tool_rows_scored_with_neuron_head.parquet", standard_rows, debug=bool(args.debug))
    trivia.save_rows_to_parquet(output_dir / "tool_enabled_rows.parquet", tool_rows, debug=bool(args.debug))

    y_true = [int(row.get("base_correct", 0)) for row in standard_rows]
    y_prob = [float(row.get("aux_prob_correct")) for row in standard_rows if row.get("aux_prob_correct") is not None]
    aux_metrics = trivia.compute_aux_binary_metrics(y_true, y_prob) if len(y_true) == len(y_prob) and y_prob else None
    no_head_summary = trivia.compute_no_head_summary(base_rows=standard_rows, tool_rows=tool_rows)
    no_tool_score = float(sum(y_true) / max(len(y_true), 1))
    no_head_summary["score_standard_no_tool"] = no_tool_score

    threshold_summaries = [
        trivia.compute_threshold_summary(
            threshold=threshold,
            base_rows=standard_rows,
            no_head_summary=no_head_summary,
            call_tool_if_missing_final_answer=bool(args.call_tool_if_missing_final_answer),
        )
        for threshold in thresholds
    ]
    if args.report_threshold is not None:
        chosen = next((row for row in threshold_summaries if math.isclose(float(row["threshold"]), float(args.report_threshold), abs_tol=1.0e-8)), None)
        if chosen is None:
            raise ValueError(f"--report-threshold {args.report_threshold} is not in --thresholds {thresholds}")
        chosen = dict(chosen)
        chosen["selection_rule"] = "requested_report_threshold"
    else:
        chosen = trivia.select_best_threshold(threshold_summaries, no_head_summary)

    threshold_rows_flat = [{k: v for k, v in row.items() if k != "per_example"} for row in threshold_summaries]
    threshold_decision_rows = []
    for row in threshold_summaries:
        threshold_decision_rows.extend(list(row.get("per_example", [])))

    merged_rows = []
    for base_row, tool_row in zip(standard_rows, tool_rows):
        merged_rows.append({
            "sample_idx": base_row.get("sample_idx"),
            "id": base_row.get("id"),
            "question": base_row.get("question"),
            "gold_answer": base_row.get("gold_answer"),
            "standard_raw_response": base_row.get("raw_response"),
            "standard_correct": int(base_row.get("base_correct", 0)),
            "standard_has_final_answer": int(base_row.get("base_has_final_answer", False)),
            "aux_prob_correct": base_row.get("aux_prob_correct"),
            "tool_enabled_raw_response": tool_row.get("raw_response"),
            "tool_called_by_model": int(tool_row.get("tool_called_by_model", 0)),
            "tool_query": tool_row.get("tool_query"),
            "model_selftool_simulated_correct": int(tool_row.get("model_selftool_simulated_correct", 0)),
            "model_selftool_unnecessary_tool_call": int(tool_row.get("model_selftool_unnecessary_tool_call", 0)),
            "model_selftool_missed_incorrect_without_tool": int(tool_row.get("model_selftool_missed_incorrect_without_tool", 0)),
        })

    write_jsonl(output_dir / "merged_per_example_rows.jsonl", merged_rows)
    write_jsonl(output_dir / "threshold_decisions_per_example.jsonl", threshold_decision_rows)
    write_csv(output_dir / "threshold_summary.csv", threshold_rows_flat)
    write_csv(output_dir / "merged_per_example_rows.csv", merged_rows)
    write_csv(output_dir / "threshold_decisions_per_example.csv", threshold_decision_rows)
    _plot_threshold_sweep(output_dir / "plots" / "threshold_score_calls.png", threshold_rows_flat)

    ours_row = _table_row_from_threshold(no_tool_score, chosen or {})
    table_rows = _paper_table_rows(args.model_path, ours_row)
    write_json(output_dir / "paper_table4_comparison.json", {"rows": table_rows, "paper_source": "Multi-Head Latent Control Table 4"})
    write_csv(output_dir / "paper_table4_comparison.csv", table_rows)
    summary = {
        "benchmark": "triviaqa",
        "model_path": args.model_path,
        "num_rows": len(examples),
        "aux_metrics": aux_metrics,
        "no_head_summary": no_head_summary,
        "threshold_summaries": threshold_rows_flat,
        "chosen_threshold_summary": {k: v for k, v in (chosen or {}).items() if k != "per_example"},
        "paper_table4_comparison": table_rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    _print_table(table_rows)
    print(f"[done] saved Capability TriviaQA eval to {rel(output_dir)}", flush=True)


if __name__ == "__main__":
    main()
