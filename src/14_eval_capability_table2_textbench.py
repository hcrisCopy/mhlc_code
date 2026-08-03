#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel
from mhlc_neuron_probe.io_utils import model_tag, output_dirs, read_jsonl, write_csv, write_json, write_jsonl
from mhlc_neuron_probe.paper_baselines import format_float


TEXT_BENCHMARKS = ("triviaqa", "math", "mmlu_pro")
DEFAULT_BASELINE_HEAD = (
    "../mhlc_data/trained_models/baseline_capability_heads/"
    "Qwen__Qwen3-VL-4B-Instruct/full/capability_head.pt"
)
DEFAULT_JUDGE_MODEL = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MHLC and neuron Capability Heads on Table-2 text benchmarks.")
    parser.add_argument("--model1-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model2-path", default="../Qwen/Qwen3-VL-32B-Thinking-FP8")
    parser.add_argument("--data-root", default="../mhlc_data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--benchmarks", default="triviaqa,math,mmlu_pro")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--triviaqa-dataset-path", default="../mhlc_data/data/benchmarks/triviaqa/dataset")
    parser.add_argument("--math-csv-path", default="../mhlc_data/data/benchmarks/merged_math.csv")
    parser.add_argument("--mmlu-pro-csv-path", default="../mhlc_data/data/benchmarks/test.csv")
    parser.add_argument("--baseline-head-path", default=DEFAULT_BASELINE_HEAD)
    parser.add_argument("--ours-head-checkpoint-path", default=None)
    parser.add_argument("--ours-neuron-path", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated routing thresholds. Defaults to the original Table 2 grid: 0.5,0.6,0.7,0.8,0.9.",
    )
    parser.add_argument("--call-model2-if-missing-final-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model1-family", default="qwen3_vl", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    parser.add_argument("--model1-thinking-mode", default="off", choices=["auto", "on", "off"])
    parser.add_argument("--model2-family", default="qwen3_vl", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    parser.add_argument("--model2-thinking-mode", default="on", choices=["auto", "on", "off"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--prefer-unsloth-mirror", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--use-gradient-checkpointing", default="unsloth")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--max-seq-len", type=int, default=32000)
    parser.add_argument("--max-pixels", type=int, default=200000)
    parser.add_argument("--ours-head-input-mode", default="completion_text_only")
    parser.add_argument("--baseline-head-input-mode", default=None)
    parser.add_argument("--baseline-hidden-layer-selection", default="last")
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--vllm-max-model-len", type=int, default=32000)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--model1-gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--model2-gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--model1-max-num-seqs", type=int, default=32)
    parser.add_argument("--model2-max-num-seqs", type=int, default=16)
    parser.add_argument("--model1-generation-batch-size", type=int, default=8)
    parser.add_argument("--model2-generation-batch-size", type=int, default=4)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--m2-input-cost-per-1m-usd", type=float, default=0.70)
    parser.add_argument("--m2-output-cost-per-1m-usd", type=float, default=8.40)
    parser.add_argument(
        "--judge-model-path",
        default=DEFAULT_JUDGE_MODEL,
        help="Optional judge model for textbench fallback grading. Empty string disables judge loading.",
    )
    parser.add_argument("--judge-model-family", default="auto", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4", "other"])
    parser.add_argument("--judge-thinking-mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--judge-batch-size", type=int, default=16)
    parser.add_argument("--reuse-generations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-evaluations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _parse_benchmarks(text: str) -> list[str]:
    values = [part.strip().lower() for part in str(text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("No benchmarks provided.")
    bad = [value for value in values if value not in TEXT_BENCHMARKS]
    if bad:
        raise ValueError(f"Unsupported text benchmarks: {bad}. Supported: {TEXT_BENCHMARKS}")
    return values


def _parse_thresholds(thresholds_text: str | None, single_threshold: float | None) -> list[float]:
    text = str(thresholds_text or "").strip()
    if text:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    elif single_threshold is not None:
        values = [float(single_threshold)]
    else:
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
    if not values:
        raise ValueError("No routing thresholds provided.")
    bad = [value for value in values if value < 0.0 or value > 1.0]
    if bad:
        raise ValueError(f"Thresholds must be in [0, 1], got: {bad}")
    return sorted(set(float(value) for value in values))


def _threshold_tag(threshold: float) -> str:
    return f"{float(threshold):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _strategy_dir(bench_dir: Path, strategy_name: str, threshold: float, *, multi_threshold: bool) -> Path:
    if not multi_threshold:
        return bench_dir / strategy_name
    return bench_dir / f"{strategy_name}_threshold_{_threshold_tag(threshold)}"


def _chunked(items: Sequence[Any], batch_size: int):
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _score_key(row: dict[str, Any], fallback_index: int) -> str:
    for key in ("sample_idx", "dataset_index", "id", "example_id", "question_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(fallback_index)


def _rows_by_key(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {_score_key(row, idx): row for idx, row in enumerate(read_jsonl(path))}


def _ordered_rows(rows_by_key: dict[str, dict[str, Any]], examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        key = _score_key(ex, idx)
        if key in rows_by_key:
            out.append(rows_by_key[key])
    return out


def _dataset_cfgs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "triviaqa": {
            "data_mode": "disk",
            "data_path": rel(resolve_from_code_root(args.triviaqa_dataset_path)),
            "dataset_name": "mandarjoshi/trivia_qa",
            "dataset_config_name": "rc",
            "split": "validation",
            "max_samples": int(args.max_samples),
        },
        "math": {
            "data_mode": "csv",
            "data_path": rel(resolve_from_code_root(args.math_csv_path)),
            "max_samples": int(args.max_samples),
        },
        "mmlu_pro": {
            "data_mode": "csv",
            "data_path": rel(resolve_from_code_root(args.mmlu_pro_csv_path)),
            "max_samples": int(args.max_samples),
        },
    }


def _validate_benchmark_inputs(args: argparse.Namespace, benchmarks: Sequence[str]) -> None:
    paths = {
        "triviaqa": resolve_from_code_root(args.triviaqa_dataset_path),
        "math": resolve_from_code_root(args.math_csv_path),
        "mmlu_pro": resolve_from_code_root(args.mmlu_pro_csv_path),
    }
    for benchmark in benchmarks:
        if not paths[benchmark].exists():
            raise FileNotFoundError(f"Missing {benchmark} benchmark data: {rel(paths[benchmark])}")


def _candidate_head_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists() or not path.is_dir():
        return []
    preferred = [
        path / "capability_head.pt",
        path / "aux_head_final.pt",
    ]
    files = [p for p in preferred if p.exists()]
    seen = {p.resolve() for p in files}
    for pattern in ("capability_head.pt", "aux_head_final.pt", "*.pt"):
        for p in sorted(path.rglob(pattern)):
            resolved = p.resolve()
            if resolved not in seen:
                files.append(p)
                seen.add(resolved)
    return files


def _match_score(path: Path, model_path: str) -> tuple[int, int, str]:
    haystack = str(path).replace("\\", "/").lower()
    model_name = Path(str(model_path).replace("\\", "/").rstrip("/")).name.lower()
    compact_haystack = haystack.replace("_", "").replace("-", "").replace(".", "")
    compact_model = model_name.replace("_", "").replace("-", "").replace(".", "")
    score = 0
    if compact_model and compact_model in compact_haystack:
        score += 100
    if "full" in haystack:
        score += 20
    if path.name == "capability_head.pt":
        score += 10
    if path.name == "aux_head_final.pt":
        score += 5
    return (-score, len(str(path)), str(path))


def _resolve_baseline_head_path(args: argparse.Namespace, data_root: Path) -> Path:
    requested = resolve_from_code_root(args.baseline_head_path)
    if requested.exists():
        candidates = _candidate_head_files(requested)
        if not candidates:
            raise FileNotFoundError(f"No .pt file found under baseline head path: {rel(requested)}")
        chosen = sorted(candidates, key=lambda p: _match_score(p, args.model1_path))[0]
        if chosen != requested:
            print(f"[resolve] baseline head directory/file -> {rel(chosen)}", flush=True)
        return chosen

    if requested.name == "aux_head_final.pt":
        sibling = requested.with_name("capability_head.pt")
        if sibling.exists():
            print(f"[resolve] requested aux_head_final.pt is absent; using downloaded release file: {rel(sibling)}", flush=True)
            return sibling

    search_roots = [
        data_root / "trained_models" / "baseline_capability_heads",
        resolve_from_code_root("Multi-Head-Latent-Control") / "trained_models",
    ]
    candidates: list[Path] = []
    for root in search_roots:
        candidates.extend(_candidate_head_files(root))
    if candidates:
        chosen = sorted(candidates, key=lambda p: _match_score(p, args.model1_path))[0]
        print(f"[resolve] baseline head path not found: {rel(requested)}", flush=True)
        print(f"[resolve] using discovered baseline head: {rel(chosen)}", flush=True)
        return chosen

    raise FileNotFoundError(
        "Missing original MHLC capability head. Expected a file such as:\n"
        f"  {rel(requested)}\n"
        "or the downloaded release file:\n"
        f"  {rel(requested.with_name('capability_head.pt'))}\n"
        "Download it with:\n"
        "  python src/13_download_baseline_capability_head.py "
        "--model ../Qwen/Qwen3-VL-4B-Instruct --variant full --thinking-mode off"
    )


def _runtime_profile(args: argparse.Namespace, slot: str) -> dict[str, Any]:
    if slot == "model1":
        gpu_util = float(args.model1_gpu_memory_utilization)
        max_num_seqs = int(args.model1_max_num_seqs)
        model_family = args.model1_family
        thinking_mode = args.model1_thinking_mode
    else:
        gpu_util = float(args.model2_gpu_memory_utilization)
        max_num_seqs = int(args.model2_max_num_seqs)
        model_family = args.model2_family
        thinking_mode = args.model2_thinking_mode
    return {
        "dtype": str(args.vllm_dtype),
        "max_model_len": int(args.vllm_max_model_len),
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_util,
        "max_num_seqs": max_num_seqs,
        "enforce_eager": bool(args.vllm_enforce_eager),
        "trust_remote_code": bool(args.trust_remote_code),
        "limit_mm_images": 1,
        "model_family": str(model_family),
        "thinking_mode": str(thinking_mode),
    }


def _aux_profile(args: argparse.Namespace, *, head_input_mode: str | None) -> dict[str, Any]:
    return {
        "trust_remote_code": bool(args.trust_remote_code),
        "prefer_unsloth_mirror": bool(args.prefer_unsloth_mirror),
        "dtype": str(args.dtype),
        "max_seq_len": int(args.max_seq_len),
        "max_pixels": int(args.max_pixels),
        "attn_implementation": str(args.attn_implementation),
        "regression_threshold": float(args.threshold),
        "head_input_mode": head_input_mode or "completion_text_only",
        "hidden_layer_selection": str(args.baseline_hidden_layer_selection),
        "hidden_layer_index": None,
        "hidden_layer_indices": None,
        "model_family": str(args.model1_family),
        "thinking_mode": str(args.model1_thinking_mode),
    }


def _build_generation_bundle(shared: Any, generate_mod: Any, args: argparse.Namespace, slot: str, benchmark: str):
    if slot == "model1":
        model_path = args.model1_path
        family = args.model1_family
        thinking = args.model1_thinking_mode
    else:
        model_path = args.model2_path
        family = args.model2_family
        thinking = args.model2_thinking_mode
    sampling_override = generate_mod._official_sampling_override_for_model(
        model_name_or_path=model_path,
        model_family=family,
        thinking_mode=thinking,
        benchmark=benchmark,
    )
    if args.generation_max_new_tokens is not None:
        sampling_override = dict(sampling_override or {})
        sampling_override["max_new_tokens"] = int(args.generation_max_new_tokens)
    return shared.build_model_bundle(
        model_name_or_path=model_path,
        aux_head_ckpt="",
        runtime_profile=_runtime_profile(args, slot),
        sampling_profiles=generate_mod.SAMPLING_PROFILES,
        aux_profile=_aux_profile(args, head_input_mode="completion_text_only"),
        sampling_override=sampling_override,
        model_family=family,
        thinking_mode=thinking,
    )


def _release_accelerator_memory(label: str) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    print(f"[memory] released after {label}", flush=True)


def _usage_template() -> dict[str, dict[str, Any]]:
    keys = ("prompt_tokens", "completion_tokens", "aux_scored_tokens", "aux_calls", "generation_calls", "generation_time_sec")
    return {
        "model1": {key: 0 for key in keys},
        "model2": {key: 0 for key in keys},
    }


def _add_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "aux_scored_tokens", "aux_calls", "generation_calls"):
        target[key] = int(target.get(key, 0) or 0) + int(source.get(key, 0) or 0)
    target["generation_time_sec"] = float(target.get("generation_time_sec", 0.0) or 0.0) + float(source.get("generation_time_sec", 0.0) or 0.0)


def _generate_single_agent_rows(
    *,
    shared: Any,
    bundle: Any,
    benchmark: str,
    examples: list[dict[str, Any]],
    slot: str,
    strategy_name: str,
    path: Path,
    batch_size: int,
    reuse: bool,
) -> list[dict[str, Any]]:
    rows_by_key = _rows_by_key(path) if reuse else {}
    if len(rows_by_key) == len(examples):
        print(f"[skip] {benchmark}:{strategy_name} generations already complete: {rel(path)}", flush=True)
        return _ordered_rows(rows_by_key, examples)

    runtime = shared.VLLMChatRuntime(bundle.generator_cfg)
    model_key = slot
    try:
        missing = [(idx, ex) for idx, ex in enumerate(examples) if _score_key(ex, idx) not in rows_by_key]
        for chunk in tqdm(list(_chunked(missing, int(batch_size))), desc=f"{benchmark}:{strategy_name}", dynamic_ncols=True):
            indices = [idx for idx, _ex in chunk]
            chunk_examples = [ex for _idx, ex in chunk]
            messages_list = [shared.build_initial_messages(benchmark, ex) for ex in chunk_examples]
            images = [shared.get_example_image_for_benchmark(ex) for ex in chunk_examples]
            t0 = time.time()
            gens = runtime.generate_batch(
                messages_list=messages_list,
                images=images,
                sampling_cfg=bundle.sampling_cfg,
                continue_final_messages=[False] * len(chunk_examples),
            )
            elapsed = float(time.time() - t0)
            for idx, ex, gen in zip(indices, chunk_examples, gens):
                usage_by_model = _usage_template()
                usage_by_model[model_key]["prompt_tokens"] += int(gen.prompt_tokens)
                usage_by_model[model_key]["completion_tokens"] += int(gen.completion_tokens)
                usage_by_model[model_key]["generation_calls"] += 1
                usage_by_model[model_key]["generation_time_sec"] += float(gen.generation_time_sec)
                usage_objects = {
                    "model1": shared.TokenUsage(**usage_by_model["model1"]),
                    "model2": shared.TokenUsage(**usage_by_model["model2"]),
                }
                trace = [{
                    "event": "full_generation",
                    "model": model_key,
                    "completion_tokens": int(gen.completion_tokens),
                    "generation_time_sec": float(gen.generation_time_sec),
                }]
                row = shared.build_generation_row(
                    benchmark=benchmark,
                    ex=ex,
                    strategy_name=strategy_name,
                    final_model_name=model_key,
                    final_response=gen.text,
                    usage_by_model=usage_objects,
                    trace=trace,
                    wall_time_sec=elapsed / max(len(chunk_examples), 1),
                )
                rows_by_key[_score_key(ex, idx)] = row
            write_jsonl(path, _ordered_rows(rows_by_key, examples))
            print(f"[write] {rel(path)} rows={len(rows_by_key)}/{len(examples)}", flush=True)
    finally:
        runtime.unload(drop_processor=False)
        _release_accelerator_memory(f"{benchmark}:{strategy_name}")
    return _ordered_rows(rows_by_key, examples)


def _messages_for_scoring(shared: Any, benchmark: str, ex: dict[str, Any], response_text: str) -> list[dict[str, Any]]:
    return shared.build_initial_messages(benchmark, ex) + [{"role": "assistant", "content": str(response_text)}]


def _score_mhlc_head(
    *,
    shared: Any,
    args: argparse.Namespace,
    benchmark: str,
    examples: list[dict[str, Any]],
    m1_rows: list[dict[str, Any]],
    baseline_head_path: Path,
    out_path: Path,
    reuse: bool,
) -> list[dict[str, Any]]:
    scores_by_key = _rows_by_key(out_path) if reuse else {}
    valid_by_key = {
        key: row
        for key, row in scores_by_key.items()
        if row.get("head_type") == "mhlc_hidden_states" and row.get("head_checkpoint_path") == rel(baseline_head_path)
    }
    if len(valid_by_key) == len(examples):
        print(f"[skip] {benchmark}: MHLC head scores already complete: {rel(out_path)}", flush=True)
        return _ordered_rows(valid_by_key, examples)

    cfg = shared.AuxHeadRuntimeConfig(
        enabled=True,
        model_name_or_path=str(args.model1_path),
        aux_head_ckpt=str(baseline_head_path),
        trust_remote_code=bool(args.trust_remote_code),
        prefer_unsloth_mirror=bool(args.prefer_unsloth_mirror),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        use_gradient_checkpointing=str(args.use_gradient_checkpointing),
        dtype=str(args.dtype),
        max_seq_len=int(args.max_seq_len),
        max_pixels=int(args.max_pixels),
        attn_implementation=str(args.attn_implementation),
        regression_threshold=float(args.threshold),
        head_input_mode=args.baseline_head_input_mode,
        hidden_layer_selection=str(args.baseline_hidden_layer_selection),
        model_family=str(args.model1_family),
        thinking_mode=str(args.model1_thinking_mode),
    )
    scorer = shared.AuxHeadRuntime(cfg)
    try:
        scorer.load()
        m1_by_key = {_score_key(row, idx): row for idx, row in enumerate(m1_rows)}
        missing = [(idx, ex) for idx, ex in enumerate(examples) if _score_key(ex, idx) not in valid_by_key]
        for idx, ex in tqdm(missing, desc=f"{benchmark}:score MHLC head", dynamic_ncols=True):
            key = _score_key(ex, idx)
            row = m1_by_key[key]
            image = shared.get_example_image_for_benchmark(ex)
            score = scorer.score_messages(
                messages=_messages_for_scoring(shared, benchmark, ex, str(row.get("raw_response", ""))),
                image=image,
            )
            valid_by_key[key] = {
                "sample_idx": ex.get("sample_idx", idx),
                "id": ex.get("id"),
                "head_type": "mhlc_hidden_states",
                "head_checkpoint_path": rel(baseline_head_path),
                "prob_correct": float(score.prob_correct),
                "pred": int(score.pred),
                "probs": [float(x) for x in score.probs],
            }
            write_jsonl(out_path, _ordered_rows(valid_by_key, examples))
    finally:
        scorer.unload(drop_processor=False)
        _release_accelerator_memory(f"{benchmark}:score MHLC head")
    return _ordered_rows(valid_by_key, examples)


def _score_ours_head(
    *,
    shared: Any,
    args: argparse.Namespace,
    benchmark: str,
    examples: list[dict[str, Any]],
    m1_rows: list[dict[str, Any]],
    head_path: Path,
    neuron_path: Path,
    out_path: Path,
    reuse: bool,
) -> list[dict[str, Any]]:
    from mhlc_neuron_probe.eval_utils import NeuronHeadScorer

    scores_by_key = _rows_by_key(out_path) if reuse else {}
    valid_by_key = {
        key: row
        for key, row in scores_by_key.items()
        if row.get("head_type") == "neuron_head" and row.get("head_checkpoint_path") == rel(head_path) and row.get("neuron_path") == rel(neuron_path)
    }
    if len(valid_by_key) == len(examples):
        print(f"[skip] {benchmark}: neuron head scores already complete: {rel(out_path)}", flush=True)
        return _ordered_rows(valid_by_key, examples)

    m1_by_key = {_score_key(row, idx): row for idx, row in enumerate(m1_rows)}
    missing = [(idx, ex) for idx, ex in enumerate(examples) if _score_key(ex, idx) not in valid_by_key]
    with NeuronHeadScorer(
        task="capability",
        model_path=args.model1_path,
        head_checkpoint_path=head_path,
        neuron_path=neuron_path,
        model_family=args.model1_family,
        thinking_mode=args.model1_thinking_mode,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
        prefer_unsloth_mirror=bool(args.prefer_unsloth_mirror),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        dtype=args.dtype,
        max_seq_len=int(args.max_seq_len),
        max_pixels=int(args.max_pixels),
        head_input_mode=args.ours_head_input_mode,
    ) as scorer:
        for chunk in tqdm(list(_chunked(missing, int(args.score_batch_size))), desc=f"{benchmark}:score neuron head", dynamic_ncols=True):
            messages_batch = [
                _messages_for_scoring(
                    shared,
                    benchmark,
                    ex,
                    str(m1_by_key[_score_key(ex, idx)].get("raw_response", "")),
                )
                for idx, ex in chunk
            ]
            score_batch = scorer.score_messages(messages_batch)
            probs = score_batch.probs.squeeze(-1).tolist()
            logits = score_batch.logits.squeeze(-1).tolist()
            for (idx, ex), prob, logit in zip(chunk, probs, logits):
                key = _score_key(ex, idx)
                valid_by_key[key] = {
                    "sample_idx": ex.get("sample_idx", idx),
                    "id": ex.get("id"),
                    "head_type": "neuron_head",
                    "head_checkpoint_path": rel(head_path),
                    "neuron_path": rel(neuron_path),
                    "prob_correct": float(prob),
                    "logit": float(logit),
                    "pred": int(float(prob) >= float(args.threshold)),
                    "probs": [float(1.0 - float(prob)), float(prob)],
                }
            write_jsonl(out_path, _ordered_rows(valid_by_key, examples))
    return _ordered_rows(valid_by_key, examples)


def _build_routed_rows(
    *,
    generate_mod: Any,
    shared: Any,
    benchmark: str,
    examples: list[dict[str, Any]],
    m1_rows: list[dict[str, Any]],
    m2_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    strategy_name: str,
    threshold: float,
    call_model2_if_missing_final_answer: bool,
) -> list[dict[str, Any]]:
    m1_by_key = {_score_key(row, idx): row for idx, row in enumerate(m1_rows)}
    m2_by_key = {_score_key(row, idx): row for idx, row in enumerate(m2_rows)}
    score_by_key = {_score_key(row, idx): row for idx, row in enumerate(score_rows)}
    routed_rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        key = _score_key(ex, idx)
        m1_row = m1_by_key[key]
        m2_row = m2_by_key[key]
        score_row = score_by_key[key]
        aux_score = float(score_row["prob_correct"])
        status = generate_mod.get_model1_first_pass_status(benchmark, str(m1_row.get("raw_response", "")), aux_score, float(threshold))
        low_aux = aux_score < float(threshold)
        missing_final = not bool(status.get("has_final_answer", False))
        use_model2 = bool(low_aux or (call_model2_if_missing_final_answer and missing_final))

        base = copy.deepcopy(m1_row)
        usage = _usage_template()
        _add_usage(usage["model1"], m1_row.get("usage_by_model", {}).get("model1", {}))
        usage["model1"]["aux_calls"] += 1
        usage["model1"]["aux_scored_tokens"] += int(m1_row.get("usage_by_model", {}).get("model1", {}).get("completion_tokens", 0) or 0)

        trace = list(m1_row.get("trace", [])) if isinstance(m1_row.get("trace"), list) else []
        trace.append({
            "event": "cached_aux_score",
            "model": "model1",
            "prob_correct": aux_score,
            "threshold": float(threshold),
            "has_final_answer": bool(status.get("has_final_answer", False)),
            "final_answer_reason": str(status.get("final_answer_reason", "unknown")),
            "routing_reason": str(status.get("reason", "unknown")),
            "decision": "handoff" if use_model2 else "accept",
            "head_type": score_row.get("head_type"),
        })

        if use_model2:
            _add_usage(usage["model2"], m2_row.get("usage_by_model", {}).get("model2", {}))
            trace.append({
                "event": "handoff",
                "from_model": "model1",
                "to_model": "model2",
                "mode": "handoff_fresh",
                "decision": "handoff",
                "routing_reason": str(status.get("reason", "unknown")),
            })
            for item in m2_row.get("trace", []):
                if isinstance(item, dict):
                    trace.append(copy.deepcopy(item))
            base["final_model_name"] = "model2"
            base["raw_response"] = m2_row.get("raw_response", "")
        else:
            base["final_model_name"] = "model1"
            base["raw_response"] = m1_row.get("raw_response", "")

        base["strategy_name"] = strategy_name
        base["usage_by_model"] = usage
        base["trace"] = trace
        base["boxed_answer"] = shared.extract_last_boxed(str(base.get("raw_response", "")))
        base["wall_time_sec"] = float(m1_row.get("wall_time_sec", 0.0) or 0.0) + (float(m2_row.get("wall_time_sec", 0.0) or 0.0) if use_model2 else 0.0)
        base["capability_prob_correct"] = aux_score
        base["capability_threshold"] = float(threshold)
        base["routed_to_model2"] = int(use_model2)
        base["routing_reason"] = str(status.get("reason", "unknown"))
        routed_rows.append(base)
    return routed_rows


def _evaluate_rows(
    *,
    eval_mod: Any,
    benchmark: str,
    rows: list[dict[str, Any]],
    out_dir: Path,
    strategy_name: str,
    judge_runtime: Any,
    judge_sampling: Any,
    judge_batch_size: int,
    reuse: bool,
    debug: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored_path = out_dir / "results_scored.jsonl"
    skipped_path = out_dir / "results_scored_skipped.json"
    summary_path = out_dir / "summary_scored.json"
    if reuse and scored_path.exists() and summary_path.exists():
        scored_rows = read_jsonl(scored_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[skip] {benchmark}:{strategy_name} scored rows already complete: {rel(scored_path)}", flush=True)
        return scored_rows, summary

    scored_rows, skipped_rows = eval_mod.evaluate_saved_rows_batched(
        benchmark=benchmark,
        rows=rows,
        judge_runtime=judge_runtime,
        judge_sampling=judge_sampling,
        judge_batch_size=int(judge_batch_size),
        debug=bool(debug),
        progress_desc=f"{benchmark}:{strategy_name}:eval",
    )
    write_jsonl(scored_path, scored_rows)
    write_json(skipped_path, skipped_rows)
    summary = eval_mod.safe_summarize_scored_rows(benchmark, strategy_name, scored_rows, skipped_rows)
    summary.update(eval_mod.summarize_strategy_cost_and_routing(benchmark, scored_rows))
    summary["primary_metric_name"] = eval_mod.benchmark_primary_metric_name(benchmark)
    summary["primary_metric_value"] = eval_mod.benchmark_primary_metric_value(benchmark, summary)
    write_json(summary_path, summary)
    return scored_rows, summary


def _paid_cost(summary: dict[str, Any], input_price: float, output_price: float) -> float:
    usage = summary.get("usage_totals_by_model", {}).get("model2", {})
    prompt_tokens = float(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = float(usage.get("completion_tokens", 0) or 0)
    return prompt_tokens * float(input_price) / 1_000_000.0 + completion_tokens * float(output_price) / 1_000_000.0


def _summary_row(
    method: str,
    benchmark: str,
    summary: dict[str, Any],
    input_price: float,
    output_price: float,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    usage = summary.get("usage_totals_by_model", {}).get("model2", {})
    row = {
        "method": method,
        "benchmark": benchmark,
        "score": float(summary.get("primary_metric_value", summary.get("accuracy", 0.0)) or 0.0),
        "paid_cost_usd": _paid_cost(summary, input_price, output_price),
        "num_rows": int(summary.get("num_rows", summary.get("num_scored", 0)) or 0),
        "model2_calls": int(usage.get("generation_calls", 0) or 0),
        "model2_prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "model2_completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }
    if threshold is not None:
        row["threshold"] = float(threshold)
    return row


def _overall_rows(rows: list[dict[str, Any]], benchmarks: Sequence[str]) -> list[dict[str, Any]]:
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    out = list(rows)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method and row["benchmark"] in benchmarks]
        if not method_rows:
            continue
        overall = {
            "method": method,
            "benchmark": "overall",
            "score": sum(float(row["score"]) for row in method_rows) / len(method_rows),
            "paid_cost_usd": sum(float(row["paid_cost_usd"]) for row in method_rows),
            "num_rows": sum(int(row["num_rows"]) for row in method_rows),
            "model2_calls": sum(int(row["model2_calls"]) for row in method_rows),
            "model2_prompt_tokens": sum(int(row["model2_prompt_tokens"]) for row in method_rows),
            "model2_completion_tokens": sum(int(row["model2_completion_tokens"]) for row in method_rows),
        }
        if "threshold" in method_rows[0]:
            overall["threshold"] = float(method_rows[0]["threshold"])
        out.append(overall)
    return out


def _print_table(rows: list[dict[str, Any]], benchmarks: Sequence[str], threshold: float | None = None) -> None:
    order = [*benchmarks, "overall"]
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["benchmark"]] = row
    suffix = "" if threshold is None else f" (threshold={float(threshold):.2f})"
    print(f"\nTable 2 style text benchmark score / paid cost{suffix}", flush=True)
    print("| Method | " + " | ".join(name if name != "overall" else "Overall" for name in order) + " |", flush=True)
    print("|---|" + "|".join("---:" for _ in order) + "|", flush=True)
    for method, payload in by_method.items():
        cells = []
        for benchmark in order:
            row = payload.get(benchmark)
            if row is None:
                cells.append("--")
            else:
                cells.append(f"{format_float(row['score'], 3)} / ${float(row['paid_cost_usd']):.2f}")
        print("| " + method + " | " + " | ".join(cells) + " |", flush=True)


def _plot_table(rows: list[dict[str, Any]], output_path: Path, benchmarks: Sequence[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skip plot: {exc}", flush=True)
        return
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    bench_order = [*benchmarks, "overall"]
    x = list(range(len(bench_order)))
    width = 0.8 / max(len(methods), 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for m_idx, method in enumerate(methods):
        offsets = [pos - 0.4 + width * (m_idx + 0.5) for pos in x]
        method_rows = {row["benchmark"]: row for row in rows if row["method"] == method}
        ax1.bar(offsets, [float(method_rows[b]["score"]) if b in method_rows else 0.0 for b in bench_order], width=width, label=method)
        ax2.bar(offsets, [float(method_rows[b]["paid_cost_usd"]) if b in method_rows else 0.0 for b in bench_order], width=width, label=method)
    ax1.set_ylabel("Score")
    ax2.set_ylabel("Paid cost (USD)")
    ax2.set_xticks(x, [b if b != "overall" else "Overall" for b in bench_order], rotation=20, ha="right")
    ax1.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds, args.threshold)
    args.threshold = thresholds[0]
    multi_threshold = len(thresholds) > 1
    benchmarks = _parse_benchmarks(args.benchmarks)
    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    _validate_benchmark_inputs(args, benchmarks)

    tag = model_tag(args.model1_path)
    dirs = output_dirs(data_root, tag, "capability")
    ours_head_path = dirs["trained"] / "neuron_head_final.pt" if args.ours_head_checkpoint_path is None else resolve_from_code_root(args.ours_head_checkpoint_path)
    ours_neuron_path = dirs["neurons"] / "selected_neurons.jsonl" if args.ours_neuron_path is None else resolve_from_code_root(args.ours_neuron_path)
    baseline_head_path = _resolve_baseline_head_path(args, data_root)
    output_dir = (
        data_root / "eval_outputs" / "neuron_heads" / tag / "capability_table2_textbench"
        if args.output_dir is None else resolve_from_code_root(args.output_dir)
    )

    if args.clean:
        clean_path(output_dir, [data_root], "capability table2 textbench eval output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ours_head_path.exists():
        raise FileNotFoundError(f"Missing neuron capability head checkpoint: {rel(ours_head_path)}")
    if not ours_neuron_path.exists():
        raise FileNotFoundError(f"Missing selected capability neurons: {rel(ours_neuron_path)}")

    shared = load_upstream_module(
        "multi_agenT_bench/compact_multi_agent_shared_optimized_v4_textbench.py",
        "mhlc_upstream_textbench_shared_for_table2",
    )
    generate_mod = load_upstream_module(
        "multi_agenT_bench/compact_multi_agent_generate.py",
        "mhlc_upstream_textbench_generate_for_table2",
    )
    eval_mod = load_upstream_module(
        "multi_agenT_bench/compact_multi_agent_evaluate.py",
        "mhlc_upstream_textbench_eval_for_table2",
    )
    if hasattr(shared, "set_seed"):
        shared.set_seed(int(args.seed))

    write_json(output_dir / "run_config.json", {
        "model1_path": args.model1_path,
        "model2_path": args.model2_path,
        "benchmarks": benchmarks,
        "max_samples": int(args.max_samples),
        "threshold": float(args.threshold),
        "thresholds": thresholds,
        "score_pred_threshold": float(args.threshold),
        "baseline_head_path": rel(baseline_head_path),
        "ours_head_checkpoint_path": rel(ours_head_path),
        "ours_neuron_path": rel(ours_neuron_path),
        "judge_model_path": args.judge_model_path,
        "judge_model_family": args.judge_model_family,
        "judge_thinking_mode": args.judge_thinking_mode,
        "judge_batch_size": int(args.judge_batch_size),
        "m2_input_cost_per_1m_usd": float(args.m2_input_cost_per_1m_usd),
        "m2_output_cost_per_1m_usd": float(args.m2_output_cost_per_1m_usd),
        "cost_note": "Paid cost counts only model2 generation tokens. Model1 and head scoring are treated as local/free, matching Table 2 convention.",
    })

    judge_runtime = None
    judge_sampling = None
    if str(args.judge_model_path).strip() and any(shared.benchmark_needs_judge(benchmark) for benchmark in benchmarks):
        judge_profile = {
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.40,
            "max_num_seqs": 32,
            "enforce_eager": False,
            "trust_remote_code": True,
        }
        judge_runtime, judge_sampling = shared.build_judge_runtime_and_sampling(
            judge_model_name_or_path=str(args.judge_model_path),
            judge_runtime_profile=judge_profile,
            judge_sampling_profiles={
                "default": eval_mod.JUDGE_DEFAULT_SAMPLING_PROFILE,
                "thinking": eval_mod.JUDGE_THINKING_SAMPLING_PROFILE,
                "instruct": eval_mod.JUDGE_INSTRUCT_SAMPLING_PROFILE,
            },
            judge_model_family=str(args.judge_model_family),
            judge_thinking_mode=str(args.judge_thinking_mode),
        )

    dataset_cfgs = _dataset_cfgs(args)
    all_rows_by_threshold: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds}
    try:
        for benchmark in benchmarks:
            bench_dir = output_dir / benchmark
            bench_dir.mkdir(parents=True, exist_ok=True)
            examples = shared.load_examples_for_benchmark(benchmark, dataset_cfgs[benchmark])
            write_json(bench_dir / "dataset_summary.json", {"benchmark": benchmark, "num_examples": len(examples), "dataset_cfg": dataset_cfgs[benchmark]})
            print(f"[data] {benchmark} examples={len(examples)}", flush=True)

            model1_bundle = _build_generation_bundle(shared, generate_mod, args, "model1", benchmark)
            model2_bundle = _build_generation_bundle(shared, generate_mod, args, "model2", benchmark)
            m1_rows = _generate_single_agent_rows(
                shared=shared,
                bundle=model1_bundle,
                benchmark=benchmark,
                examples=examples,
                slot="model1",
                strategy_name="single_agent_model1",
                path=bench_dir / "single_agent_model1" / "results.jsonl",
                batch_size=int(args.model1_generation_batch_size),
                reuse=bool(args.reuse_generations),
            )
            m2_rows = _generate_single_agent_rows(
                shared=shared,
                bundle=model2_bundle,
                benchmark=benchmark,
                examples=examples,
                slot="model2",
                strategy_name="single_agent_model2",
                path=bench_dir / "single_agent_model2" / "results.jsonl",
                batch_size=int(args.model2_generation_batch_size),
                reuse=bool(args.reuse_generations),
            )

            mhlc_scores = _score_mhlc_head(
                shared=shared,
                args=args,
                benchmark=benchmark,
                examples=examples,
                m1_rows=m1_rows,
                baseline_head_path=baseline_head_path,
                out_path=bench_dir / "head_scores_mhlc.jsonl",
                reuse=bool(args.reuse_scores),
            )
            ours_scores = _score_ours_head(
                shared=shared,
                args=args,
                benchmark=benchmark,
                examples=examples,
                m1_rows=m1_rows,
                head_path=ours_head_path,
                neuron_path=ours_neuron_path,
                out_path=bench_dir / "head_scores_ours.jsonl",
                reuse=bool(args.reuse_scores),
            )

            route_specs = [
                ("Backbone + Capability Head（MHLC）", "routed_mhlc", mhlc_scores),
                ("Backbone + Capability Head（Ours）", "routed_ours", ours_scores),
            ]
            generation_sets = [
                ("Backbone Choice", "single_agent_model1", m1_rows),
                ("Always Call Strong Model", "single_agent_model2", m2_rows),
            ]
            for display_name, strategy_name, rows in generation_sets:
                strategy_dir = bench_dir / strategy_name
                write_jsonl(strategy_dir / "results.jsonl", rows)
                _scored, summary = _evaluate_rows(
                    eval_mod=eval_mod,
                    benchmark=benchmark,
                    rows=rows,
                    out_dir=strategy_dir,
                    strategy_name=strategy_name,
                    judge_runtime=judge_runtime,
                    judge_sampling=judge_sampling,
                    judge_batch_size=int(args.judge_batch_size),
                    reuse=bool(args.reuse_evaluations),
                    debug=bool(args.debug),
                )
                for threshold in thresholds:
                    all_rows_by_threshold[threshold].append(_summary_row(
                        display_name,
                        benchmark,
                        summary,
                        args.m2_input_cost_per_1m_usd,
                        args.m2_output_cost_per_1m_usd,
                        threshold=threshold,
                    ))

            for display_name, strategy_name, score_rows in route_specs:
                for threshold in thresholds:
                    routed_rows = _build_routed_rows(
                        generate_mod=generate_mod,
                        shared=shared,
                        benchmark=benchmark,
                        examples=examples,
                        m1_rows=m1_rows,
                        m2_rows=m2_rows,
                        score_rows=score_rows,
                        strategy_name=strategy_name,
                        threshold=float(threshold),
                        call_model2_if_missing_final_answer=bool(args.call_model2_if_missing_final_answer),
                    )
                    strategy_dir = _strategy_dir(
                        bench_dir,
                        strategy_name,
                        threshold,
                        multi_threshold=multi_threshold,
                    )
                    strategy_dir.mkdir(parents=True, exist_ok=True)
                    write_jsonl(strategy_dir / "results.jsonl", routed_rows)
                    _scored, summary = _evaluate_rows(
                        eval_mod=eval_mod,
                        benchmark=benchmark,
                        rows=routed_rows,
                        out_dir=strategy_dir,
                        strategy_name=f"{strategy_name}_threshold_{_threshold_tag(threshold)}" if multi_threshold else strategy_name,
                        judge_runtime=judge_runtime,
                        judge_sampling=judge_sampling,
                        judge_batch_size=int(args.judge_batch_size),
                        reuse=bool(args.reuse_evaluations),
                        debug=bool(args.debug),
                    )
                    all_rows_by_threshold[threshold].append(_summary_row(
                        display_name,
                        benchmark,
                        summary,
                        args.m2_input_cost_per_1m_usd,
                        args.m2_output_cost_per_1m_usd,
                        threshold=threshold,
                    ))
    finally:
        if judge_runtime is not None:
            judge_runtime.unload(drop_processor=False)

    tables: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for idx, threshold in enumerate(thresholds):
        table_rows = _overall_rows(all_rows_by_threshold[threshold], benchmarks)
        tables.append({"threshold": float(threshold), "rows": table_rows})
        combined_rows.extend(table_rows)
        suffix = _threshold_tag(threshold)
        write_json(
            output_dir / f"table2_textbench_comparison_threshold_{suffix}.json",
            {"threshold": float(threshold), "rows": table_rows},
        )
        write_csv(output_dir / f"table2_textbench_comparison_threshold_{suffix}.csv", table_rows)
        threshold_plot = output_dir / "plots" / f"table2_score_cost_threshold_{suffix}.png"
        _plot_table(table_rows, threshold_plot, benchmarks)
        if idx == 0:
            _plot_table(table_rows, output_dir / "plots" / "table2_score_cost.png", benchmarks)
        _print_table(table_rows, benchmarks, threshold)

    write_json(output_dir / "table2_textbench_comparison.json", {"thresholds": thresholds, "tables": tables, "rows": combined_rows})
    write_csv(output_dir / "table2_textbench_comparison.csv", combined_rows)
    print(f"[done] saved Table 2 style textbench eval to {rel(output_dir)}", flush=True)


if __name__ == "__main__":
    main()
