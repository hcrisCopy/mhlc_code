#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mhlc_data_prep.parallel import (
    configure_parallel_context,
    contiguous_range,
    require_single_gpu_vllm,
    worker_dir,
)

PARALLEL = configure_parallel_context()

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel, temporary_argv
from mhlc_data_prep.specs import WHEN2CALL_REPO


def patch_parallel_eval_partition(module) -> None:
    original_load_eval_rows = module.load_eval_rows

    def partitioned_load_eval_rows(cfg):
        rows, stats = original_load_eval_rows(cfg)
        start, end = contiguous_range(len(rows), PARALLEL)
        stats = dict(stats)
        stats.update({"parallel_rank": PARALLEL.rank, "parallel_world_size": PARALLEL.world_size, "partition_rows": end - start})
        return rows[start:end], stats

    module.load_eval_rows = partitioned_load_eval_rows


def _completion_marker(output_path):
    return output_path.with_suffix(".eight_gpu_complete.json")


def _merge_parallel_outputs(module, output_path, work_root) -> None:
    rows = []
    stats_parts = []
    for rank in range(PARALLEL.world_size):
        rank_output = work_root / f"rank_{rank:02d}" / output_path.name
        if not rank_output.exists():
            raise FileNotFoundError(f"Missing eight-GPU eval completion partition: {rel(rank_output)}")
        rows.extend(module.load_dataset("parquet", data_files=str(rank_output), split="train").to_list())
        stats_parts.append(json.loads(rank_output.with_suffix(".stats.json").read_text(encoding="utf-8")))
    module.Dataset.from_list(rows).to_parquet(str(output_path))
    base = dict(stats_parts[0])
    base["num_generated_rows"] = len(rows)
    base["parallel_workers"] = PARALLEL.world_size
    output_path.with_suffix(".stats.json").write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl_path = output_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _completion_marker(output_path).write_text(
        json.dumps({"world_size": PARALLEL.world_size, "rows": len(rows)}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[merge] when2call eval completion workers -> {rel(output_path)} rows={len(rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optional: generate official When2Call test completions before head evaluation."
    )
    ap.add_argument("--data-root", default="../mhlc_data")
    ap.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--output-path", default="../mhlc_data/eval_outputs/when2call/Qwen3-VL-4B-Instruct/when2call_test_generated_4class.parquet")
    ap.add_argument("--model-family", default="qwen3_vl", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4"])
    ap.add_argument("--thinking-mode", default="off")
    ap.add_argument("--max-eval-rows", type=int, default=None, help="Smoke-run eval row cap. Omit for the formal full run.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32000)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-hf-fallback", action="store_true")
    ap.add_argument("--clean", action="store_true", help="Remove existing eval completion outputs before generating.")
    args = ap.parse_args()
    require_single_gpu_vllm(args.tensor_parallel_size, PARALLEL)
    from datasets import DatasetDict, load_from_disk

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    model_path = resolve_from_code_root(args.model_path)
    tokenizer_path = resolve_from_code_root(args.tokenizer_path) if args.tokenizer_path else model_path
    output_path = resolve_from_code_root(args.output_path)
    if PARALLEL.enabled and _completion_marker(output_path).exists() and not args.clean:
        if PARALLEL.is_main:
            print(f"[skip] completed eight-GPU eval completions: {rel(output_path)}", flush=True)
        return
    if PARALLEL.enabled and output_path.exists() and not args.clean:
        raise FileExistsError(
            f"Existing output is not an eight-GPU completed run: {rel(output_path)}. "
            "Use --clean to start a new eight-GPU run without mixing artifacts."
        )
    work_root = worker_dir(output_path, "when2call_eval_completions", PARALLEL).parent
    if args.clean and (not PARALLEL.enabled or PARALLEL.is_main):
        clean_path(output_path, [data_root], "when2call eval parquet")
        clean_path(output_path.with_suffix(".stats.json"), [data_root], "when2call eval stats")
        clean_path(output_path.with_suffix(".jsonl"), [data_root], "when2call eval jsonl")
        clean_path(_completion_marker(output_path), [data_root], "when2call eval parallel completion marker")
        if PARALLEL.enabled:
            clean_path(work_root, [data_root], "when2call eval worker cache")
    elif not PARALLEL.enabled and output_path.exists():
        print(f"[skip] eval completion already exists: {rel(output_path)}")
        print("[hint] pass --clean to regenerate it")
        return
    PARALLEL.barrier()

    module = load_upstream_module(
        "when2call/eval/generate_when2call_eval_completions_4class.py",
        "mhlc_upstream_generate_when2call_eval_completions_4class",
    )

    original_load_dataset = module.load_dataset
    local_test_dir = data_root / "data" / "sources" / "when2call" / "test"

    def local_load_dataset(path, name=None, *load_args, **kwargs):
        if path == WHEN2CALL_REPO and name == "test" and local_test_dir.exists():
            ds = load_from_disk(str(local_test_dir))
            if isinstance(ds, DatasetDict):
                return ds
            return DatasetDict({"mcq": ds})
        if path == WHEN2CALL_REPO and name == "test" and not args.allow_hf_fallback:
            raise FileNotFoundError(
                f"Missing materialized When2Call test split: {rel(local_test_dir)}\n"
                "Run: python src/01_download_data.py --group when2call"
            )
        return original_load_dataset(path, name, *load_args, **kwargs)

    module.load_dataset = local_load_dataset
    if PARALLEL.enabled:
        patch_parallel_eval_partition(module)
    original_llm = module.LLM

    def patched_llm(*llm_args, **llm_kwargs):
        if args.max_num_seqs is not None:
            llm_kwargs.setdefault("max_num_seqs", int(args.max_num_seqs))
        if args.enforce_eager:
            llm_kwargs.setdefault("enforce_eager", True)
        return original_llm(*llm_args, **llm_kwargs)

    module.LLM = patched_llm

    worker_output_path = output_path if not PARALLEL.enabled else worker_dir(output_path, "when2call_eval_completions", PARALLEL) / output_path.name
    argv = [
        "generate_when2call_eval_completions_4class.py",
        "--model_id",
        rel(model_path),
        "--tokenizer_id",
        rel(tokenizer_path),
        "--model_family",
        args.model_family,
        "--thinking_mode",
        args.thinking_mode,
        "--dataset_name",
        WHEN2CALL_REPO,
        "--dataset_config",
        "test",
        "--eval_split",
        "mcq",
    ]
    if args.max_eval_rows is not None:
        argv.extend(["--max_eval_rows", str(args.max_eval_rows)])
    argv.extend([
        "--batch_size",
        str(args.batch_size),
        "--max_tokens",
        str(args.max_tokens),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--max_model_len",
        str(args.max_model_len),
        "--seed",
        str(args.seed),
        "--output_path",
        rel(worker_output_path),
        "--also_write_jsonl",
        "true",
    ])

    print("[stage] optional when2call eval completion generation")
    print(f"[model] {rel(model_path)}")
    print(f"[output] {rel(output_path)}")
    if PARALLEL.enabled:
        print(f"[parallel] rank={PARALLEL.rank}/{PARALLEL.world_size} worker_output={rel(worker_output_path)}")
    if args.max_eval_rows is not None:
        print(f"[mode] smoke max_eval_rows={args.max_eval_rows}; omit it for formal full run")
    with temporary_argv(argv):
        module.main()
    PARALLEL.barrier()
    if PARALLEL.enabled and PARALLEL.is_main:
        _merge_parallel_outputs(module, output_path, work_root)
    PARALLEL.barrier()


if __name__ == "__main__":
    main()
