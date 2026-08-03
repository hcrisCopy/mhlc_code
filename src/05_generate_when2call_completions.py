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


def patch_parallel_row_partition(module) -> None:
    """Partition immediately after upstream applies its seeded global shuffle."""
    original_random = module.random.Random

    class PartitioningRandom(original_random):
        def shuffle(self, values, *shuffle_args, **shuffle_kwargs):
            result = super().shuffle(values, *shuffle_args, **shuffle_kwargs)
            start, end = contiguous_range(len(values), PARALLEL)
            values[:] = values[start:end]
            return result

    module.random.Random = PartitioningRandom


def _completion_marker(output_dir):
    return output_dir / ".when2call_completions_eight_gpu_complete.json"


def _merge_parallel_outputs(module, output_dir, work_root, args) -> None:
    all_rows = []
    stats_parts = []
    for rank in range(PARALLEL.world_size):
        rank_dir = work_root / f"rank_{rank:02d}"
        shard_dir = rank_dir / "parquet_shards"
        for shard_path in sorted(shard_dir.glob("shard-*.parquet")):
            all_rows.extend(module.load_dataset("parquet", data_files=str(shard_path), split="train").to_list())
        stats_parts.append(json.loads((rank_dir / "generation_stats.json").read_text(encoding="utf-8")))

    shard_dir = output_dir / "parquet_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = int(stats_parts[0].get("parquet_chunk_size", 2000))
    for shard_index, start in enumerate(range(0, len(all_rows), chunk_size)):
        module.write_parquet_shard(all_rows[start:start + chunk_size], shard_dir / f"shard-{shard_index:06d}.parquet")
    base = dict(stats_parts[0])
    base.update(
        {
            "output_dir": str(output_dir),
            "parquet_shard_dir": str(shard_dir),
            "num_rows_after_filtering": sum(int(item.get("num_rows_after_filtering", 0)) for item in stats_parts),
            "num_total_candidates": sum(int(item.get("num_total_candidates", 0)) for item in stats_parts),
            "num_skipped_resume": sum(int(item.get("num_skipped_resume", 0)) for item in stats_parts),
            "num_generated_new": sum(int(item.get("num_generated_new", 0)) for item in stats_parts),
            "num_rows_written_new": len(all_rows),
            "next_shard_idx": len(list(shard_dir.glob("shard-*.parquet"))),
            "parallel_workers": PARALLEL.world_size,
        }
    )
    (output_dir / "generation_stats.json").write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    _completion_marker(output_dir).write_text(
        json.dumps({"world_size": PARALLEL.world_size, "rows": len(all_rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[merge] when2call completion workers -> {rel(output_dir)} rows={len(all_rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate target-model completions for the MHLC When2Call 4-class head dataset."
    )
    ap.add_argument("--data-root", default="../mhlc_data")
    ap.add_argument(
        "--input-path",
        default="../mhlc_data/data/train/when2call/when2call_processed_4class/when2call_aux_labels.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class",
    )
    ap.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--model-family", default="qwen3_vl", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4"])
    ap.add_argument("--thinking-mode", default="off")
    ap.add_argument("--limit", type=int, default=None, help="Smoke-run row cap. Omit for the formal full run.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="Remove existing When2Call completions before generating.")
    args = ap.parse_args()

    require_single_gpu_vllm(args.tensor_parallel_size, PARALLEL)

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    input_path = resolve_from_code_root(args.input_path)
    output_dir = resolve_from_code_root(args.output_dir)
    model_path = resolve_from_code_root(args.model_path)
    tokenizer_path = resolve_from_code_root(args.tokenizer_path) if args.tokenizer_path else model_path
    if PARALLEL.enabled and _completion_marker(output_dir).exists() and not args.clean:
        if PARALLEL.is_main:
            print(f"[skip] completed eight-GPU When2Call completions: {rel(output_dir)}", flush=True)
        return
    if PARALLEL.enabled and output_dir.exists() and not args.clean:
        raise FileExistsError(
            f"Existing output is not an eight-GPU completed run: {rel(output_dir)}. "
            "Use --clean to start a new eight-GPU run without mixing artifacts."
        )
    work_root = worker_dir(output_dir, "when2call_completions", PARALLEL).parent
    if args.clean and (not PARALLEL.enabled or PARALLEL.is_main):
        clean_path(output_dir, [data_root], "when2call completion dir")
        if PARALLEL.enabled:
            clean_path(work_root, [data_root], "when2call completion worker cache")
    PARALLEL.barrier()

    module = load_upstream_module(
        "when2call/when2call_generate_completions_4class.py",
        "mhlc_upstream_when2call_generate_completions_4class",
    )
    if PARALLEL.enabled:
        patch_parallel_row_partition(module)
    worker_output_dir = output_dir if not PARALLEL.enabled else worker_dir(output_dir, "when2call_completions", PARALLEL)
    argv = [
        "when2call_generate_completions_4class.py",
        "--input_path",
        rel(input_path),
        "--output_dir",
        rel(worker_output_dir),
        "--model_id",
        rel(model_path),
        "--tokenizer_id",
        rel(tokenizer_path),
        "--model_family",
        args.model_family,
        "--thinking_mode",
        args.thinking_mode,
    ]
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
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
        "--resume",
    ])

    print("[stage] when2call target-model completion generation")
    print(f"[input] {rel(input_path)}")
    print(f"[model] {rel(model_path)}")
    print(f"[output] {rel(output_dir)}")
    if PARALLEL.enabled:
        print(f"[parallel] rank={PARALLEL.rank}/{PARALLEL.world_size} worker_output={rel(worker_output_dir)}")
    if args.limit is not None:
        print(f"[mode] smoke limit={args.limit}; omit it for formal full run")
    with temporary_argv(argv):
        module.main()
    PARALLEL.barrier()
    if PARALLEL.enabled and PARALLEL.is_main:
        _merge_parallel_outputs(module, output_dir, work_root, args)
    PARALLEL.barrier()


if __name__ == "__main__":
    main()
