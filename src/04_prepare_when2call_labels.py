#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

from mhlc_data_prep.parallel import (
    configure_parallel_context,
    contiguous_range,
    require_single_gpu_vllm,
    worker_dir,
)

PARALLEL = configure_parallel_context()

from datasets import DatasetDict, load_from_disk
from tqdm.auto import tqdm

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel, temporary_argv
from mhlc_data_prep.specs import WHEN2CALL_REPO


def _select_split(ds: Any, split: str):
    if isinstance(ds, DatasetDict):
        if split in ds:
            return ds[split]
        if "train" in ds:
            return ds["train"]
        return ds[next(iter(ds.keys()))]
    return ds


def _stable_seed(base_seed: int, name: str) -> int:
    return int(base_seed) + sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def patch_when2call_loader(
    module: Any,
    data_root,
    allow_hf_fallback: bool,
    *,
    random_sample: bool,
    seed: int,
) -> None:
    original_load_dataset = module.load_dataset
    local_root = data_root / "data" / "sources" / "when2call"

    def local_load_dataset(path: str, name: str | None = None, *args, **kwargs):
        if path == WHEN2CALL_REPO and name:
            local_dir = local_root / str(name)
            if local_dir.exists():
                split = kwargs.get("split", "train")
                ds = _select_split(load_from_disk(str(local_dir)), split)
                if random_sample and len(ds) > 0:
                    ds = ds.shuffle(seed=_stable_seed(seed, str(name)))
                return ds
            if not allow_hf_fallback:
                raise FileNotFoundError(
                    f"Missing materialized When2Call split {name}: {rel(local_dir)}\n"
                    "Run: python src/01_download_data.py --group when2call"
                )
        return original_load_dataset(path, name, *args, **kwargs)

    module.load_dataset = local_load_dataset


def patch_parallel_sample_partition(module: Any) -> None:
    """Partition the canonical upstream sample sequence without changing it."""
    original_iter = module.iter_canonical_samples

    def partitioned_iter(*iter_args, **iter_kwargs):
        samples = list(original_iter(*iter_args, **iter_kwargs))
        start, end = contiguous_range(len(samples), PARALLEL)
        selected = samples[start:end]
        if PARALLEL.enabled:
            return iter(
                tqdm(
                    selected,
                    desc=f"rank {PARALLEL.rank + 1}/{PARALLEL.world_size} label when2call",
                    unit="sample",
                    dynamic_ncols=True,
                    position=PARALLEL.rank,
                )
            )
        return iter(selected)

    module.iter_canonical_samples = partitioned_iter


def _completion_marker(output_dir) -> Any:
    return output_dir / ".when2call_labels_eight_gpu_complete.json"


def _merge_parallel_outputs(module: Any, output_dir, work_root) -> None:
    jsonl_path = output_dir / "when2call_aux_labels.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as merged_file:
        for rank in range(PARALLEL.world_size):
            rank_dir = work_root / f"rank_{rank:02d}"
            rank_jsonl = rank_dir / "when2call_aux_labels.jsonl"
            if not rank_jsonl.exists():
                raise FileNotFoundError(f"Missing eight-GPU label partition: {rel(rank_jsonl)}")
            with open(rank_jsonl, "r", encoding="utf-8") as part_file:
                for line in part_file:
                    merged_file.write(line)
    module.export_jsonl_to_parquet(jsonl_path, output_dir / "when2call_aux_labels.parquet")

    summary = {"parallel_workers": PARALLEL.world_size, "total_seen": 0, "total_written": 0, "per_split": {}, "latent_category_counts": {}, "behavior_counts": {}, "usable_counts": {}}
    for rank in range(PARALLEL.world_size):
        rank_stats = work_root / f"rank_{rank:02d}" / "when2call_aux_labels_stats.json"
        payload = json.loads(rank_stats.read_text(encoding="utf-8"))
        for key in ("total_seen", "total_written"):
            summary[key] += int(payload.get(key, 0))
        for key in ("latent_category_counts", "behavior_counts", "usable_counts"):
            for name, count in payload.get(key, {}).items():
                summary[key][name] = int(summary[key].get(name, 0)) + int(count)
        for split, values in payload.get("per_split", {}).items():
            dest = summary["per_split"].setdefault(split, {"rows": 0})
            dest["rows"] += int(values.get("rows", 0))
    (output_dir / "when2call_aux_labels_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _completion_marker(output_dir).write_text(
        json.dumps({"world_size": PARALLEL.world_size, "rows": summary["total_written"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[merge] when2call label workers -> {rel(output_dir)} rows={summary['total_written']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build MHLC When2Call 4-class labels with upstream prompts and annotator settings."
    )
    ap.add_argument("--data-root", default="../mhlc_data")
    ap.add_argument("--output-dir", default="../mhlc_data/data/train/when2call/when2call_processed_4class")
    ap.add_argument("--annotator-model-id", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    ap.add_argument("--tokenizer-id", default=None)
    ap.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Smoke-run row cap for each upstream When2Call split. Omit for the formal full run.",
    )
    ap.add_argument(
        "--random-sample",
        action="store_true",
        help="Shuffle each local When2Call split before applying --max-rows-per-split.",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-hf-fallback", action="store_true")
    ap.add_argument("--clean", action="store_true", help="Remove existing When2Call labels before labeling.")
    args = ap.parse_args()

    require_single_gpu_vllm(args.tensor_parallel_size, PARALLEL)

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    output_dir = resolve_from_code_root(args.output_dir)
    if PARALLEL.enabled and _completion_marker(output_dir).exists() and not args.clean:
        if PARALLEL.is_main:
            print(f"[skip] completed eight-GPU When2Call labels: {rel(output_dir)}", flush=True)
        return
    if PARALLEL.enabled and output_dir.exists() and not args.clean:
        raise FileExistsError(
            f"Existing output is not an eight-GPU completed run: {rel(output_dir)}. "
            "Use --clean to start a new eight-GPU run without mixing artifacts."
        )
    work_root = worker_dir(output_dir, "when2call_labels", PARALLEL).parent
    if args.clean and (not PARALLEL.enabled or PARALLEL.is_main):
        clean_path(output_dir, [data_root], "when2call label dir")
        if PARALLEL.enabled:
            clean_path(work_root, [data_root], "when2call label worker cache")
    PARALLEL.barrier()
    tokenizer_id = args.tokenizer_id or args.annotator_model_id

    module = load_upstream_module(
        "when2call/when2call_build_head_labels_4class.py",
        "mhlc_upstream_when2call_build_head_labels_4class",
    )
    patch_when2call_loader(
        module,
        data_root,
        allow_hf_fallback=bool(args.allow_hf_fallback),
        random_sample=bool(args.random_sample and args.max_rows_per_split is not None),
        seed=int(args.seed),
    )
    if PARALLEL.enabled:
        patch_parallel_sample_partition(module)
    worker_output_dir = output_dir if not PARALLEL.enabled else worker_dir(output_dir, "when2call_labels", PARALLEL)

    argv = [
        "when2call_build_head_labels_4class.py",
        "--dataset_id",
        WHEN2CALL_REPO,
        "--splits",
        "train_sft",
        "train_pref",
        "--output_dir",
        rel(worker_output_dir),
        "--model_id",
        args.annotator_model_id,
        "--tokenizer_id",
        tokenizer_id,
        "--dtype",
        "auto",
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
        "--export_parquet",
    ]
    if args.max_rows_per_split is not None:
        argv.extend(["--max_rows_per_split", str(args.max_rows_per_split)])

    print("[stage] when2call label construction")
    print(f"[output] {rel(output_dir)}")
    print(f"[annotator] {args.annotator_model_id}")
    if PARALLEL.enabled:
        print(f"[parallel] rank={PARALLEL.rank}/{PARALLEL.world_size} worker_output={rel(worker_output_dir)}")
    if args.max_rows_per_split is not None:
        print(f"[mode] smoke max_rows_per_split={args.max_rows_per_split}; omit it for formal full run")
        if args.random_sample:
            print(f"[mode] random sampling enabled with seed={args.seed}")
    with temporary_argv(argv):
        module.main()
    PARALLEL.barrier()
    if PARALLEL.enabled and PARALLEL.is_main:
        _merge_parallel_outputs(module, output_dir, work_root)
    PARALLEL.barrier()


if __name__ == "__main__":
    main()
