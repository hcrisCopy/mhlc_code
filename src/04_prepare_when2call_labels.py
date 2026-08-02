#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from datasets import DatasetDict, load_from_disk

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


def patch_when2call_loader(module: Any, data_root, allow_hf_fallback: bool) -> None:
    original_load_dataset = module.load_dataset
    local_root = data_root / "data" / "sources" / "when2call"

    def local_load_dataset(path: str, name: str | None = None, *args, **kwargs):
        if path == WHEN2CALL_REPO and name:
            local_dir = local_root / str(name)
            if local_dir.exists():
                split = kwargs.get("split", "train")
                return _select_split(load_from_disk(str(local_dir)), split)
            if not allow_hf_fallback:
                raise FileNotFoundError(
                    f"Missing materialized When2Call split {name}: {rel(local_dir)}\n"
                    "Run: python src/01_download_data.py --group when2call"
                )
        return original_load_dataset(path, name, *args, **kwargs)

    module.load_dataset = local_load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build MHLC When2Call 4-class labels with upstream prompts and annotator settings."
    )
    ap.add_argument("--data-root", default="../mhlc_data")
    ap.add_argument("--output-dir", default="../mhlc_data/data/train/when2call/when2call_processed_4class")
    ap.add_argument("--annotator-model-id", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    ap.add_argument("--tokenizer-id", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-hf-fallback", action="store_true")
    ap.add_argument("--clean", action="store_true", help="Remove existing When2Call labels before labeling.")
    args = ap.parse_args()

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    output_dir = resolve_from_code_root(args.output_dir)
    if args.clean:
        clean_path(output_dir, [data_root], "when2call label dir")
    tokenizer_id = args.tokenizer_id or args.annotator_model_id

    module = load_upstream_module(
        "when2call/when2call_build_head_labels_4class.py",
        "mhlc_upstream_when2call_build_head_labels_4class",
    )
    patch_when2call_loader(module, data_root, allow_hf_fallback=bool(args.allow_hf_fallback))

    argv = [
        "when2call_build_head_labels_4class.py",
        "--dataset_id",
        WHEN2CALL_REPO,
        "--splits",
        "train_sft",
        "train_pref",
        "--output_dir",
        rel(output_dir),
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

    print("[stage] when2call label construction")
    print(f"[output] {rel(output_dir)}")
    print(f"[annotator] {args.annotator_model_id}")
    with temporary_argv(argv):
        module.main()


if __name__ == "__main__":
    main()
