#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import (
    ensure_mhlc_data_layout,
    resolve_from_code_root,
    set_hf_dirs_inside_data_root,
)
from mhlc_data_prep.run_utils import clean_path, rel, temporary_argv
from mhlc_data_prep.specs import TEXT_SOURCE_COUNTS, TEXT_TOTAL_QA_PAIRS


DEFAULT_RUN_NAME = f"Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_{TEXT_TOTAL_QA_PAIRS}"
DEFAULT_SAVE_REL = f"../mhlc_data/data/train/Qwen3VL/{DEFAULT_RUN_NAME}"


def _select_split(ds: Any, split: str):
    if isinstance(ds, DatasetDict):
        if split in ds:
            return ds[split]
        if "train" in ds:
            return ds["train"]
        return ds[next(iter(ds.keys()))]
    return ds


def patch_source_loader(module: Any, data_root: Path, allow_hf_fallback: bool) -> None:
    original_load_source = module._load_source
    source_root = data_root / "data" / "sources" / "capability"

    def local_load_source(source_name: str, source_seed: int):
        cfg = module.SOURCE_CONFIGS[source_name]
        local_dir = source_root / source_name
        if not local_dir.exists():
            if allow_hf_fallback:
                return original_load_source(source_name, source_seed)
            raise FileNotFoundError(
                f"Missing materialized source {source_name}: {rel(local_dir)}\n"
                "Run: python src/01_download_data.py --group capability"
            )

        ds = load_from_disk(str(local_dir))
        ds = _select_split(ds, cfg["split"])
        if len(ds) > 0:
            # Upstream shuffles each loaded source with this same source_seed.
            ds = ds.shuffle(seed=source_seed)
        return ds

    module._load_source = local_load_source


def patch_text_only_sources(module: Any) -> None:
    """Keep only the original mixed recipe's text-source allocation."""
    text_names = list(TEXT_SOURCE_COUNTS)
    module.SOURCE_PORTIONS = TEXT_SOURCE_COUNTS.copy()
    module.SOURCE_CONFIGS = {
        name: module.SOURCE_CONFIGS[name]
        for name in text_names
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate text-only MHLC Capability Head raw parquet with upstream logic and local materialized sources."
    )
    ap.add_argument("--data-root", default="../mhlc_data")
    ap.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--model-family", default="qwen3_vl", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4"])
    ap.add_argument("--thinking-mode", default="off", choices=["auto", "on", "off"])
    ap.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    ap.add_argument("--save-root", default=DEFAULT_SAVE_REL)
    ap.add_argument("--total-qa-pairs", type=int, default=TEXT_TOTAL_QA_PAIRS)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen-chunk-size", type=int, default=128)
    ap.add_argument("--raw-shard-size", type=int, default=4000)
    ap.add_argument("--clean", action="store_true", help="Remove the output run directory before generating.")
    ap.add_argument(
        "--allow-hf-fallback",
        action="store_true",
        help="If a materialized source is missing, fall back to upstream HF load_dataset.",
    )
    args = ap.parse_args()

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)

    model_path = resolve_from_code_root(args.model_path)
    save_root = resolve_from_code_root(args.save_root)
    if args.clean:
        clean_path(save_root, [data_root], "capability raw run")

    module = load_upstream_module(
        "combined_all_datagen_multimodel.py",
        "mhlc_upstream_combined_all_datagen_multimodel",
    )
    patch_text_only_sources(module)
    patch_source_loader(module, data_root, allow_hf_fallback=bool(args.allow_hf_fallback))

    argv = [
        "combined_all_datagen_multimodel.py",
        "--model-id",
        rel(model_path),
        "--model-family",
        args.model_family,
        "--thinking-mode",
        args.thinking_mode,
        "--run-name",
        args.run_name,
        "--save-root",
        rel(save_root),
        "--total-qa-pairs",
        str(args.total_qa_pairs),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--seed",
        str(args.seed),
        "--gen-chunk-size",
        str(args.gen_chunk_size),
        "--raw-shard-size",
        str(args.raw_shard_size),
    ]

    print("[stage] capability raw generation")
    print(f"[sources] text_only={dict(TEXT_SOURCE_COUNTS)} total={args.total_qa_pairs}")
    print(f"[model] {rel(model_path)}")
    print(f"[output] {rel(save_root)}")
    with temporary_argv(argv):
        module.main()


if __name__ == "__main__":
    main()
