#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel, temporary_argv


DEFAULT_RUN_NAME = "Qwen3_VL_4B_Instruct_hard_Mixed_Sources_120k"
DEFAULT_RUN_ROOT = f"../mhlc_data/data/train/Qwen3VL/{DEFAULT_RUN_NAME}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Label generated Capability Head raw shards with upstream MHLC labeling logic."
    )
    ap.add_argument("--data-root", default="../mhlc_data")
    ap.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    ap.add_argument(
        "--judge-model-id",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        help="Upstream default for mixed-data correctness labeling.",
    )
    ap.add_argument("--judge-batch-size", type=int, default=64)
    ap.add_argument("--clean", action="store_true", help="Remove existing verified labels for this run before labeling.")
    args = ap.parse_args()

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    run_root = resolve_from_code_root(args.run_root)
    if args.clean:
        clean_path(run_root / "verified", [data_root], "capability verified dir")
        clean_path(run_root / "verification_stats.json", [data_root], "capability verification stats")

    module = load_upstream_module(
        "combined_all_labeling_multimodel.py",
        "mhlc_upstream_combined_all_labeling_multimodel",
    )
    argv = [
        "combined_all_labeling_multimodel.py",
        "--run-root",
        rel(run_root),
        "--judge-model-id",
        args.judge_model_id,
        "--judge-batch-size",
        str(args.judge_batch_size),
    ]

    print("[stage] capability raw labeling")
    print(f"[input] {rel(run_root / 'raw')}")
    print(f"[output] {rel(run_root / 'verified')}")
    print(f"[judge] {args.judge_model_id}")
    with temporary_argv(argv):
        module.main()


if __name__ == "__main__":
    main()
