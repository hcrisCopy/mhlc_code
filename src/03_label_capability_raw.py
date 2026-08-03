#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from mhlc_data_prep.parallel import configure_parallel_context, contiguous_range

PARALLEL = configure_parallel_context()

from tqdm.auto import tqdm

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel, temporary_argv
from mhlc_data_prep.specs import TEXT_TOTAL_QA_PAIRS


DEFAULT_RUN_NAME = f"Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_{TEXT_TOTAL_QA_PAIRS}"
DEFAULT_RUN_ROOT = f"../mhlc_data/data/train/Qwen3VL/{DEFAULT_RUN_NAME}"


def patch_parallel_raw_paths(module, run_root: Path) -> None:
    """Filter upstream's raw-shard glob without changing its labeling code."""
    original_path = module.Path
    raw_dir = run_root / "raw"

    class RankRawDirectory:
        def __init__(self, path: Path):
            self.path = path

        def glob(self, pattern: str):
            paths = sorted(self.path.glob(pattern))
            start, end = contiguous_range(len(paths), PARALLEL)
            selected = paths[start:end]
            if PARALLEL.enabled:
                return tqdm(
                    selected,
                    desc=f"rank {PARALLEL.rank + 1}/{PARALLEL.world_size} label capability shards",
                    unit="shard",
                    dynamic_ncols=True,
                    position=PARALLEL.rank,
                )
            return selected

    def patched_path(*items):
        path = original_path(*items)
        if path.resolve() == raw_dir.resolve():
            return RankRawDirectory(path)
        return path

    module.Path = patched_path
    original_save_json = module._save_json

    def rank_safe_save_json(path, obj):
        target = Path(path)
        if target.name == "verification_stats.json":
            target = run_root / f".verification_stats.rank_{PARALLEL.rank:02d}.json"
        return original_save_json(target, obj)

    module._save_json = rank_safe_save_json


def merge_parallel_stats(run_root: Path) -> None:
    """Write the same aggregate verification summary a one-GPU run exposes."""
    merged = {
        "model_id": None,
        "num_raw_shards": 0,
        "processed_shards": 0,
        "skipped_shards": 0,
        "num_rows": 0,
        "num_judge_calls": 0,
        "per_subset": {},
        "parallel_workers": PARALLEL.world_size,
    }
    for rank in range(PARALLEL.world_size):
        path = run_root / f".verification_stats.rank_{rank:02d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing eight-GPU labeling summary: {rel(path)}")
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        merged["model_id"] = merged["model_id"] or payload.get("model_id")
        for key in ("num_raw_shards", "processed_shards", "skipped_shards", "num_rows", "num_judge_calls"):
            merged[key] += int(payload.get(key, 0))
        for subset, values in payload.get("per_subset", {}).items():
            dest = merged["per_subset"].setdefault(subset, {"rows": 0, "judge_rows": 0})
            dest["rows"] += int(values.get("rows", 0))
            dest["judge_rows"] += int(values.get("judge_rows", 0))
    (run_root / "verification_stats.json").write_text(
        __import__("json").dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[merge] capability label workers -> {rel(run_root / 'verification_stats.json')}", flush=True)


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
    if args.clean and (not PARALLEL.enabled or PARALLEL.is_main):
        clean_path(run_root / "verified", [data_root], "capability verified dir")
        clean_path(run_root / "verification_stats.json", [data_root], "capability verification stats")
        for rank in range(PARALLEL.world_size):
            clean_path(run_root / f".verification_stats.rank_{rank:02d}.json", [data_root], "capability worker verification stats")
    PARALLEL.barrier()

    module = load_upstream_module(
        "combined_all_labeling_multimodel.py",
        "mhlc_upstream_combined_all_labeling_multimodel",
    )
    if PARALLEL.enabled:
        patch_parallel_raw_paths(module, run_root)
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
    if PARALLEL.enabled:
        print(f"[parallel] rank={PARALLEL.rank}/{PARALLEL.world_size} raw shard partition", flush=True)
    with temporary_argv(argv):
        module.main()
    PARALLEL.barrier()
    if PARALLEL.enabled and PARALLEL.is_main:
        merge_parallel_stats(run_root)
    PARALLEL.barrier()


if __name__ == "__main__":
    main()
