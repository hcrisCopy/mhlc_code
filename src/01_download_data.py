#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mhlc_data_prep.hf_local import materialize_hf_split, snapshot_dataset_repo
from mhlc_data_prep.paths import (
    ensure_mhlc_data_layout,
    resolve_from_code_root,
    set_hf_dirs_inside_data_root,
)
from mhlc_data_prep.specs import (
    BENCHMARK_DATASETS,
    CAPABILITY_SOURCE_DATASETS,
    CSV_BENCHMARK_TARGETS,
    SCREENSPOT_REPO,
    WHEN2CALL_CONFIGS,
    WHEN2CALL_REPO,
)
from mhlc_data_prep.run_utils import rel

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def parse_csv(value: str, allowed: set[str]) -> list[str]:
    if value.strip().lower() == "all":
        return sorted(allowed)
    out = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in out if x not in allowed]
    if bad:
        raise SystemExit(f"Unknown item(s): {bad}. Allowed: {sorted(allowed)}")
    return out


def progress(items: list[str], desc: str):
    if tqdm is None:
        print(desc)
        return items
    return tqdm(items, desc=desc, dynamic_ncols=True)


def copy_csv_benchmark(
    *,
    name: str,
    source_path: str | None,
    target_path: Path,
    overwrite: bool,
    strict: bool,
) -> str:
    if target_path.exists() and not overwrite:
        return "skipped"
    if not source_path:
        message = (
            f"CSV benchmark {name!r} requires a paper/local snapshot. "
            f"Pass --{name.replace('_', '-')}-csv to copy it to {rel(target_path)}."
        )
        if strict:
            raise FileNotFoundError(message)
        print(f"[missing] {message}")
        return "missing"

    src = resolve_from_code_root(source_path)
    if not src.exists():
        raise FileNotFoundError(f"CSV benchmark source not found for {name}: {rel(src)}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target_path)
    return "copied"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Materialize MHLC upstream datasets under ../mhlc_data, not the default HF cache."
    )
    ap.add_argument("--data-root", default="../mhlc_data")
    ap.add_argument(
        "--group",
        default="all",
        choices=["all", "capability", "benchmarks", "when2call"],
        help="Which dataset family to materialize.",
    )
    ap.add_argument("--capability-sources", default="all")
    ap.add_argument("--benchmarks", default="all")
    ap.add_argument("--math-csv", default=None, help="Existing merged_math.csv snapshot to copy into mhlc_data.")
    ap.add_argument("--mmlu-pro-csv", default=None, help="Existing mmlu_pro test.csv snapshot to copy into mhlc_data.")
    ap.add_argument(
        "--strict-benchmark-csv",
        action="store_true",
        help="Fail if math/mmlu_pro is selected but the required CSV snapshot is not provided.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--clean", action="store_true", help="Alias for --overwrite on selected materialized outputs.")
    ap.add_argument(
        "--keep-runtime-cache",
        action="store_true",
        help="Keep temporary HF cache under mhlc_data/downloads/hf_runtime_cache.",
    )
    args = ap.parse_args()

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    runtime_cache = data_root / "downloads" / "hf_runtime_cache"
    dataset_cache = runtime_cache / "datasets"
    hub_cache = runtime_cache / "hub"
    overwrite_outputs = bool(args.overwrite or args.clean)

    print(f"[paths] data_root={rel(data_root)}")
    print(f"[paths] temporary_hf_runtime_cache={rel(runtime_cache)}")

    if args.group in {"all", "capability"}:
        selected = parse_csv(args.capability_sources, set(CAPABILITY_SOURCE_DATASETS))
        for name in progress(selected, "capability sources"):
            cfg = CAPABILITY_SOURCE_DATASETS[name]
            out_dir = data_root / "data" / "sources" / "capability" / name
            status = materialize_hf_split(
                dataset_id=cfg["dataset_id"],
                dataset_config=cfg.get("dataset_config"),
                split=cfg["split"],
                out_dir=out_dir,
                cache_dir=dataset_cache,
                overwrite=overwrite_outputs,
            )
            print(f"[{status}] {name} -> {rel(out_dir)}")

    if args.group in {"all", "benchmarks"}:
        allowed_benchmarks = set(BENCHMARK_DATASETS) | {"screenspot_pro"} | set(CSV_BENCHMARK_TARGETS)
        if args.benchmarks.strip().lower() == "all":
            # The upstream repo does not ship the paper CSV snapshots for
            # math/mmlu_pro, so "all" means all downloadable benchmarks.
            selected_benchmarks = sorted(set(BENCHMARK_DATASETS) | {"screenspot_pro"})
        else:
            selected_benchmarks = parse_csv(args.benchmarks, allowed_benchmarks)
        for name in progress(selected_benchmarks, "benchmarks"):
            if name in CSV_BENCHMARK_TARGETS:
                source_path = args.math_csv if name == "math" else args.mmlu_pro_csv
                target_path = data_root / "data" / "benchmarks" / CSV_BENCHMARK_TARGETS[name]
                status = copy_csv_benchmark(
                    name=name,
                    source_path=source_path,
                    target_path=target_path,
                    overwrite=overwrite_outputs,
                    strict=args.strict_benchmark_csv,
                )
                print(f"[{status}] {name} -> {rel(target_path)}")
                continue

            if name == "screenspot_pro":
                out_dir = data_root / "data" / "benchmarks" / "screenspot_pro"
                status = snapshot_dataset_repo(
                    repo_id=SCREENSPOT_REPO,
                    out_dir=out_dir,
                    cache_dir=hub_cache,
                    allow_patterns=["annotations/*.json", "images/**"],
                    overwrite=overwrite_outputs,
                )
                print(f"[{status}] {name} -> {rel(out_dir)}")
                continue

            cfg = BENCHMARK_DATASETS[name]
            out_dir = data_root / "data" / "benchmarks" / name / "dataset"
            status = materialize_hf_split(
                dataset_id=cfg["dataset_id"],
                dataset_config=cfg.get("dataset_config"),
                split=cfg["split"],
                out_dir=out_dir,
                cache_dir=dataset_cache,
                overwrite=overwrite_outputs,
            )
            print(f"[{status}] {name} -> {rel(out_dir)}")

    if args.group in {"all", "when2call"}:
        for config_name in progress(WHEN2CALL_CONFIGS, "when2call"):
            out_dir = data_root / "data" / "sources" / "when2call" / config_name
            split = "mcq" if config_name == "test" else "train"
            status = materialize_hf_split(
                dataset_id=WHEN2CALL_REPO,
                dataset_config=config_name,
                split=split,
                out_dir=out_dir,
                cache_dir=dataset_cache,
                overwrite=overwrite_outputs,
            )
            print(f"[{status}] when2call/{config_name} -> {rel(out_dir)}")

    if not args.keep_runtime_cache and runtime_cache.exists():
        shutil.rmtree(runtime_cache)
        print(f"[clean] removed temporary HF runtime cache: {rel(runtime_cache)}")

    print("[done] dataset materialization complete")


if __name__ == "__main__":
    main()
