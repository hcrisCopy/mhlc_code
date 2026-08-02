#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from mhlc_data_prep.hf_local import materialize_hf_split
from mhlc_data_prep.paths import (
    ensure_mhlc_data_layout,
    resolve_from_code_root,
    set_hf_dirs_inside_data_root,
)
from mhlc_data_prep.specs import (
    BENCHMARK_DATASETS,
    CAPABILITY_SOURCE_DATASETS,
    CSV_BENCHMARK_TARGETS,
    PUBLIC_CSV_BENCHMARK_SOURCES,
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


def _first_nonempty(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None and str(row[key]).strip():
            return row[key]
    return ""


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _extract_boxed_answer(text: str) -> str:
    raw = str(text or "")
    marker_matches = list(re.finditer(r"\\boxed\s*\{", raw))
    if not marker_matches:
        return ""
    start = marker_matches[-1].end() - 1
    depth = 0
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if ch == "{":
            depth += 1
            if depth == 1:
                answer_start = idx + 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[answer_start:idx].strip()
    return ""


def _choice_letter(index: int) -> str:
    return chr(ord("A") + int(index))


def _normalize_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_as_text(item) for item in value if _as_text(item)]
    if isinstance(value, tuple):
        return [_as_text(item) for item in value if _as_text(item)]
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [_as_text(item) for item in parsed if _as_text(item)]
        except Exception:
            pass
    return []


def _mmlu_answer_letter(row: dict[str, Any], options: list[str]) -> str:
    answer = _as_text(_first_nonempty(row, ["answer", "target", "final_answer"]))
    if len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return answer.upper()
    normalized_answer = answer.strip().lower()
    for idx, option in enumerate(options):
        if option.strip().lower() == normalized_answer:
            return _choice_letter(idx)

    index = row.get("answer_index")
    if index is None:
        index = row.get("answer_idx")
    if index is None:
        index = row.get("label")
    try:
        idx = int(index)
        if 0 <= idx < 26:
            return _choice_letter(idx)
    except Exception:
        pass

    return answer


def _load_public_csv_benchmark_dataset(name: str, cache_dir: Path):
    from datasets import load_dataset

    errors = []
    for cfg in PUBLIC_CSV_BENCHMARK_SOURCES[name]["candidates"]:
        dataset_id = cfg["dataset_id"]
        dataset_config = cfg.get("dataset_config")
        split = cfg["split"]
        try:
            if dataset_config:
                ds = load_dataset(dataset_id, dataset_config, split=split, cache_dir=str(cache_dir))
            else:
                ds = load_dataset(dataset_id, split=split, cache_dir=str(cache_dir))
            return ds, cfg
        except Exception as exc:
            errors.append(f"{dataset_id}/{split}: {type(exc).__name__}: {exc}")
    joined = "\n  - ".join(errors)
    raise RuntimeError(f"Could not load public fallback for {name}. Tried:\n  - {joined}")


def _select_public_benchmark_rows(ds, sample_size: int, seed: int):
    if sample_size > 0 and len(ds) > sample_size:
        ds = ds.shuffle(seed=int(seed)).select(range(int(sample_size)))
    return ds


def _write_public_math_csv(ds, target_path: Path, source_cfg: dict[str, Any]) -> None:
    fields = [
        "id",
        "sample_idx",
        "question",
        "solution",
        "answer",
        "level",
        "type",
        "source_dataset",
        "source_split",
    ]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(ds):
            row = dict(row)
            solution = _as_text(_first_nonempty(row, ["solution", "rationale", "steps", "explanation", "cot"]))
            answer = _as_text(_first_nonempty(row, ["answer", "final_answer", "target"]))
            if not answer:
                answer = _extract_boxed_answer(solution)
            question = _as_text(_first_nonempty(row, ["problem", "question", "prompt", "query", "input"]))
            writer.writerow(
                {
                    "id": _as_text(_first_nonempty(row, ["id", "example_id", "question_id"])) or str(idx),
                    "sample_idx": idx,
                    "question": question,
                    "solution": solution,
                    "answer": answer,
                    "level": _as_text(row.get("level")),
                    "type": _as_text(_first_nonempty(row, ["type", "category", "subject"])),
                    "source_dataset": source_cfg["dataset_id"],
                    "source_split": source_cfg["split"],
                }
            )


def _write_public_mmlu_pro_csv(ds, target_path: Path, source_cfg: dict[str, Any]) -> None:
    fields = [
        "id",
        "sample_idx",
        "question",
        "answer",
        "answer_index",
        "category",
        "source",
        "choices_json",
        "source_dataset",
        "source_split",
    ]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(ds):
            row = dict(row)
            base_question = _as_text(_first_nonempty(row, ["question", "prompt", "query", "input", "instruction"]))
            options = _normalize_options(_first_nonempty(row, ["options", "choices", "candidates", "mcq_choices"]))
            choices_block = "\n".join(f"{_choice_letter(i)}. {option}" for i, option in enumerate(options))
            question = base_question
            if choices_block:
                question = f"{base_question}\n\nChoices:\n{choices_block}"
            answer = _mmlu_answer_letter(row, options)
            writer.writerow(
                {
                    "id": _as_text(_first_nonempty(row, ["question_id", "id", "example_id"])) or str(idx),
                    "sample_idx": idx,
                    "question": question,
                    "answer": answer,
                    "answer_index": _as_text(_first_nonempty(row, ["answer_index", "answer_idx", "label"])),
                    "category": _as_text(_first_nonempty(row, ["category", "subject", "task"])),
                    "source": _as_text(_first_nonempty(row, ["src", "source"])),
                    "choices_json": json.dumps(options, ensure_ascii=False),
                    "source_dataset": source_cfg["dataset_id"],
                    "source_split": source_cfg["split"],
                }
            )


def materialize_public_csv_benchmark(
    *,
    name: str,
    target_path: Path,
    cache_dir: Path,
    overwrite: bool,
    sample_size: int,
    seed: int,
) -> str:
    done_marker = target_path.with_suffix(target_path.suffix + ".manifest.json")
    if target_path.exists() and not overwrite:
        return "skipped"

    ds, source_cfg = _load_public_csv_benchmark_dataset(name, cache_dir)
    effective_sample_size = sample_size
    if effective_sample_size <= 0:
        effective_sample_size = int(PUBLIC_CSV_BENCHMARK_SOURCES[name].get("sample_size", 1000))
    ds = _select_public_benchmark_rows(ds, sample_size=effective_sample_size, seed=seed)

    if name == "math":
        _write_public_math_csv(ds, target_path, source_cfg)
    elif name == "mmlu_pro":
        _write_public_mmlu_pro_csv(ds, target_path, source_cfg)
    else:
        raise ValueError(f"Unsupported public CSV benchmark: {name}")

    manifest = {
        "benchmark": name,
        "format": "csv",
        "target_path": str(target_path),
        "num_rows": len(ds),
        "sample_size": effective_sample_size,
        "seed": int(seed),
        "source": source_cfg,
        "note": "Public HF fallback generated because the upstream MHLC paper CSV snapshot is not included.",
    }
    done_marker.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return "generated"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Materialize text-only MHLC upstream datasets under ../mhlc_data, not the default HF cache."
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
    ap.add_argument("--csv-benchmark-sample-size", type=int, default=1000)
    ap.add_argument("--csv-benchmark-seed", type=int, default=42)
    ap.add_argument(
        "--strict-benchmark-csv",
        action="store_true",
        help="Fail if math/mmlu_pro is selected but the paper CSV snapshot is not provided.",
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
        allowed_benchmarks = set(BENCHMARK_DATASETS) | set(CSV_BENCHMARK_TARGETS)
        if args.benchmarks.strip().lower() == "all":
            selected_benchmarks = sorted(allowed_benchmarks)
        else:
            selected_benchmarks = parse_csv(args.benchmarks, allowed_benchmarks)
        for name in progress(selected_benchmarks, "benchmarks"):
            if name in CSV_BENCHMARK_TARGETS:
                source_path = args.math_csv if name == "math" else args.mmlu_pro_csv
                target_path = data_root / "data" / "benchmarks" / CSV_BENCHMARK_TARGETS[name]
                if source_path:
                    status = copy_csv_benchmark(
                        name=name,
                        source_path=source_path,
                        target_path=target_path,
                        overwrite=overwrite_outputs,
                        strict=args.strict_benchmark_csv,
                    )
                elif args.strict_benchmark_csv:
                    status = copy_csv_benchmark(
                        name=name,
                        source_path=source_path,
                        target_path=target_path,
                        overwrite=overwrite_outputs,
                        strict=True,
                    )
                else:
                    status = materialize_public_csv_benchmark(
                        name=name,
                        target_path=target_path,
                        cache_dir=dataset_cache,
                        overwrite=overwrite_outputs,
                        sample_size=int(args.csv_benchmark_sample_size),
                        seed=int(args.csv_benchmark_seed),
                    )
                print(f"[{status}] {name} -> {rel(target_path)}")
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
