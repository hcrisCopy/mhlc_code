#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mhlc_data_prep.parallel import (
    configure_parallel_context,
    contiguous_range,
    require_single_gpu_vllm,
    worker_dir,
)

# Must happen before an upstream module imports torch / vLLM.
PARALLEL = configure_parallel_context()

from datasets import DatasetDict, load_from_disk
from tqdm.auto import tqdm

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


def parse_source_counts(value: str | None) -> dict[str, int] | None:
    """Parse smoke-run counts like dapo=20,triviaqa=20,apigen-mt-5k=20."""
    if value is None:
        return None

    allowed = set(TEXT_SOURCE_COUNTS)
    counts: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid --source-counts item {item!r}; expected name=count.")
        name, raw_count = [part.strip() for part in item.split("=", 1)]
        if name not in allowed:
            raise SystemExit(f"Unknown source {name!r}; allowed: {sorted(allowed)}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise SystemExit(f"Invalid count for {name!r}: {raw_count!r}") from exc
        if count < 0:
            raise SystemExit(f"Count for {name!r} must be >= 0, got {count}.")
        counts[name] = count

    missing = [name for name in TEXT_SOURCE_COUNTS if name not in counts]
    if missing:
        raise SystemExit(f"--source-counts must include every text source; missing: {missing}")
    if sum(counts.values()) <= 0:
        raise SystemExit("--source-counts total must be > 0.")
    return {name: counts[name] for name in TEXT_SOURCE_COUNTS}


def patch_exact_source_counts(module: Any, source_counts: dict[str, int]) -> None:
    """Make upstream count allocation return the smoke-run counts exactly."""

    exact_counts = source_counts.copy()

    def allocate_exact_counts(total: int, portions: Any):
        expected_total = sum(exact_counts.values())
        if int(total) != expected_total:
            raise ValueError(f"Expected total_qa_pairs={expected_total}, got {total}.")
        return module.OrderedDict((name, exact_counts[name]) for name in exact_counts)

    module.SOURCE_PORTIONS = exact_counts.copy()
    module._allocate_counts = allocate_exact_counts


def patch_generation_progress(module: Any) -> None:
    """Mirror upstream generation logic while adding a source-level tqdm bar."""

    def generate_source_with_progress(
        llm: Any,
        ds: Any,
        source_name: str,
        pending_specs: list[dict[str, Any]],
        sampling_bundle: dict[str, Any],
        shard_rows: list[dict[str, Any]],
        shard_idx: int,
        completed_counts: dict[Any, int],
        total_requests: int,
        total_outputs: int,
    ):
        conversations: list[Any] = []
        chunk_specs: list[dict[str, Any]] = []
        chunk_saved_images: list[Any] = []
        source_requests_done = 0
        first_flush = True
        modality = module.SOURCE_CONFIGS[source_name]["modality"]

        rank_prefix = (
            f"rank {PARALLEL.rank + 1}/{PARALLEL.world_size} "
            if PARALLEL.enabled
            else ""
        )
        progress_bar = tqdm(
            total=len(pending_specs),
            desc=f"{rank_prefix}generate {source_name}",
            unit="req",
            dynamic_ncols=True,
            position=PARALLEL.rank if PARALLEL.enabled else 0,
            leave=True,
        )

        def flush_chunk(start_row_marker: int) -> None:
            nonlocal conversations, chunk_specs, chunk_saved_images
            nonlocal shard_rows, shard_idx, total_requests, total_outputs, first_flush
            if not conversations:
                return
            if module.DEBUG_MODE and (first_flush or module.DEBUG_PRINT_REQUEST_DETAILS_EVERY_CHUNK):
                preview_n = min(module.DEBUG_PROMPT_PREVIEWS, len(chunk_specs))
                module._debug(
                    f"[debug][generate] previewing {preview_n} {modality} request(s) "
                    f"for source={source_name} near row {start_row_marker}"
                )
                for i in range(preview_n):
                    spec = chunk_specs[i]
                    module._debug(
                        f"  request#{i+1} subset={spec['subset_name']} "
                        f"row_index={spec['row_index']} qa_index={spec['qa_index']} "
                        f"turn_index={spec['turn_index']} "
                        f"q={module._preview_text(spec.get('question', ''))}"
                    )

            chat_kwargs = module._chat_template_kwargs_for_runtime()
            llm_chat_kwargs: dict[str, Any] = {
                "sampling_params": sampling_bundle["params"],
                "use_tqdm": False,
            }
            if chat_kwargs:
                llm_chat_kwargs["chat_template_kwargs"] = chat_kwargs
            outputs = llm.chat(conversations, **llm_chat_kwargs)
            total_requests += len(conversations)

            debug_completion_printed = 0
            for spec, saved_images, out in zip(chunk_specs, chunk_saved_images, outputs):
                for gen_idx, candidate in enumerate(out.outputs):
                    completion = candidate.text or ""
                    raw = module._default_raw_row()
                    raw.update(spec)
                    raw.update(
                        {
                            "images": saved_images,
                            "completion": completion,
                            "generation_index": int(gen_idx),
                            "completion_length": len(completion),
                            "two_step_applied": False,
                        }
                    )
                    shard_rows.append(raw)
                    total_outputs += 1
                    spec_key = module._spec_identity(spec)
                    completed_counts[spec_key] = completed_counts.get(spec_key, 0) + 1
                    shard_rows, shard_idx = module._flush_save_if_needed(shard_rows, shard_idx)
                    if (
                        module.DEBUG_MODE
                        and (first_flush or module.DEBUG_PRINT_COMPLETION_DETAILS_EVERY_CHUNK)
                        and debug_completion_printed < module.DEBUG_COMPLETION_PREVIEWS
                    ):
                        debug_completion_printed += 1
                        module._debug(
                            f"[debug][completion] subset={spec['subset_name']} "
                            f"row_index={spec['row_index']} qa_index={spec['qa_index']} "
                            f"turn_index={spec['turn_index']} model={module._preview_text(completion)}"
                        )

            progress_bar.update(len(conversations))
            progress_bar.set_postfix(
                shard=shard_idx,
                outputs=total_outputs,
                refresh=False,
            )
            conversations.clear()
            chunk_specs.clear()
            chunk_saved_images.clear()
            first_flush = False
            module.torch.cuda.empty_cache()
            module.gc.collect()

        try:
            for spec in pending_specs:
                conversation, raw_images = module._build_conversation(spec, ds)
                conversations.append(conversation)
                chunk_specs.append(spec)
                chunk_saved_images.append(raw_images)
                source_requests_done += 1
                if len(conversations) >= module.GEN_CHUNK_SIZE:
                    flush_chunk(int(spec["row_index"]))

            flush_chunk(-1)
            progress_bar.set_postfix(
                shard=shard_idx,
                outputs=total_outputs,
                refresh=True,
            )
        finally:
            progress_bar.close()

        print(f"[generate] source={source_name} finished with {source_requests_done}/{len(pending_specs)} pending requests done")
        return shard_rows, shard_idx, total_requests, total_outputs

    module._generate_source = generate_source_with_progress


def patch_parallel_source_partition(module: Any) -> None:
    """Keep upstream sampling intact, then give each worker a disjoint slice."""
    original_sample_source = module._sample_source

    def partitioned_sample_source(ds: Any, source_name: str, target_count: int):
        selected, stats = original_sample_source(ds, source_name, target_count)
        start, end = contiguous_range(len(selected), PARALLEL)
        part = selected[start:end]
        stats = dict(stats)
        stats.update(
            {
                "selected_before_partition": len(selected),
                "selected": len(part),
                "parallel_rank": PARALLEL.rank,
                "parallel_world_size": PARALLEL.world_size,
                "parallel_start": start,
                "parallel_end": end,
            }
        )
        return part, stats

    module._sample_source = partitioned_sample_source


def _parallel_completion_marker(save_root: Path) -> Path:
    return save_root / ".capability_raw_eight_gpu_complete.json"


def _merge_parallel_raw_outputs(module: Any, save_root: Path, work_root: Path, args: argparse.Namespace) -> None:
    """Restore the ordinary upstream raw layout after all workers finish."""
    raw_rows: list[dict[str, Any]] = []
    for rank in range(PARALLEL.world_size):
        rank_raw = work_root / f"rank_{rank:02d}" / "raw"
        for shard_path in sorted(rank_raw.glob("shard-*.parquet")):
            raw_rows.extend(module.load_dataset("parquet", data_files=str(shard_path), split="train").to_list())

    source_order = {name: index for index, name in enumerate(TEXT_SOURCE_COUNTS)}
    raw_rows.sort(
        key=lambda row: (
            source_order.get(str(row.get("subset_name", "")), len(source_order)),
            int(row.get("row_index", -1)),
            int(row.get("qa_index", -1)),
            int(row.get("turn_index", -1)),
            int(row.get("generation_index", 0)),
        )
    )
    raw_dir = save_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard_size = int(args.raw_shard_size)
    for shard_index, start in enumerate(range(0, len(raw_rows), shard_size), start=1):
        module._save_shard(raw_rows[start:start + shard_size], raw_dir, shard_index)

    requested_counts = module._allocate_counts(int(args.total_qa_pairs), module.SOURCE_PORTIONS)
    module._save_json(
        save_root / "selection_manifest.json",
        {
            "model_id": str(args.model_path),
            "model_family": args.model_family,
            "thinking_mode_requested": args.thinking_mode,
            "total_examples_requested": int(args.total_qa_pairs),
            "requested_source_counts": dict(requested_counts),
            "parallel_workers": PARALLEL.world_size,
            "merge_order": "upstream source order, then source row / turn identity",
        },
    )
    module._save_json(
        save_root / "generation_stats.json",
        {
            "model_id": str(args.model_path),
            "parallel_workers": PARALLEL.world_size,
            "total_selected_examples": len(raw_rows),
            "total_outputs": len(raw_rows),
            "raw_shards": len(list(raw_dir.glob("shard-*.parquet"))),
        },
    )
    _parallel_completion_marker(save_root).write_text(
        json.dumps({"world_size": PARALLEL.world_size, "rows": len(raw_rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[merge] capability raw workers -> {rel(raw_dir)} rows={len(raw_rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate text-only MHLC Capability Head raw parquet with upstream logic and local materialized sources."
    )
    ap.add_argument("--data-root", default="../mhlc_data")
    ap.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--model-family", default="qwen3_vl", choices=["auto", "qwen3_5", "qwen3", "qwen3_vl", "gemma4"])
    ap.add_argument("--thinking-mode", default="off", choices=["auto", "on", "off"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-root", default=None)
    ap.add_argument("--total-qa-pairs", type=int, default=TEXT_TOTAL_QA_PAIRS)
    ap.add_argument(
        "--source-counts",
        default=None,
        help=(
            "Smoke-run exact counts, for example "
            "dapo=20,triviaqa=20,apigen-mt-5k=20. "
            "Omit this for the formal full text-only recipe."
        ),
    )
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

    require_single_gpu_vllm(args.tensor_parallel_size, PARALLEL)

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    source_counts = parse_source_counts(args.source_counts)
    if source_counts is not None:
        args.total_qa_pairs = sum(source_counts.values())
    if args.run_name is None:
        if source_counts is None:
            args.run_name = DEFAULT_RUN_NAME
        else:
            args.run_name = f"Qwen3_VL_4B_Instruct_text_only_smoke_{args.total_qa_pairs}"
    if args.save_root is None:
        args.save_root = f"../mhlc_data/data/train/Qwen3VL/{args.run_name}"

    model_path = resolve_from_code_root(args.model_path)
    save_root = resolve_from_code_root(args.save_root)
    if PARALLEL.enabled and _parallel_completion_marker(save_root).exists() and not args.clean:
        if PARALLEL.is_main:
            print(f"[skip] completed eight-GPU capability raw run: {rel(save_root)}", flush=True)
        return
    if PARALLEL.enabled and save_root.exists() and not args.clean:
        raise FileExistsError(
            f"Existing output is not an eight-GPU completed run: {rel(save_root)}. "
            "Use --clean to start a new eight-GPU run without mixing artifacts."
        )
    work_root = worker_dir(save_root, "capability_raw", PARALLEL).parent
    if args.clean and (not PARALLEL.enabled or PARALLEL.is_main):
        clean_path(save_root, [data_root], "capability raw run")
        if PARALLEL.enabled:
            clean_path(work_root, [data_root], "capability raw worker cache")
    PARALLEL.barrier()

    module = load_upstream_module(
        "combined_all_datagen_multimodel.py",
        "mhlc_upstream_combined_all_datagen_multimodel",
    )
    patch_text_only_sources(module)
    if source_counts is not None:
        patch_exact_source_counts(module, source_counts)
    patch_source_loader(module, data_root, allow_hf_fallback=bool(args.allow_hf_fallback))
    patch_generation_progress(module)
    worker_save_root = save_root
    if PARALLEL.enabled:
        patch_parallel_source_partition(module)
        worker_save_root = worker_dir(save_root, "capability_raw", PARALLEL)

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
        rel(worker_save_root),
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
    print(f"[sources] text_only={dict(module.SOURCE_PORTIONS)} total={args.total_qa_pairs}")
    if source_counts is not None:
        print("[mode] smoke source counts override is active; omit --source-counts for formal full run")
    print(f"[model] {rel(model_path)}")
    print(f"[output] {rel(save_root)}")
    if PARALLEL.enabled:
        print(f"[parallel] rank={PARALLEL.rank}/{PARALLEL.world_size} worker_output={rel(worker_save_root)}")
    with temporary_argv(argv):
        module.main()
    PARALLEL.barrier()
    if PARALLEL.enabled and PARALLEL.is_main:
        _merge_parallel_raw_outputs(module, save_root, work_root, args)
    PARALLEL.barrier()


if __name__ == "__main__":
    main()
