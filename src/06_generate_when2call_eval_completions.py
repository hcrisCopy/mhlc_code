#!/usr/bin/env python3
from __future__ import annotations

import argparse

from datasets import DatasetDict, load_from_disk

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel, temporary_argv
from mhlc_data_prep.specs import WHEN2CALL_REPO


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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-hf-fallback", action="store_true")
    ap.add_argument("--clean", action="store_true", help="Remove existing eval completion outputs before generating.")
    args = ap.parse_args()

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    model_path = resolve_from_code_root(args.model_path)
    tokenizer_path = resolve_from_code_root(args.tokenizer_path) if args.tokenizer_path else model_path
    output_path = resolve_from_code_root(args.output_path)
    if args.clean:
        clean_path(output_path, [data_root], "when2call eval parquet")
        clean_path(output_path.with_suffix(".stats.json"), [data_root], "when2call eval stats")
        clean_path(output_path.with_suffix(".jsonl"), [data_root], "when2call eval jsonl")
    elif output_path.exists():
        print(f"[skip] eval completion already exists: {rel(output_path)}")
        print("[hint] pass --clean to regenerate it")
        return

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
        rel(output_path),
        "--also_write_jsonl",
        "true",
    ])

    print("[stage] optional when2call eval completion generation")
    print(f"[model] {rel(model_path)}")
    print(f"[output] {rel(output_path)}")
    if args.max_eval_rows is not None:
        print(f"[mode] smoke max_eval_rows={args.max_eval_rows}; omit it for formal full run")
    with temporary_argv(argv):
        module.main()


if __name__ == "__main__":
    main()
