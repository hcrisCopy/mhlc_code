#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mhlc_data_prep.original import load_upstream_module
from mhlc_data_prep.paths import ensure_mhlc_data_layout, resolve_from_code_root, set_hf_dirs_inside_data_root
from mhlc_data_prep.run_utils import clean_path, rel, temporary_argv


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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="Remove existing When2Call completions before generating.")
    args = ap.parse_args()

    data_root = resolve_from_code_root(args.data_root)
    ensure_mhlc_data_layout(data_root)
    set_hf_dirs_inside_data_root(data_root)
    input_path = resolve_from_code_root(args.input_path)
    output_dir = resolve_from_code_root(args.output_dir)
    model_path = resolve_from_code_root(args.model_path)
    tokenizer_path = resolve_from_code_root(args.tokenizer_path) if args.tokenizer_path else model_path
    if args.clean:
        clean_path(output_dir, [data_root], "when2call completion dir")

    module = load_upstream_module(
        "when2call/when2call_generate_completions_4class.py",
        "mhlc_upstream_when2call_generate_completions_4class",
    )
    argv = [
        "when2call_generate_completions_4class.py",
        "--input_path",
        rel(input_path),
        "--output_dir",
        rel(output_dir),
        "--model_id",
        rel(model_path),
        "--tokenizer_id",
        rel(tokenizer_path),
        "--model_family",
        args.model_family,
        "--thinking_mode",
        args.thinking_mode,
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
    ]

    print("[stage] when2call target-model completion generation")
    print(f"[input] {rel(input_path)}")
    print(f"[model] {rel(model_path)}")
    print(f"[output] {rel(output_dir)}")
    with temporary_argv(argv):
        module.main()


if __name__ == "__main__":
    main()
