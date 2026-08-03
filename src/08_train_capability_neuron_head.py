#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mhlc_data_prep.parallel import configure_parallel_context

PARALLEL = configure_parallel_context()

CAPABILITY_DEFAULT_DATASET = (
    "../mhlc_data/data/train/Qwen3VL/"
    "Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/verified"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MHLC Capability head from selected FFN neurons.")
    parser.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset-path", default=CAPABILITY_DEFAULT_DATASET)
    parser.add_argument("--data-root", default="../mhlc_data")
    parser.add_argument("--neuron-path", default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-family", default="auto")
    parser.add_argument("--thinking-mode", default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--prefer-unsloth-mirror", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--use-gradient-checkpointing", default="unsloth")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--max-seq-len", type=int, default=32000)
    parser.add_argument("--max-pixels", type=int, default=200000)
    parser.add_argument("--head-input-mode", default="completion_text_only")
    parser.add_argument("--aux-label-column", default="correctness_score")
    parser.add_argument("--subset-name", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--feature-shard-size", type=int, default=512)
    parser.add_argument("--feature-save-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.10)
    parser.add_argument("--failure-threshold", type=float, default=0.5)
    parser.add_argument("--severity-power", type=float, default=1.0)
    parser.add_argument("--min-class-weight", type=float, default=1.0)
    parser.add_argument("--max-class-weight", type=float, default=10.0)
    parser.add_argument("--metric-bins", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from mhlc_neuron_probe.feature_cache import ensure_feature_cache
    from mhlc_neuron_probe.io_utils import maybe_clean, model_tag, output_dirs, prepare_data_root, rel
    from mhlc_neuron_probe.train_utils import train_neuron_head

    data_root = prepare_data_root(args.data_root)
    tag = model_tag(args.model_path)
    dirs = output_dirs(data_root, tag, "capability")
    feature_dir = dirs["features"] if args.feature_cache_dir is None else data_root / args.feature_cache_dir
    out_dir = dirs["trained"] if args.output_dir is None else data_root / args.output_dir
    neuron_path = dirs["neurons"] / "selected_neurons.jsonl" if args.neuron_path is None else data_root / args.neuron_path
    if not PARALLEL.enabled or PARALLEL.is_main:
        maybe_clean(feature_dir, data_root, "capability feature cache", bool(args.clean))
        maybe_clean(out_dir, data_root, "capability neuron head", bool(args.clean))
    PARALLEL.barrier()
    manifest_path = ensure_feature_cache(args, task="capability", neuron_path=neuron_path, cache_dir=feature_dir)
    if not PARALLEL.enabled or PARALLEL.is_main:
        final_path = train_neuron_head(args, task="capability", manifest_path=manifest_path, out_dir=out_dir)
        print(f"[done] {rel(final_path)}", flush=True)
    PARALLEL.barrier()


if __name__ == "__main__":
    main()
