#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mhlc_data_prep.parallel import configure_parallel_context, contiguous_range

PARALLEL = configure_parallel_context()

CAPABILITY_DEFAULT_DATASET = (
    "../mhlc_data/data/train/Qwen3VL/"
    "Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/verified"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe MHLC Capability FFN intermediate neurons.")
    parser.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset-path", default=CAPABILITY_DEFAULT_DATASET)
    parser.add_argument("--data-root", default="../mhlc_data")
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--high-threshold", type=float, default=0.8)
    parser.add_argument("--low-threshold", type=float, default=0.5)
    parser.add_argument("--fallback-ratio", type=float, default=0.30)
    parser.add_argument("--top-ratio", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--use-down-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heatmap-top-n", type=int, default=300)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from dataclasses import asdict

    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from mhlc_neuron_probe.baseline_bridge import build_forward_inputs, load_capability_trainer_module, load_frozen_backbone, move_batch_to_device
    from mhlc_neuron_probe.data_tasks import (
        CapabilityDataConfig,
        capability_collator,
        capability_group_thresholds,
        capability_labels,
        load_capability_dataset,
    )
    from mhlc_neuron_probe.ffn_hooks import FFNActivationCollector, discover_ffn_intermediate_layers, down_weight_norms, module_meta_from_layers
    from mhlc_neuron_probe.io_utils import (
        maybe_clean,
        model_tag,
        output_dirs,
        prepare_data_root,
        rel,
        should_skip,
        stable_hash,
        write_csv,
        write_json,
        write_jsonl,
    )
    from mhlc_neuron_probe.selection import select_neurons
    from mhlc_neuron_probe.stats import (
        CapabilityRunningStats,
        capability_stats_state,
        merge_capability_stats,
        summary_from_module_meta,
    )
    from mhlc_neuron_probe.visualization import plot_capability_direction, plot_layer_top_score_heatmap, plot_selected_density

    upstream = load_capability_trainer_module()
    upstream.set_seed(int(args.seed))
    data_root = prepare_data_root(args.data_root)
    tag = model_tag(args.model_path)
    dirs = output_dirs(data_root, tag, "capability")
    if not PARALLEL.enabled or PARALLEL.is_main:
        maybe_clean(dirs["neurons"], data_root, "capability neuron artifacts", bool(args.clean))
        maybe_clean(dirs["viz"], data_root, "capability neuron visualizations", bool(args.clean))
    PARALLEL.barrier()
    dirs["neurons"].mkdir(parents=True, exist_ok=True)
    dirs["viz"].mkdir(parents=True, exist_ok=True)

    data_cfg = CapabilityDataConfig(
        dataset_path=args.dataset_path,
        aux_label_column=args.aux_label_column,
        subset_name=args.subset_name,
        seed=int(args.seed),
        max_samples=int(args.max_samples),
        max_seq_len=int(args.max_seq_len),
        head_input_mode=args.head_input_mode,
    )
    dataset = load_capability_dataset(data_cfg)
    labels = capability_labels(dataset, args.aux_label_column)
    group_info = capability_group_thresholds(labels, args.high_threshold, args.low_threshold, args.fallback_ratio)
    params = {
        "stage": "probe_capability_neurons",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "dataset_len": len(dataset),
        "labels_hash": stable_hash(labels.tolist()),
        "data_config": asdict(data_cfg),
        "group_info": group_info,
        "top_ratio": float(args.top_ratio),
        "min_score": float(args.min_score),
        "epsilon": float(args.epsilon),
        "use_down_norm": bool(args.use_down_norm),
        "parallel_workers": PARALLEL.world_size,
        "score_formula": "relu_z(weighted_separation)+relu_z(abs_correlation)+0.5*relu_z(weighted_responsiveness)",
    }
    selected_path = dirs["neurons"] / "selected_neurons.jsonl"
    score_path = dirs["neurons"] / "neuron_scores.pt"
    layer_csv = dirs["neurons"] / "layer_summary.csv"
    if should_skip(dirs["neurons"], params, [selected_path, score_path, layer_csv], bool(args.overwrite)):
        return

    worker_dataset = dataset
    if PARALLEL.enabled:
        start, end = contiguous_range(len(dataset), PARALLEL)
        worker_dataset = dataset.select(range(start, end))
        print(f"[parallel] rank={PARALLEL.rank}/{PARALLEL.world_size} samples={start}:{end}", flush=True)

    runtime = load_frozen_backbone(
        model_name_or_path=args.model_path,
        model_family=args.model_family,
        thinking_mode=args.thinking_mode,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
        prefer_unsloth_mirror=bool(args.prefer_unsloth_mirror),
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        dtype=args.dtype,
        max_seq_len=int(args.max_seq_len),
        max_pixels=int(args.max_pixels),
    )
    layers = discover_ffn_intermediate_layers(runtime.forward_model)
    module_meta = module_meta_from_layers(layers)
    down_norms = down_weight_norms(layers)
    stats = CapabilityRunningStats(
        high_threshold=float(group_info["high_threshold"]),
        low_threshold=float(group_info["low_threshold"]),
    )
    loader = DataLoader(
        worker_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=capability_collator(runtime.processor, data_cfg),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )
    with FFNActivationCollector(layers, save_dtype=torch.float32) as collector:
        for batch in tqdm(loader, desc="probe capability", dynamic_ncols=True):
            batch = move_batch_to_device(batch, runtime.device, runtime.fp_dtype)
            batch_labels = batch.pop("aux_labels").detach().cpu().float()
            token_mask = batch.pop("head_token_mask")
            collector.set_token_mask(token_mask)
            collector.clear()
            forward_inputs = build_forward_inputs(batch, runtime.model_family)
            autocast_enabled = runtime.device.type == "cuda" and runtime.fp_dtype in {torch.float16, torch.bfloat16}
            autocast_dtype = runtime.fp_dtype if runtime.fp_dtype is not None else torch.bfloat16
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    _ = runtime.forward_model(**forward_inputs, use_cache=False, return_dict=True)
            stats.update(collector.captures, batch_labels)

    if PARALLEL.enabled:
        partial_dir = dirs["neurons"] / ".probe_partials"
        partial_dir.mkdir(parents=True, exist_ok=True)
        torch.save(capability_stats_state(stats), partial_dir / f"rank_{PARALLEL.rank:02d}.pt")
        PARALLEL.barrier()
        if not PARALLEL.is_main:
            PARALLEL.barrier()
            return
        stats = merge_capability_stats(
            [torch.load(partial_dir / f"rank_{rank:02d}.pt", map_location="cpu") for rank in range(PARALLEL.world_size)]
        )

    score_pack = stats.scores(down_norms=down_norms, use_down_norm=bool(args.use_down_norm), eps=float(args.epsilon))
    rows, layer_rows = select_neurons(
        score_pack=score_pack,
        module_meta=module_meta,
        task="capability",
        top_ratio=float(args.top_ratio),
        min_score=float(args.min_score),
        model_tag=tag,
    )
    write_jsonl(selected_path, rows)
    write_csv(layer_csv, layer_rows)
    torch.save(
        {
            "task": "capability",
            "module_meta": module_meta,
            "scores": score_pack,
            "group_info": group_info,
            "summary": summary_from_module_meta(module_meta),
        },
        score_path,
    )
    plot_layer_top_score_heatmap(
        score_pack=score_pack,
        module_meta=module_meta,
        out_path=dirs["viz"] / "layer_top_neuron_score_heatmap.png",
        title="Capability: per-layer top neuron scores",
        top_n=int(args.heatmap_top_n),
    )
    plot_selected_density(layer_rows=layer_rows, out_path=dirs["viz"] / "selected_density_by_layer.png", title="Capability selected neuron density")
    plot_capability_direction(rows=rows, out_path=dirs["viz"] / "selected_direction_by_layer.png", title="Capability selected neuron direction")
    summary = {
        **summary_from_module_meta(module_meta),
        "selected_neurons": len(rows),
        "selected_ratio": len(rows) / max(summary_from_module_meta(module_meta)["total_ffn_neurons"], 1),
        "outputs": {
            "selected_neurons": rel(selected_path),
            "scores": rel(score_path),
            "layer_summary": rel(layer_csv),
            "visualization_dir": rel(dirs["viz"]),
        },
    }
    write_json(dirs["neurons"] / "manifest.json", {"params": params, "summary": summary})
    print(f"[write] {rel(selected_path)}", flush=True)
    print(f"[write] {rel(dirs['viz'])}", flush=True)
    PARALLEL.barrier()


if __name__ == "__main__":
    main()
