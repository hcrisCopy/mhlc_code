#!/usr/bin/env python3
from __future__ import annotations

import argparse

RESOLUTION_DEFAULT_DATASET = "../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe MHLC Resolution FFN intermediate neurons.")
    parser.add_argument("--model-path", default="../Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dataset-path", default=RESOLUTION_DEFAULT_DATASET)
    parser.add_argument("--data-root", default="../mhlc_data")
    parser.add_argument("--model-family", default="auto")
    parser.add_argument("--thinking-mode", default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--prefer-unsloth-mirror", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--use-gradient-checkpointing", default="unsloth")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "auto"])
    parser.add_argument("--max-seq-len", type=int, default=32000)
    parser.add_argument("--max-pixels", type=int, default=200000)
    parser.add_argument("--head-input-mode", default="completion_text_only")
    parser.add_argument("--max-head-input-tokens", type=int, default=None)
    parser.add_argument("--subset-name", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--top-ratio", type=float, default=0.10)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-class-count", type=int, default=2)
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
        ResolutionDataConfig,
        load_resolution_dataset,
        resolution_collator,
        resolution_targets,
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
    from mhlc_neuron_probe.stats import ResolutionRunningStats, summary_from_module_meta
    from mhlc_neuron_probe.visualization import plot_layer_top_score_heatmap, plot_resolution_classes, plot_selected_density

    upstream = load_capability_trainer_module()
    upstream.set_seed(int(args.seed))
    data_root = prepare_data_root(args.data_root)
    tag = model_tag(args.model_path)
    dirs = output_dirs(data_root, tag, "resolution")
    maybe_clean(dirs["neurons"], data_root, "resolution neuron artifacts", bool(args.clean))
    maybe_clean(dirs["viz"], data_root, "resolution neuron visualizations", bool(args.clean))
    dirs["neurons"].mkdir(parents=True, exist_ok=True)
    dirs["viz"].mkdir(parents=True, exist_ok=True)

    data_cfg = ResolutionDataConfig(
        dataset_path=args.dataset_path,
        subset_name=args.subset_name,
        seed=int(args.seed),
        max_samples=int(args.max_samples),
        max_seq_len=int(args.max_seq_len),
        head_input_mode=args.head_input_mode,
        max_head_input_tokens=args.max_head_input_tokens,
        drop_unusable_rows=True,
    )
    dataset = load_resolution_dataset(data_cfg)
    targets, usable, behavior_ids = resolution_targets(dataset, data_cfg)
    params = {
        "stage": "probe_resolution_neurons",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "dataset_len": len(dataset),
        "targets_hash": stable_hash({"targets": targets.tolist(), "behavior_ids": behavior_ids.tolist()}),
        "data_config": asdict(data_cfg),
        "top_ratio": float(args.top_ratio),
        "min_score": float(args.min_score),
        "min_class_count": int(args.min_class_count),
        "epsilon": float(args.epsilon),
        "use_down_norm": bool(args.use_down_norm),
        "score_formula": "max_c(relu_z(weighted_separation_c)+0.5*relu_z(weighted_responsiveness_c))",
    }
    selected_path = dirs["neurons"] / "selected_neurons.jsonl"
    score_path = dirs["neurons"] / "neuron_scores.pt"
    layer_csv = dirs["neurons"] / "layer_summary.csv"
    if should_skip(dirs["neurons"], params, [selected_path, score_path, layer_csv], bool(args.overwrite)):
        return

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
    stats = ResolutionRunningStats()
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=resolution_collator(runtime.processor, data_cfg),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )
    with FFNActivationCollector(layers, save_dtype=torch.float32) as collector:
        for batch in tqdm(loader, desc="probe resolution", dynamic_ncols=True):
            batch = move_batch_to_device(batch, runtime.device, runtime.fp_dtype)
            batch_targets = batch.pop("aux_targets").detach().cpu().float()
            batch_usable = batch.pop("aux_usable_mask").detach().cpu().float()
            _ = batch.pop("aux_behavior_ids")
            token_mask = batch.pop("head_token_mask")
            collector.set_token_mask(token_mask)
            collector.clear()
            forward_inputs = build_forward_inputs(batch, runtime.model_family)
            autocast_enabled = runtime.device.type == "cuda" and runtime.fp_dtype in {torch.float16, torch.bfloat16}
            autocast_dtype = runtime.fp_dtype if runtime.fp_dtype is not None else torch.bfloat16
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                    _ = runtime.forward_model(**forward_inputs, use_cache=False, return_dict=True)
            stats.update(collector.captures, batch_targets, batch_usable)

    score_pack = stats.scores(
        down_norms=down_norms,
        use_down_norm=bool(args.use_down_norm),
        min_class_count=int(args.min_class_count),
        eps=float(args.epsilon),
    )
    rows, layer_rows = select_neurons(
        score_pack=score_pack,
        module_meta=module_meta,
        task="resolution",
        top_ratio=float(args.top_ratio),
        min_score=float(args.min_score),
        model_tag=tag,
    )
    write_jsonl(selected_path, rows)
    write_csv(layer_csv, layer_rows)
    torch.save(
        {
            "task": "resolution",
            "module_meta": module_meta,
            "scores": score_pack,
            "summary": summary_from_module_meta(module_meta),
        },
        score_path,
    )
    plot_layer_top_score_heatmap(
        score_pack=score_pack,
        module_meta=module_meta,
        out_path=dirs["viz"] / "layer_top_neuron_score_heatmap.png",
        title="Resolution: per-layer top neuron scores",
        top_n=int(args.heatmap_top_n),
    )
    plot_selected_density(layer_rows=layer_rows, out_path=dirs["viz"] / "selected_density_by_layer.png", title="Resolution selected neuron density")
    plot_resolution_classes(rows=rows, out_path=dirs["viz"] / "selected_class_by_layer.png", title="Resolution selected neuron class")
    total_info = summary_from_module_meta(module_meta)
    summary = {
        **total_info,
        "selected_neurons": len(rows),
        "selected_ratio": len(rows) / max(total_info["total_ffn_neurons"], 1),
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


if __name__ == "__main__":
    main()
