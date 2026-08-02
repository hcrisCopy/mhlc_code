from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mhlc_data_prep.paths import upstream_repo_root


def _load_upstream_script(relative_path: str, module_name: str):
    repo = upstream_repo_root()
    script = repo / relative_path
    for path in [repo, script.parent]:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream script: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_capability_trainer_module():
    return _load_upstream_script(
        "train_head_standalone_unsloth_regression_weighted_multimodel.py",
        "mhlc_upstream_capability_trainer",
    )


def load_resolution_trainer_module():
    return _load_upstream_script(
        "when2call/train_when2call_head_4class_3sigmoid.py",
        "mhlc_upstream_resolution_trainer",
    )


def dtype_from_str(name: str) -> torch.dtype | None:
    module = load_capability_trainer_module()
    return module.dtype_from_str(name)


@dataclass
class BackboneRuntime:
    model: Any
    processor: Any
    forward_model: Any
    model_id_for_load: str
    model_family: str
    thinking_enabled: bool
    attn_implementation: str
    fp_dtype: torch.dtype | None
    device: torch.device


def load_frozen_backbone(
    *,
    model_name_or_path: str,
    model_family: str = "auto",
    thinking_mode: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "sdpa",
    prefer_unsloth_mirror: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_gradient_checkpointing: str = "unsloth",
    dtype: str = "bf16",
    max_seq_len: int = 32000,
    max_pixels: int = 200000,
) -> BackboneRuntime:
    module = load_capability_trainer_module()
    device = module.get_device()
    fp_dtype = module.dtype_from_str(dtype)

    requested_model_id = str(model_name_or_path)
    resolved_model_family = module._infer_model_family(requested_model_id, model_family)
    thinking_enabled = module._resolve_thinking_enabled(
        requested_model_id,
        resolved_model_family,
        thinking_mode,
    )
    model_id = module._resolve_model_id_for_load(
        requested_model_id=requested_model_id,
        resolved_model_family=resolved_model_family,
        prefer_unsloth_mirror=prefer_unsloth_mirror,
    )
    actual_attn = module._resolve_attn_implementation(attn_implementation, resolved_model_family)

    model, processor = module.FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=int(max_seq_len),
        load_in_4bit=bool(load_in_4bit),
        load_in_8bit=bool(load_in_8bit),
        use_gradient_checkpointing=use_gradient_checkpointing,
        trust_remote_code=bool(trust_remote_code),
        attn_implementation=actual_attn,
    )
    model = model.to(device)
    module.FastVisionModel.for_inference(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    module._safe_set_max_pixels(processor, int(max_pixels))
    processor, _ = module._patch_processor_for_runtime_prompting(
        processor=processor,
        model_family=resolved_model_family,
        thinking_enabled=thinking_enabled,
    )
    forward_model = getattr(model, "model", None) or model

    return BackboneRuntime(
        model=model,
        processor=processor,
        forward_model=forward_model,
        model_id_for_load=str(model_id),
        model_family=str(resolved_model_family),
        thinking_enabled=bool(thinking_enabled),
        attn_implementation=str(actual_attn),
        fp_dtype=fp_dtype,
        device=device,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device, fp_dtype: torch.dtype | None) -> dict[str, Any]:
    module = load_capability_trainer_module()
    return module.move_batch_to_device(batch, device, fp_dtype)


def forward_input_keys(model_family: str) -> tuple[str, ...]:
    if model_family == "gemma4":
        return (
            "input_ids",
            "attention_mask",
            "position_ids",
            "cache_position",
            "pixel_values",
            "image_position_ids",
            "pixel_attention_mask",
            "image_attention_mask",
            "image_sizes",
        )
    return (
        "input_ids",
        "attention_mask",
        "position_ids",
        "cache_position",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "mm_token_type_ids",
    )


def build_forward_inputs(batch: dict[str, Any], model_family: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in forward_input_keys(model_family):
        if key not in batch:
            continue
        value = batch[key]
        if value is None:
            continue
        if torch.is_tensor(value) and value.numel() == 0:
            continue
        out[key] = value
    return out
