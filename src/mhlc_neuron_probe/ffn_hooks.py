from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


ACTIVATION_DEFINITION = "completion_text_only_mean_ffn_intermediate_before_down_proj"


@dataclass(frozen=True)
class FFNIntermediateLayer:
    layer: int
    mlp_name: str
    gate_name: str
    up_name: str
    down_name: str
    down: nn.Module
    dim: int

    @property
    def key(self) -> str:
        return self.down_name


def discover_ffn_intermediate_layers(model: nn.Module) -> list[FFNIntermediateLayer]:
    named = dict(model.named_modules())
    layers: list[FFNIntermediateLayer] = []
    for name, module in named.items():
        if not name.endswith(".mlp.down_proj"):
            continue
        parts = name.split(".")
        try:
            layer_index = int(parts[-3])
        except (ValueError, IndexError):
            continue
        mlp_name = ".".join(parts[:-1])
        gate_name = f"{mlp_name}.gate_proj"
        up_name = f"{mlp_name}.up_proj"
        if gate_name not in named or up_name not in named:
            continue
        if not hasattr(module, "in_features"):
            continue
        dim = int(module.in_features)
        layers.append(
            FFNIntermediateLayer(
                layer=layer_index,
                mlp_name=mlp_name,
                gate_name=gate_name,
                up_name=up_name,
                down_name=name,
                down=module,
                dim=dim,
            )
        )
    layers.sort(key=lambda item: item.layer)
    if not layers:
        raise RuntimeError("No FFN intermediate layers found. Expected modules ending with .mlp.down_proj.")
    return layers


def module_meta_from_layers(layers: list[FFNIntermediateLayer]) -> list[dict[str, Any]]:
    return [
        {
            "layer": int(layer.layer),
            "module": "ffn_intermediate",
            "key": layer.key,
            "mlp_name": layer.mlp_name,
            "down_name": layer.down_name,
            "dim": int(layer.dim),
        }
        for layer in layers
    ]


def down_weight_norms(layers: list[FFNIntermediateLayer]) -> dict[str, torch.Tensor]:
    norms: dict[str, torch.Tensor] = {}
    for layer in layers:
        weight = getattr(layer.down, "weight", None)
        if torch.is_tensor(weight):
            norms[layer.key] = weight.detach().float().norm(dim=0).cpu()
    return norms


class FFNActivationCollector:
    def __init__(self, layers: list[FFNIntermediateLayer], save_dtype: torch.dtype = torch.float32):
        self.layers = layers
        self.save_dtype = save_dtype
        self.token_mask: torch.Tensor | None = None
        self.captures: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def set_token_mask(self, token_mask: torch.Tensor) -> None:
        self.token_mask = token_mask.detach()

    def clear(self) -> None:
        self.captures = {}

    def _aligned_mask(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.token_mask is None:
            raise RuntimeError("token_mask must be set before the model forward pass.")
        mask = self.token_mask.to(device=hidden.device, dtype=torch.bool)
        seq_len = hidden.shape[1]
        if mask.shape[1] > seq_len:
            mask = mask[:, -seq_len:]
        elif mask.shape[1] < seq_len:
            pad = torch.zeros(mask.shape[0], seq_len - mask.shape[1], device=mask.device, dtype=torch.bool)
            mask = torch.cat([pad, mask], dim=1)
        return mask

    def _pool_completion_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        mask = self._aligned_mask(hidden)
        counts = mask.sum(dim=1).clamp_min(1).to(hidden.dtype)
        pooled = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / counts.unsqueeze(-1)
        return pooled.detach().to(device="cpu", dtype=self.save_dtype)

    def _make_hook(self, key: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            hidden = inputs[0]
            if not torch.is_tensor(hidden) or hidden.dim() != 3:
                raise RuntimeError(f"Unexpected FFN intermediate tensor for {key}: {type(hidden)}")
            self.captures[key] = self._pool_completion_tokens(hidden)

        return hook

    def __enter__(self):
        self.handles = [layer.down.register_forward_pre_hook(self._make_hook(layer.key)) for layer in self.layers]
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

