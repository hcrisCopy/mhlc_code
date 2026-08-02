from __future__ import annotations

import gc
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .baseline_bridge import build_forward_inputs, load_capability_trainer_module, load_frozen_backbone, move_batch_to_device
from .feature_cache import build_feature_plan, selected_features_from_captures
from .ffn_hooks import FFNActivationCollector, discover_ffn_intermediate_layers, module_meta_from_layers
from .io_utils import read_jsonl, rel
from .train_utils import NeuronHead


@dataclass
class NeuronHeadScoreBatch:
    logits: torch.Tensor
    probs: torch.Tensor
    features: torch.Tensor


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def load_jsonl_by_sample_idx(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        if "sample_idx" not in row:
            continue
        rows[int(row["sample_idx"])] = row
    return rows


def ordered_rows_from_index(rows_by_idx: dict[int, dict[str, Any]], examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ex in examples:
        sample_idx = int(ex.get("sample_idx", ex.get("dataset_index", len(out))))
        if sample_idx in rows_by_idx:
            out.append(rows_by_idx[sample_idx])
    return out


class NeuronHeadScorer:
    def __init__(
        self,
        *,
        task: str,
        model_path: str,
        head_checkpoint_path: Path,
        neuron_path: Path,
        model_family: str,
        thinking_mode: str,
        trust_remote_code: bool,
        attn_implementation: str,
        prefer_unsloth_mirror: bool,
        load_in_4bit: bool,
        load_in_8bit: bool,
        use_gradient_checkpointing: str,
        dtype: str,
        max_seq_len: int,
        max_pixels: int,
        head_input_mode: str,
        max_head_input_tokens: int | None = None,
    ):
        self.task = str(task)
        if self.task not in {"capability", "resolution"}:
            raise ValueError(f"Unsupported task: {task}")

        self.runtime = load_frozen_backbone(
            model_name_or_path=model_path,
            model_family=model_family,
            thinking_mode=thinking_mode,
            trust_remote_code=bool(trust_remote_code),
            attn_implementation=attn_implementation,
            prefer_unsloth_mirror=bool(prefer_unsloth_mirror),
            load_in_4bit=bool(load_in_4bit),
            load_in_8bit=bool(load_in_8bit),
            use_gradient_checkpointing=use_gradient_checkpointing,
            dtype=dtype,
            max_seq_len=int(max_seq_len),
            max_pixels=int(max_pixels),
        )
        self.layers = discover_ffn_intermediate_layers(self.runtime.forward_model)
        self.module_meta = module_meta_from_layers(self.layers)
        self.plan = build_feature_plan(read_jsonl(neuron_path))
        self.collector = FFNActivationCollector(self.layers, save_dtype=torch.float32)
        self.collector.__enter__()

        upstream = load_capability_trainer_module()
        builder_cls = getattr(upstream, "ChatBatchBuilder", None)
        if builder_cls is None:
            builder_cls = importlib.import_module("aux_head_shared_utils").ChatBatchBuilder
        self.chat_builder = builder_cls(
            self.runtime.processor,
            int(max_seq_len),
            str(head_input_mode),
        )
        self.max_head_input_tokens = max_head_input_tokens

        ckpt = torch.load(head_checkpoint_path, map_location=self.runtime.device)
        ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        output_dim = 1 if self.task == "capability" else 3
        self.head = NeuronHead(
            input_dim=int(self.plan.num_features),
            output_dim=output_dim,
            hidden_dim=int(ckpt_config.get("hidden_dim", 512)),
            dropout=float(ckpt_config.get("dropout", 0.10)),
        ).to(self.runtime.device)
        self.head.load_state_dict(ckpt["head_state"] if isinstance(ckpt, dict) and "head_state" in ckpt else ckpt)
        self.head.eval()

        norm = ckpt.get("feature_norm") if isinstance(ckpt, dict) else None
        if norm is None:
            raise ValueError(
                f"Missing feature_norm in {rel(head_checkpoint_path)}. "
                "Use the neuron head checkpoint produced by src/08 or src/10."
            )
        self.mean = norm["mean"].to(self.runtime.device)
        self.std = norm["std"].to(self.runtime.device).clamp_min(1.0e-6)

    def close(self) -> None:
        if self.collector is not None:
            self.collector.__exit__(None, None, None)
            self.collector = None
        self.head = None
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.model = None
            runtime.forward_model = None
            runtime.processor = None
            self.runtime = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "NeuronHeadScorer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @torch.no_grad()
    def score_messages(
        self,
        messages_batch: Sequence[list[dict[str, Any]]],
        *,
        images: Sequence[Any | None] | None = None,
    ) -> NeuronHeadScoreBatch:
        if images is None:
            images = [None] * len(messages_batch)
        batch = self.chat_builder.build_from_messages(list(messages_batch), list(images), labels=None)
        batch = move_batch_to_device(batch, self.runtime.device, self.runtime.fp_dtype)
        token_mask = batch.pop("head_token_mask")
        if self.max_head_input_tokens is not None:
            max_tokens = int(self.max_head_input_tokens)
            if max_tokens <= 0:
                raise ValueError(f"max_head_input_tokens must be positive, got {max_tokens}.")
            active = token_mask > 0
            active_rank = active.to(torch.long).cumsum(dim=-1)
            token_mask = (active & (active_rank <= max_tokens)).to(dtype=token_mask.dtype)
        self.collector.set_token_mask(token_mask)
        self.collector.clear()

        forward_inputs = build_forward_inputs(batch, self.runtime.model_family)
        autocast_enabled = self.runtime.device.type == "cuda" and self.runtime.fp_dtype in {torch.float16, torch.bfloat16}
        autocast_dtype = self.runtime.fp_dtype if self.runtime.fp_dtype is not None else torch.bfloat16
        autocast_device = "cuda" if self.runtime.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=autocast_enabled):
            _ = self.runtime.forward_model(**forward_inputs, use_cache=False, return_dict=True)

        missing = [meta["key"] for meta in self.module_meta if meta["key"] not in self.collector.captures]
        if missing:
            raise RuntimeError(f"Missing FFN captures: {missing[:3]}")

        features = selected_features_from_captures(self.collector.captures, self.plan).to(self.runtime.device)
        features = ((features - self.mean) / self.std).float()
        logits = self.head(features)
        probs = torch.sigmoid(logits)
        return NeuronHeadScoreBatch(
            logits=logits.detach().cpu(),
            probs=probs.detach().cpu(),
            features=features.detach().cpu(),
        )
