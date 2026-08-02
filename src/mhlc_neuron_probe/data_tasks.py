from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .baseline_bridge import load_capability_trainer_module, load_resolution_trainer_module
from .io_utils import resolve_path


CAPABILITY_DEFAULT_DATASET = (
    "../mhlc_data/data/train/Qwen3VL/"
    "Qwen3_VL_4B_Instruct_text_only_OriginalMixedShare_40851/verified"
)
RESOLUTION_DEFAULT_DATASET = "../mhlc_data/data/train/when2call/qwen3vl/Qwen3-VL-4B-Instruct_4class"


@dataclass
class CapabilityDataConfig:
    dataset_path: str = CAPABILITY_DEFAULT_DATASET
    aux_label_column: str = "correctness_score"
    subset_name: str | None = None
    seed: int = 42
    max_samples: int = 0
    max_seq_len: int = 32000
    head_input_mode: str = "completion_text_only"


@dataclass
class ResolutionDataConfig:
    dataset_path: str = RESOLUTION_DEFAULT_DATASET
    subset_name: str | None = None
    seed: int = 42
    max_samples: int = 0
    max_seq_len: int = 32000
    head_input_mode: str = "completion_text_only"
    max_head_input_tokens: int | None = None
    label_column: str = "behavior_class"
    label_name_column: str = "behavior"
    usable_column: str | None = "usable_behavior"
    drop_unusable_rows: bool = True
    class_names: list[str] = field(default_factory=lambda: ["tool_call", "request_for_info", "cannot_answer"])
    behavior_class_names: list[str] = field(
        default_factory=lambda: ["tool_call", "request_for_info", "cannot_answer", "direct_answer"]
    )


def _select_max(ds, max_samples: int):
    if max_samples and max_samples > 0:
        return ds.select(range(min(int(max_samples), len(ds))))
    return ds


def load_capability_dataset(cfg: CapabilityDataConfig):
    import datasets

    upstream = load_capability_trainer_module()
    ds = upstream.load_dataset_auto(str(resolve_path(cfg.dataset_path, Path(cfg.dataset_path))))
    train_ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    train_ds = upstream.filter_dataset_by_subset_name(train_ds, cfg.subset_name).shuffle(seed=int(cfg.seed))
    train_ds = _select_max(train_ds, int(cfg.max_samples))
    if len(train_ds) == 0:
        raise ValueError("Capability dataset is empty after filtering.")
    return train_ds


def load_resolution_dataset(cfg: ResolutionDataConfig):
    import datasets

    upstream = load_resolution_trainer_module()
    ds = upstream.load_dataset_auto_robust(str(resolve_path(cfg.dataset_path, Path(cfg.dataset_path))))
    train_ds = ds["train"] if isinstance(ds, datasets.DatasetDict) else ds
    train_ds = upstream.filter_dataset_by_subset_name(train_ds, cfg.subset_name).shuffle(seed=int(cfg.seed))
    if cfg.drop_unusable_rows:
        keep: list[int] = []
        for i, ex in enumerate(train_ds):
            usable, _targets, _name, behavior_id = upstream.derive_supervision(ex, cfg)
            if int(usable) > 0 and int(behavior_id) >= 0:
                keep.append(i)
        train_ds = train_ds.select(keep)
    train_ds = _select_max(train_ds, int(cfg.max_samples))
    if len(train_ds) == 0:
        raise ValueError("Resolution dataset is empty after filtering.")
    return train_ds


def capability_collator(processor: Any, cfg: CapabilityDataConfig):
    upstream = load_capability_trainer_module()
    return upstream.VlmCollator(processor, cfg.aux_label_column, int(cfg.max_seq_len), cfg.head_input_mode)


def resolution_collator(processor: Any, cfg: ResolutionDataConfig):
    upstream = load_resolution_trainer_module()
    return upstream.MultiLabelBehaviorCollator(processor, cfg)


def capability_labels(ds, label_column: str) -> torch.Tensor:
    return torch.tensor([float(x) for x in ds[label_column]], dtype=torch.float32).clamp(0.0, 1.0)


def resolution_targets(ds, cfg: ResolutionDataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    upstream = load_resolution_trainer_module()
    targets: list[list[float]] = []
    usable: list[float] = []
    behavior_ids: list[int] = []
    for ex in ds:
        u, t, _name, behavior_id = upstream.derive_supervision(ex, cfg)
        targets.append([float(x) for x in t])
        usable.append(float(u))
        behavior_ids.append(int(behavior_id))
    return (
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(usable, dtype=torch.float32),
        torch.tensor(behavior_ids, dtype=torch.long),
    )


def capability_group_thresholds(labels: torch.Tensor, high: float, low: float, fallback_ratio: float) -> dict[str, Any]:
    labels = labels.float().clamp(0.0, 1.0)
    high_mask = labels >= float(high)
    low_mask = labels < float(low)
    min_count = max(10, int(0.02 * labels.numel()))
    used_fallback = False
    if int(high_mask.sum()) < min_count or int(low_mask.sum()) < min_count:
        used_fallback = True
        ratio = min(max(float(fallback_ratio), 0.05), 0.45)
        sorted_labels, _ = torch.sort(labels)
        low_idx = max(0, min(labels.numel() - 1, int(labels.numel() * ratio) - 1))
        high_idx = max(0, min(labels.numel() - 1, int(labels.numel() * (1.0 - ratio))))
        low = float(sorted_labels[low_idx].item())
        high = float(sorted_labels[high_idx].item())
        low_mask = labels <= low
        high_mask = labels >= high
    return {
        "high_threshold": float(high),
        "low_threshold": float(low),
        "high_count": int(high_mask.sum().item()),
        "low_count": int(low_mask.sum().item()),
        "fallback_used": bool(used_fallback),
    }
