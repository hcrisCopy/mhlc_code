from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


HEAD_CLASS_NAMES = ["tool_call", "request_for_info", "cannot_answer"]


def _zeros(dim: int, *prefix: int) -> torch.Tensor:
    return torch.zeros((*prefix, int(dim)), dtype=torch.float64)


def _zscore(values: torch.Tensor, eps: float) -> torch.Tensor:
    values = values.float()
    std = values.std(unbiased=False).clamp_min(float(eps))
    out = (values - values.mean()) / std
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _relu_z(values: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.relu(_zscore(values, eps))


def _norm_factor(down_norms: dict[str, torch.Tensor] | None, key: str, dim: int, eps: float) -> torch.Tensor:
    if not down_norms or key not in down_norms:
        return torch.ones(dim, dtype=torch.float32)
    norm = down_norms[key].float()
    if norm.numel() != dim:
        return torch.ones(dim, dtype=torch.float32)
    return torch.nan_to_num(norm / norm.mean().clamp_min(float(eps)), nan=1.0, posinf=1.0, neginf=1.0)


@dataclass
class CapabilityLayerStats:
    dim: int
    n: int = 0
    high_n: int = 0
    low_n: int = 0
    sum_x: torch.Tensor | None = None
    sum_x2: torch.Tensor | None = None
    sum_xy: torch.Tensor | None = None
    high_sum: torch.Tensor | None = None
    high_sum2: torch.Tensor | None = None
    low_sum: torch.Tensor | None = None
    low_sum2: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.sum_x = _zeros(self.dim)
        self.sum_x2 = _zeros(self.dim)
        self.sum_xy = _zeros(self.dim)
        self.high_sum = _zeros(self.dim)
        self.high_sum2 = _zeros(self.dim)
        self.low_sum = _zeros(self.dim)
        self.low_sum2 = _zeros(self.dim)


class CapabilityRunningStats:
    def __init__(self, *, high_threshold: float, low_threshold: float):
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.layers: dict[str, CapabilityLayerStats] = {}
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0

    def _ensure(self, key: str, dim: int) -> CapabilityLayerStats:
        if key not in self.layers:
            self.layers[key] = CapabilityLayerStats(dim=int(dim))
        return self.layers[key]

    def update(self, captures: dict[str, torch.Tensor], labels: torch.Tensor) -> None:
        labels = labels.detach().cpu().float().clamp(0.0, 1.0)
        high_mask = labels >= self.high_threshold
        low_mask = labels < self.low_threshold
        self.n += int(labels.numel())
        self.sum_y += float(labels.sum().item())
        self.sum_y2 += float(torch.square(labels).sum().item())
        labels64 = labels.to(torch.float64)
        for key, x in captures.items():
            x64 = x.detach().cpu().to(torch.float64)
            stat = self._ensure(key, x64.shape[1])
            stat.n += int(x64.shape[0])
            stat.sum_x += x64.sum(dim=0)
            stat.sum_x2 += torch.square(x64).sum(dim=0)
            stat.sum_xy += (x64 * labels64.unsqueeze(1)).sum(dim=0)
            if bool(high_mask.any()):
                high = x64[high_mask]
                stat.high_n += int(high.shape[0])
                stat.high_sum += high.sum(dim=0)
                stat.high_sum2 += torch.square(high).sum(dim=0)
            if bool(low_mask.any()):
                low = x64[low_mask]
                stat.low_n += int(low.shape[0])
                stat.low_sum += low.sum(dim=0)
                stat.low_sum2 += torch.square(low).sum(dim=0)

    def scores(self, *, down_norms: dict[str, torch.Tensor] | None, use_down_norm: bool, eps: float) -> dict[str, dict[str, torch.Tensor]]:
        out: dict[str, dict[str, torch.Tensor]] = {}
        mean_y = self.sum_y / max(self.n, 1)
        var_y = max(self.sum_y2 / max(self.n, 1) - mean_y * mean_y, float(eps))
        for key, stat in self.layers.items():
            n = max(stat.n, 1)
            mean = stat.sum_x / n
            var = (stat.sum_x2 / n - mean.square()).clamp_min(float(eps))
            cov = stat.sum_xy / n - mean * mean_y
            corr = cov / torch.sqrt(var * var_y)
            corr_abs = torch.abs(torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))

            high_n = max(stat.high_n, 1)
            low_n = max(stat.low_n, 1)
            high_mean = stat.high_sum / high_n
            low_mean = stat.low_sum / low_n
            high_var = (stat.high_sum2 / high_n - high_mean.square()).clamp_min(float(eps))
            low_var = (stat.low_sum2 / low_n - low_mean.square()).clamp_min(float(eps))
            delta = high_mean - low_mean
            pooled = torch.sqrt((high_var + low_var) * 0.5).clamp_min(float(eps))
            separation = torch.abs(delta) / pooled
            responsiveness = torch.abs(delta)
            norm = _norm_factor(down_norms if use_down_norm else None, key, stat.dim, eps).to(torch.float64)
            weighted_separation = separation * norm
            weighted_responsiveness = responsiveness * norm
            score = _relu_z(weighted_separation, eps) + _relu_z(corr_abs, eps) + 0.5 * _relu_z(weighted_responsiveness, eps)
            out[key] = {
                "score": score.float(),
                "separation": separation.float(),
                "weighted_separation": weighted_separation.float(),
                "correlation": corr.float(),
                "correlation_abs": corr_abs.float(),
                "responsiveness": responsiveness.float(),
                "weighted_responsiveness": weighted_responsiveness.float(),
                "delta": delta.float(),
                "high_mean": high_mean.float(),
                "low_mean": low_mean.float(),
                "direction_sign": torch.sign(delta).float(),
            }
        return out


@dataclass
class ResolutionLayerStats:
    dim: int
    n: int = 0
    pos_n: torch.Tensor | None = None
    sum_x: torch.Tensor | None = None
    sum_x2: torch.Tensor | None = None
    pos_sum: torch.Tensor | None = None
    pos_sum2: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.pos_n = torch.zeros(3, dtype=torch.float64)
        self.sum_x = _zeros(self.dim)
        self.sum_x2 = _zeros(self.dim)
        self.pos_sum = _zeros(self.dim, 3)
        self.pos_sum2 = _zeros(self.dim, 3)


class ResolutionRunningStats:
    def __init__(self):
        self.layers: dict[str, ResolutionLayerStats] = {}
        self.n = 0

    def _ensure(self, key: str, dim: int) -> ResolutionLayerStats:
        if key not in self.layers:
            self.layers[key] = ResolutionLayerStats(dim=int(dim))
        return self.layers[key]

    def update(self, captures: dict[str, torch.Tensor], targets: torch.Tensor, usable_mask: torch.Tensor) -> None:
        targets = targets.detach().cpu().float().clamp(0.0, 1.0)
        usable = usable_mask.detach().cpu().float() > 0
        if not bool(usable.any()):
            return
        targets = targets[usable]
        self.n += int(targets.shape[0])
        for key, x in captures.items():
            x64 = x.detach().cpu()[usable].to(torch.float64)
            stat = self._ensure(key, x64.shape[1])
            stat.n += int(x64.shape[0])
            stat.sum_x += x64.sum(dim=0)
            stat.sum_x2 += torch.square(x64).sum(dim=0)
            for class_idx in range(3):
                mask = targets[:, class_idx] > 0.5
                if not bool(mask.any()):
                    continue
                pos = x64[mask]
                stat.pos_n[class_idx] += float(pos.shape[0])
                stat.pos_sum[class_idx] += pos.sum(dim=0)
                stat.pos_sum2[class_idx] += torch.square(pos).sum(dim=0)

    def scores(
        self,
        *,
        down_norms: dict[str, torch.Tensor] | None,
        use_down_norm: bool,
        min_class_count: int,
        eps: float,
    ) -> dict[str, dict[str, torch.Tensor]]:
        out: dict[str, dict[str, torch.Tensor]] = {}
        for key, stat in self.layers.items():
            total_n = max(stat.n, 1)
            total_sum = stat.sum_x
            total_sum2 = stat.sum_x2
            class_scores: list[torch.Tensor] = []
            class_delta: list[torch.Tensor] = []
            class_sep: list[torch.Tensor] = []
            for class_idx in range(3):
                pos_n = int(stat.pos_n[class_idx].item())
                neg_n = int(stat.n - pos_n)
                if pos_n < int(min_class_count) or neg_n < int(min_class_count):
                    zeros = torch.zeros(stat.dim, dtype=torch.float32)
                    class_scores.append(zeros)
                    class_delta.append(zeros)
                    class_sep.append(zeros)
                    continue
                pos_sum = stat.pos_sum[class_idx]
                pos_sum2 = stat.pos_sum2[class_idx]
                neg_sum = total_sum - pos_sum
                neg_sum2 = total_sum2 - pos_sum2
                pos_mean = pos_sum / pos_n
                neg_mean = neg_sum / neg_n
                pos_var = (pos_sum2 / pos_n - pos_mean.square()).clamp_min(float(eps))
                neg_var = (neg_sum2 / neg_n - neg_mean.square()).clamp_min(float(eps))
                delta = pos_mean - neg_mean
                pooled = torch.sqrt((pos_var + neg_var) * 0.5).clamp_min(float(eps))
                separation = torch.abs(delta) / pooled
                responsiveness = torch.abs(delta)
                norm = _norm_factor(down_norms if use_down_norm else None, key, stat.dim, eps).to(torch.float64)
                weighted_separation = separation * norm
                weighted_responsiveness = responsiveness * norm
                score = _relu_z(weighted_separation, eps) + 0.5 * _relu_z(weighted_responsiveness, eps)
                class_scores.append(score.float())
                class_delta.append(delta.float())
                class_sep.append(separation.float())
            stacked_scores = torch.stack(class_scores, dim=0)
            best_score, best_class = torch.max(stacked_scores, dim=0)
            out[key] = {
                "score": best_score.float(),
                "best_class_id": best_class.long(),
                "score_tool_call": class_scores[0].float(),
                "score_request_for_info": class_scores[1].float(),
                "score_cannot_answer": class_scores[2].float(),
                "delta_tool_call": class_delta[0].float(),
                "delta_request_for_info": class_delta[1].float(),
                "delta_cannot_answer": class_delta[2].float(),
                "separation_tool_call": class_sep[0].float(),
                "separation_request_for_info": class_sep[1].float(),
                "separation_cannot_answer": class_sep[2].float(),
                "direction_sign": torch.sign(torch.gather(torch.stack(class_delta, dim=0), 0, best_class.unsqueeze(0)).squeeze(0)).float(),
            }
        return out


def summary_from_module_meta(module_meta: list[dict[str, Any]]) -> dict[str, int]:
    total = sum(int(meta["dim"]) for meta in module_meta)
    return {"layers": len(module_meta), "total_ffn_neurons": int(total)}


def capability_stats_state(stats: CapabilityRunningStats) -> dict[str, Any]:
    """CPU-only, torch-save-safe representation for eight-GPU aggregation."""
    return {
        "high_threshold": stats.high_threshold,
        "low_threshold": stats.low_threshold,
        "n": stats.n,
        "sum_y": stats.sum_y,
        "sum_y2": stats.sum_y2,
        "layers": {
            key: {
                "dim": layer.dim,
                "n": layer.n,
                "high_n": layer.high_n,
                "low_n": layer.low_n,
                "sum_x": layer.sum_x,
                "sum_x2": layer.sum_x2,
                "sum_xy": layer.sum_xy,
                "high_sum": layer.high_sum,
                "high_sum2": layer.high_sum2,
                "low_sum": layer.low_sum,
                "low_sum2": layer.low_sum2,
            }
            for key, layer in stats.layers.items()
        },
    }


def merge_capability_stats(states: list[dict[str, Any]]) -> CapabilityRunningStats:
    if not states:
        raise ValueError("No Capability statistics were provided for merge.")
    merged = CapabilityRunningStats(
        high_threshold=float(states[0]["high_threshold"]),
        low_threshold=float(states[0]["low_threshold"]),
    )
    for state in states:
        merged.n += int(state["n"])
        merged.sum_y += float(state["sum_y"])
        merged.sum_y2 += float(state["sum_y2"])
        for key, values in state["layers"].items():
            layer = merged._ensure(key, int(values["dim"]))
            layer.n += int(values["n"])
            layer.high_n += int(values["high_n"])
            layer.low_n += int(values["low_n"])
            for name in ("sum_x", "sum_x2", "sum_xy", "high_sum", "high_sum2", "low_sum", "low_sum2"):
                setattr(layer, name, getattr(layer, name) + values[name].to(torch.float64))
    return merged


def resolution_stats_state(stats: ResolutionRunningStats) -> dict[str, Any]:
    return {
        "n": stats.n,
        "layers": {
            key: {
                "dim": layer.dim,
                "n": layer.n,
                "pos_n": layer.pos_n,
                "sum_x": layer.sum_x,
                "sum_x2": layer.sum_x2,
                "pos_sum": layer.pos_sum,
                "pos_sum2": layer.pos_sum2,
            }
            for key, layer in stats.layers.items()
        },
    }


def merge_resolution_stats(states: list[dict[str, Any]]) -> ResolutionRunningStats:
    if not states:
        raise ValueError("No Resolution statistics were provided for merge.")
    merged = ResolutionRunningStats()
    for state in states:
        merged.n += int(state["n"])
        for key, values in state["layers"].items():
            layer = merged._ensure(key, int(values["dim"]))
            layer.n += int(values["n"])
            for name in ("pos_n", "sum_x", "sum_x2", "pos_sum", "pos_sum2"):
                setattr(layer, name, getattr(layer, name) + values[name].to(torch.float64))
    return merged
