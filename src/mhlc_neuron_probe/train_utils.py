from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .feature_cache import ShardedFeatureDataset, compute_feature_norm
from .io_utils import read_json, write_json
from .stats import HEAD_CLASS_NAMES


class NeuronHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.10):
        super().__init__()
        mid = max(128, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), mid),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mid, max(64, mid // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(64, mid // 2), int(output_dim)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def lr_scale(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return float(min_ratio) + (1.0 - float(min_ratio)) * cosine


def latest_checkpoint(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return candidates[-1] if candidates else None


def _capability_weights(labels: torch.Tensor, failure_threshold: float, failure_weight: float, success_weight: float, severity_power: float) -> torch.Tensor:
    labels = labels.float().clamp(0.0, 1.0)
    is_failure = labels < float(failure_threshold)
    class_w = torch.where(
        is_failure,
        torch.full_like(labels, float(failure_weight)),
        torch.full_like(labels, float(success_weight)),
    )
    severity = 1.0 + torch.pow(1.0 - labels, float(severity_power))
    return class_w * severity


def _resolve_capability_class_weights(labels: torch.Tensor, failure_threshold: float, min_weight: float, max_weight: float) -> tuple[float, float]:
    fail = int((labels < float(failure_threshold)).sum().item())
    succ = int((labels >= float(failure_threshold)).sum().item())
    if fail <= 0 or succ <= 0:
        return 1.0, 1.0
    if fail < succ:
        fw, sw = succ / fail, 1.0
    else:
        fw, sw = 1.0, fail / succ
    fw = min(max(float(fw), float(min_weight)), float(max_weight))
    sw = min(max(float(sw), float(min_weight)), float(max_weight))
    return fw, sw


def _load_all_labels(manifest_path: Path, task: str) -> torch.Tensor:
    manifest = read_json(manifest_path)
    parts = []
    for shard in manifest["shards"]:
        payload = torch.load(shard["path"], map_location="cpu")
        parts.append(payload["labels" if task == "capability" else "targets"].float())
    return torch.cat(parts, dim=0)


def _resolution_pos_weight(manifest_path: Path, min_weight: float, max_weight: float) -> torch.Tensor:
    targets = _load_all_labels(manifest_path, "resolution")
    pos = targets.sum(dim=0)
    total = float(targets.shape[0])
    weights = []
    for value in pos.tolist():
        if value <= 0:
            w = float(max_weight)
        else:
            w = (total - value) / value
            w = min(max(float(w), float(min_weight)), float(max_weight))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def project_behavior_from_probs(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    pred = torch.argmax(probs, dim=-1)
    direct_id = len(HEAD_CLASS_NAMES)
    active = (probs >= float(threshold)).any(dim=-1)
    return pred.where(active, torch.full_like(pred, direct_id))


def _confusion_metrics(conf: torch.Tensor) -> dict[str, Any]:
    conf_f = conf.float()
    tp = torch.diag(conf_f)
    row = conf_f.sum(dim=1)
    col = conf_f.sum(dim=0)
    precision = tp / torch.clamp(col, min=1.0)
    recall = tp / torch.clamp(row, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1.0e-8)
    names = [*HEAD_CLASS_NAMES, "direct_answer"]
    out: dict[str, Any] = {
        "acc": float(tp.sum() / torch.clamp(conf_f.sum(), min=1.0)),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "confusion": conf.long().tolist(),
        "class_names": names,
    }
    for idx, name in enumerate(names):
        out[f"{name}_f1"] = float(f1[idx])
        out[f"{name}_recall"] = float(recall[idx])
        out[f"{name}_precision"] = float(precision[idx])
    return out


def train_neuron_head(args: Any, *, task: str, manifest_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest = read_json(manifest_path)
    input_dim = int(feature_manifest["summary"]["num_features"])
    output_dim = 1 if task == "capability" else 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ShardedFeatureDataset(manifest_path, task)
    norm = torch.load(compute_feature_norm(manifest_path), map_location="cpu")
    mean = norm["mean"].to(device)
    std = norm["std"].to(device).clamp_min(1.0e-6)

    loader = DataLoader(
        dataset,
        batch_size=int(args.train_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )
    head = NeuronHead(input_dim=input_dim, output_dim=output_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = math.ceil(len(loader) / int(args.grad_accum_steps)) * int(args.num_epochs)
    warmup_steps = int(float(args.warmup_ratio) * total_steps)

    micro_seen = 0
    opt_step = 0
    ckpt = latest_checkpoint(out_dir) if bool(args.resume) else None
    if ckpt is not None:
        payload = torch.load(ckpt, map_location=device)
        head.load_state_dict(payload["head_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        micro_seen = int(payload.get("micro_seen", 0))
        opt_step = int(payload.get("opt_step", 0))
        print(f"[resume] {ckpt} opt_step={opt_step} micro_seen={micro_seen}", flush=True)

    if task == "capability":
        labels_all = _load_all_labels(manifest_path, "capability")
        failure_weight, success_weight = _resolve_capability_class_weights(
            labels_all,
            float(args.failure_threshold),
            float(args.min_class_weight),
            float(args.max_class_weight),
        )
        pos_weight = None
    else:
        failure_weight = success_weight = 1.0
        pos_weight = _resolution_pos_weight(manifest_path, float(args.min_class_weight), float(args.max_class_weight)).to(device)

    config_used = {
        **vars(args),
        "task": task,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "feature_manifest": str(manifest_path),
        "resolved_failure_weight": failure_weight,
        "resolved_success_weight": success_weight,
        "resolution_pos_weight": pos_weight.detach().cpu().tolist() if pos_weight is not None else None,
    }
    write_json(out_dir / "config_used.json", config_used)

    pbar = tqdm(total=total_steps, initial=opt_step, desc=f"train {task}", dynamic_ncols=True)
    loss_sum = 0.0
    loss_n = 0
    abs_err = 0.0
    sq_err = 0.0
    reg_n = 0
    confusion = torch.zeros(4, 4, dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)

    current_micro = 0
    for _epoch in range(int(args.num_epochs)):
        accum = 0
        for batch in loader:
            if current_micro < micro_seen:
                current_micro += 1
                continue
            current_micro += 1
            features = ((batch["features"].to(device, non_blocking=True) - mean) / std).float()
            logits = head(features)
            if task == "capability":
                labels = batch["labels"].to(device, non_blocking=True).float().clamp(0.0, 1.0)
                pred = torch.sigmoid(logits.squeeze(-1))
                weights = _capability_weights(
                    labels,
                    float(args.failure_threshold),
                    failure_weight,
                    success_weight,
                    float(args.severity_power),
                )
                loss = (weights * torch.square(pred - labels)).sum() / weights.sum().clamp_min(1.0e-8)
                pred_cpu = pred.detach().cpu()
                labels_cpu = labels.detach().cpu()
                abs_err += torch.abs(pred_cpu - labels_cpu).sum().item()
                sq_err += torch.square(pred_cpu - labels_cpu).sum().item()
                reg_n += int(pred_cpu.numel())
            else:
                targets = batch["targets"].to(device, non_blocking=True).float()
                usable = batch["usable_mask"].to(device, non_blocking=True).float()
                per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
                per_item = per_elem.mean(dim=-1)
                loss = (per_item * usable).sum() / usable.sum().clamp_min(1.0e-8)
                probs = torch.sigmoid(logits.detach()).cpu()
                preds = project_behavior_from_probs(probs, float(args.decision_threshold))
                behavior_ids = batch["behavior_ids"].cpu().long()
                usable_cpu = batch["usable_mask"].cpu().long()
                for y, yhat, u in zip(behavior_ids.tolist(), preds.tolist(), usable_cpu.tolist()):
                    if int(u) > 0 and int(y) >= 0:
                        confusion[int(y), int(yhat)] += 1

            (loss / int(args.grad_accum_steps)).backward()
            loss_sum += float(loss.detach().cpu().item())
            loss_n += 1
            accum += 1
            if accum < int(args.grad_accum_steps):
                continue
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), float(args.max_grad_norm))
            opt_step += 1
            scale = lr_scale(opt_step, total_steps, warmup_steps, float(args.min_lr_ratio))
            for group in optimizer.param_groups:
                group["lr"] = float(args.lr) * scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            pbar.update(1)
            if int(args.log_every) > 0 and opt_step % int(args.log_every) == 0:
                logs: dict[str, float] = {"loss": loss_sum / max(loss_n, 1), "lr": optimizer.param_groups[0]["lr"]}
                if task == "capability":
                    logs["mae"] = abs_err / max(reg_n, 1)
                    logs["rmse"] = math.sqrt(sq_err / max(reg_n, 1))
                else:
                    logs["macro_f1"] = float(_confusion_metrics(confusion)["macro_f1"])
                print(json.dumps({"step": opt_step, **logs}, ensure_ascii=False), flush=True)
                loss_sum = loss_n = abs_err = sq_err = reg_n = 0
            if int(args.save_every) > 0 and opt_step % int(args.save_every) == 0:
                torch.save(
                    {
                        "head_state": head.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "opt_step": opt_step,
                        "micro_seen": current_micro,
                        "config": config_used,
                    },
                    out_dir / f"checkpoint_step_{opt_step}.pt",
                )
        if accum > 0:
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), float(args.max_grad_norm))
            opt_step += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
    pbar.close()

    final_path = out_dir / "neuron_head_final.pt"
    torch.save(
        {
            "head_state": head.state_dict(),
            "opt_step": opt_step,
            "config": config_used,
            "feature_norm": {"mean": mean.detach().cpu(), "std": std.detach().cpu()},
        },
        final_path,
    )
    metrics: dict[str, Any] = {"steps": opt_step}
    if task == "resolution":
        metrics.update(_confusion_metrics(confusion))
    write_json(out_dir / "final_metrics.json", metrics)
    print(f"[write] {final_path}", flush=True)
    return final_path

