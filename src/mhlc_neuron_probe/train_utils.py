from __future__ import annotations

import json
import math
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


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) != 0.0 else 0.0


def _binary_confusion(labels: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    conf = torch.zeros(2, 2, dtype=torch.long)
    labels = labels.detach().cpu().long().view(-1)
    preds = preds.detach().cpu().long().view(-1)
    for y, yhat in zip(labels.tolist(), preds.tolist()):
        if int(y) in (0, 1) and int(yhat) in (0, 1):
            conf[int(y), int(yhat)] += 1
    return conf


def _binary_metrics_from_confusion(conf: torch.Tensor) -> dict[str, Any]:
    tn = int(conf[0, 0].item())
    fp = int(conf[0, 1].item())
    fn = int(conf[1, 0].item())
    tp = int(conf[1, 1].item())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    neg_precision = _safe_div(tn, tn + fn)
    specificity = _safe_div(tn, tn + fp)
    neg_f1 = _safe_div(2 * tn, 2 * tn + fn + fp)
    return {
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_precision": neg_precision,
        "negative_recall": specificity,
        "negative_f1": neg_f1,
        "macro_precision": 0.5 * (precision + neg_precision),
        "macro_recall": 0.5 * (recall + specificity),
        "macro_f1": 0.5 * (f1 + neg_f1),
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _failure_metrics_from_confusion(conf: torch.Tensor) -> dict[str, Any]:
    metrics = _binary_metrics_from_confusion(conf)
    return {
        "failure_precision": metrics["precision"],
        "failure_recall": metrics["recall"],
        "failure_f1": metrics["f1"],
        "specificity": metrics["specificity"],
        "balanced_acc": metrics["balanced_accuracy"],
        "failure_confusion": conf.long().tolist(),
        "failure_tp": metrics["tp"],
        "failure_tn": metrics["tn"],
        "failure_fp": metrics["fp"],
        "failure_fn": metrics["fn"],
    }


def _roc_auc_binary(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    labels = labels.detach().cpu().long().view(-1)
    scores = scores.detach().cpu().float().view(-1)
    pos = int((labels == 1).sum().item())
    neg = int((labels == 0).sum().item())
    if pos == 0 or neg == 0 or labels.numel() == 0:
        return None
    sorted_scores, order = torch.sort(scores, descending=False)
    ranks = torch.empty_like(sorted_scores, dtype=torch.float64)
    i = 0
    rank = 1.0
    n = int(sorted_scores.numel())
    while i < n:
        j = i
        while j + 1 < n and float(sorted_scores[j + 1].item()) == float(sorted_scores[i].item()):
            j += 1
        ranks[i : j + 1] = 0.5 * (rank + rank + (j - i))
        rank += float(j - i + 1)
        i = j + 1
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(original_ranks[labels == 1].sum().item())
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def _average_precision_binary(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    labels = labels.detach().cpu().long().view(-1)
    scores = scores.detach().cpu().float().view(-1)
    n_pos = int((labels == 1).sum().item())
    if n_pos == 0 or labels.numel() == 0:
        return None
    _, order = torch.sort(scores, descending=True)
    sorted_labels = labels[order]
    tp = 0
    ap = 0.0
    for rank, y in enumerate(sorted_labels.tolist(), start=1):
        if int(y) == 1:
            tp += 1
            ap += tp / float(rank)
    return float(ap / n_pos)


def _fpr_at_tpr(labels: torch.Tensor, scores: torch.Tensor, target_tpr: float = 0.95) -> float | None:
    labels = labels.detach().cpu().long().view(-1)
    scores = scores.detach().cpu().float().view(-1)
    pos = int((labels == 1).sum().item())
    neg = int((labels == 0).sum().item())
    if pos == 0 or neg == 0 or labels.numel() == 0:
        return None
    _, order = torch.sort(scores, descending=True)
    sorted_labels = labels[order]
    tp = 0
    fp = 0
    best: float | None = None
    for y in sorted_labels.tolist():
        if int(y) == 1:
            tp += 1
        else:
            fp += 1
        tpr = _safe_div(tp, pos)
        if tpr >= float(target_tpr):
            fpr = _safe_div(fp, neg)
            best = fpr if best is None else min(best, fpr)
    return best


def _fixed_ece(labels: torch.Tensor, probs: torch.Tensor, bins: int) -> float:
    labels = labels.detach().cpu().float().view(-1)
    probs = probs.detach().cpu().float().view(-1).clamp(0.0, 1.0)
    n = int(labels.numel())
    if n == 0:
        return 0.0
    edges = torch.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for idx in range(int(bins)):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == int(bins) - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not bool(mask.any()):
            continue
        conf = float(probs[mask].mean().item())
        acc = float(labels[mask].mean().item())
        ece += abs(acc - conf) * _safe_div(int(mask.sum().item()), n)
    return float(ece)


def _adaptive_ece(labels: torch.Tensor, probs: torch.Tensor, bins: int) -> float:
    labels = labels.detach().cpu().float().view(-1)
    probs = probs.detach().cpu().float().view(-1).clamp(0.0, 1.0)
    n = int(labels.numel())
    if n == 0:
        return 0.0
    _, order = torch.sort(probs, descending=False)
    labels = labels[order]
    probs = probs[order]
    ece = 0.0
    for idx in range(int(bins)):
        start = round(idx * n / int(bins))
        end = round((idx + 1) * n / int(bins))
        if end <= start:
            continue
        label_bin = labels[start:end]
        prob_bin = probs[start:end]
        conf = float(prob_bin.mean().item())
        acc = float(label_bin.mean().item())
        ece += abs(acc - conf) * _safe_div(end - start, n)
    return float(ece)


def _capability_final_metrics(labels: torch.Tensor, probs: torch.Tensor, threshold: float, bins: int) -> dict[str, Any]:
    labels = labels.detach().cpu().float().view(-1).clamp(0.0, 1.0)
    probs = probs.detach().cpu().float().view(-1).clamp(0.0, 1.0)
    correct = (labels >= float(threshold)).long()
    pred_correct = (probs >= float(threshold)).long()
    failure = 1 - correct
    pred_failure = 1 - pred_correct
    correct_conf = _binary_confusion(correct, pred_correct)
    failure_conf = _binary_confusion(failure, pred_failure)
    threshold_metrics = _binary_metrics_from_confusion(correct_conf)
    ece_fixed = _fixed_ece(correct.float(), probs, int(bins))
    ece_adaptive = _adaptive_ece(correct.float(), probs, int(bins))
    eps = 1.0e-12
    clamped = probs.clamp(eps, 1.0 - eps)
    nll = -torch.mean(correct.float() * torch.log(clamped) + (1.0 - correct.float()) * torch.log(1.0 - clamped))
    brier = torch.mean(torch.square(probs - correct.float()))
    metrics: dict[str, Any] = {
        "num_rows": int(labels.numel()),
        "num_correct": int(correct.sum().item()),
        "num_incorrect": int(failure.sum().item()),
        "label_threshold": float(threshold),
        "prediction_threshold": float(threshold),
        "mae": float(torch.mean(torch.abs(probs - labels)).item()) if labels.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(torch.square(probs - labels))).item()) if labels.numel() else 0.0,
        "roc_auc": _roc_auc_binary(correct, probs),
        "aupr_c": _average_precision_binary(correct, probs),
        "aupr_i": _average_precision_binary(failure, 1.0 - probs),
        "ece": ece_fixed,
        "ece_fixed_15": ece_fixed if int(bins) == 15 else _fixed_ece(correct.float(), probs, 15),
        "ece_adaptive_15": ece_adaptive if int(bins) == 15 else _adaptive_ece(correct.float(), probs, 15),
        "brier": float(brier.item()) if labels.numel() else 0.0,
        "nll": float(nll.item()) if labels.numel() else 0.0,
        "fpr_at_95_tpr": _fpr_at_tpr(correct, probs, target_tpr=0.95),
        "threshold_0_5": threshold_metrics,
        "threshold_at_failure_threshold": threshold_metrics,
        "thr_acc": threshold_metrics["accuracy"],
        "thr_macro_precision": threshold_metrics["macro_precision"],
        "thr_macro_recall": threshold_metrics["macro_recall"],
        "thr_macro_f1": threshold_metrics["macro_f1"],
        "correctness_confusion": correct_conf.long().tolist(),
        "aupr_correct": None,
        "aupr_incorrect": None,
    }
    metrics["aupr_correct"] = metrics["aupr_c"]
    metrics["aupr_incorrect"] = metrics["aupr_i"]
    metrics.update(_failure_metrics_from_confusion(failure_conf))
    return metrics


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
        "accuracy": float(tp.sum() / torch.clamp(conf_f.sum(), min=1.0)),
        "macro_f1": float(f1.mean()),
        "f1": float(f1.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "confusion": conf.long().tolist(),
        "class_names": names,
    }
    for idx, name in enumerate(names):
        out[f"{name}_f1"] = float(f1[idx])
        out[f"{name}_recall"] = float(recall[idx])
        out[f"{name}_precision"] = float(precision[idx])
        out[f"{name}_support"] = int(row[idx].item())
    return out


@torch.no_grad()
def evaluate_neuron_head(
    args: Any,
    *,
    task: str,
    dataset: ShardedFeatureDataset,
    head: NeuronHead,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=int(args.train_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(args.pin_memory and torch.cuda.is_available()),
        drop_last=False,
    )
    head.eval()
    if task == "capability":
        pred_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        for batch in tqdm(loader, desc="eval capability", dynamic_ncols=True):
            features = ((batch["features"].to(device, non_blocking=True) - mean) / std).float()
            logits = head(features)
            pred_parts.append(torch.sigmoid(logits.squeeze(-1)).detach().cpu())
            label_parts.append(batch["labels"].detach().cpu().float())
        labels = torch.cat(label_parts, dim=0) if label_parts else torch.empty(0)
        preds = torch.cat(pred_parts, dim=0) if pred_parts else torch.empty(0)
        return _capability_final_metrics(
            labels,
            preds,
            threshold=float(args.failure_threshold),
            bins=int(getattr(args, "metric_bins", 15)),
        )

    confusion = torch.zeros(4, 4, dtype=torch.long)
    for batch in tqdm(loader, desc="eval resolution", dynamic_ncols=True):
        features = ((batch["features"].to(device, non_blocking=True) - mean) / std).float()
        logits = head(features)
        probs = torch.sigmoid(logits.detach()).cpu()
        preds = project_behavior_from_probs(probs, float(args.decision_threshold))
        behavior_ids = batch["behavior_ids"].cpu().long()
        usable_cpu = batch["usable_mask"].cpu().long()
        for y, yhat, u in zip(behavior_ids.tolist(), preds.tolist(), usable_cpu.tolist()):
            if int(u) > 0 and int(y) >= 0:
                confusion[int(y), int(yhat)] += 1
    metrics = _confusion_metrics(confusion)
    metrics["decision_threshold"] = float(args.decision_threshold)
    return metrics


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
    metrics = evaluate_neuron_head(
        args,
        task=task,
        dataset=dataset,
        head=head,
        mean=mean,
        std=std,
        device=device,
    )
    metrics["steps"] = opt_step
    if task == "resolution":
        metrics["head_class_names"] = HEAD_CLASS_NAMES
        metrics["behavior_class_names"] = [*HEAD_CLASS_NAMES, "direct_answer"]
        metrics["positive_class_weights"] = pos_weight.detach().cpu().tolist() if pos_weight is not None else None
    else:
        metrics["paper_metric_keys"] = ["roc_auc", "aupr_c", "aupr_i", "ece"]
    write_json(out_dir / "final_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[write] {final_path}", flush=True)
    return final_path
