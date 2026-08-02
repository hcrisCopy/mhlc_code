from __future__ import annotations

import math
from typing import Any

import torch

from .ffn_hooks import ACTIVATION_DEFINITION
from .stats import HEAD_CLASS_NAMES


def _top_indices(scores: torch.Tensor, limit: int, min_score: float) -> list[int]:
    if limit <= 0 or scores.numel() == 0:
        return []
    candidate = torch.nonzero(scores > float(min_score), as_tuple=False).flatten()
    if candidate.numel() == 0:
        return []
    candidate_scores = scores.index_select(0, candidate)
    values, order = torch.sort(candidate_scores, descending=True)
    selected = candidate.index_select(0, order[: min(limit, order.numel())]).tolist()
    return [int(x) for x in selected]


def select_neurons(
    *,
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    task: str,
    top_ratio: float,
    min_score: float,
    model_tag: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < float(top_ratio) <= 0.10:
        raise ValueError("top_ratio must be in (0, 0.10]. The experiment caps selected neurons at 10%.")

    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for meta in module_meta:
        key = str(meta["key"])
        dim = int(meta["dim"])
        scores = score_pack[key]["score"].float()
        layer_limit = max(1, int(math.floor(float(top_ratio) * dim)))
        selected = _top_indices(scores, layer_limit, min_score)
        selected_scores = [float(scores[idx].item()) for idx in selected]
        layer_rows.append(
            {
                "model_tag": model_tag,
                "task": task,
                "layer": int(meta["layer"]),
                "module": "ffn_intermediate",
                "module_key": key,
                "module_dim": dim,
                "selected_neurons": len(selected),
                "selected_ratio": len(selected) / max(dim, 1),
                "top_ratio_cap": float(top_ratio),
                "score_mean": sum(selected_scores) / len(selected_scores) if selected_scores else 0.0,
                "score_max": max(selected_scores) if selected_scores else 0.0,
                "score_min": min(selected_scores) if selected_scores else 0.0,
            }
        )
        for rank_in_layer, idx in enumerate(selected, start=1):
            row: dict[str, Any] = {
                "model_tag": model_tag,
                "task": task,
                "layer": int(meta["layer"]),
                "module": "ffn_intermediate",
                "module_key": key,
                "index": int(idx),
                "rank_in_layer": int(rank_in_layer),
                "module_dim": dim,
                "selected_neurons_in_layer": len(selected),
                "score": float(scores[idx].item()),
                "top_ratio_cap": float(top_ratio),
                "activation_definition": ACTIVATION_DEFINITION,
            }
            if task == "capability":
                direction_sign = float(score_pack[key]["direction_sign"][idx].item())
                row.update(
                    {
                        "direction_sign": direction_sign,
                        "direction": "correct_high" if direction_sign >= 0 else "failure_high",
                        "separation": float(score_pack[key]["separation"][idx].item()),
                        "weighted_separation": float(score_pack[key]["weighted_separation"][idx].item()),
                        "correlation": float(score_pack[key]["correlation"][idx].item()),
                        "responsiveness": float(score_pack[key]["responsiveness"][idx].item()),
                        "delta": float(score_pack[key]["delta"][idx].item()),
                        "high_mean": float(score_pack[key]["high_mean"][idx].item()),
                        "low_mean": float(score_pack[key]["low_mean"][idx].item()),
                    }
                )
            else:
                best_class_id = int(score_pack[key]["best_class_id"][idx].item())
                direction_sign = float(score_pack[key]["direction_sign"][idx].item())
                row.update(
                    {
                        "best_class": HEAD_CLASS_NAMES[best_class_id],
                        "best_class_id": best_class_id,
                        "direction_sign": direction_sign,
                        "direction": f"{HEAD_CLASS_NAMES[best_class_id]}_high" if direction_sign >= 0 else f"not_{HEAD_CLASS_NAMES[best_class_id]}_high",
                        "score_tool_call": float(score_pack[key]["score_tool_call"][idx].item()),
                        "score_request_for_info": float(score_pack[key]["score_request_for_info"][idx].item()),
                        "score_cannot_answer": float(score_pack[key]["score_cannot_answer"][idx].item()),
                        "delta_tool_call": float(score_pack[key]["delta_tool_call"][idx].item()),
                        "delta_request_for_info": float(score_pack[key]["delta_request_for_info"][idx].item()),
                        "delta_cannot_answer": float(score_pack[key]["delta_cannot_answer"][idx].item()),
                    }
                )
            rows.append(row)

    total_dim = sum(int(meta["dim"]) for meta in module_meta)
    global_limit = int(math.floor(float(top_ratio) * total_dim))
    rows.sort(key=lambda item: (-float(item["score"]), int(item["layer"]), int(item["index"])))
    if len(rows) > global_limit:
        keep_ids = {(row["module_key"], int(row["index"])) for row in rows[:global_limit]}
        rows = rows[:global_limit]
        for layer_row in layer_rows:
            key = layer_row["module_key"]
            layer_row["selected_neurons"] = sum(1 for module_key, _idx in keep_ids if module_key == key)
            layer_row["selected_ratio"] = layer_row["selected_neurons"] / max(int(layer_row["module_dim"]), 1)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows, layer_rows

