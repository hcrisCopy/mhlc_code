from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_layer_top_score_heatmap(
    *,
    score_pack: dict[str, dict[str, torch.Tensor]],
    module_meta: list[dict[str, Any]],
    out_path: Path,
    title: str,
    score_field: str = "score",
    score_label: str = "score",
    top_n: int = 300,
) -> None:
    row_values: list[torch.Tensor] = []
    row_labels: list[str] = []
    max_cols = 0
    for meta in sorted(module_meta, key=lambda item: int(item["layer"])):
        key = str(meta["key"])
        values = score_pack[key][score_field].detach().float().cpu()
        k = max(1, min(int(top_n), values.numel()))
        top_values = torch.topk(values, k).values
        row_values.append(top_values)
        row_labels.append(f"L{int(meta['layer'])}")
        max_cols = max(max_cols, int(top_values.numel()))
    if not row_values:
        return
    plt = _plt()
    matrix = torch.full((len(row_values), max_cols), float("nan"), dtype=torch.float32)
    for row_idx, values in enumerate(row_values):
        matrix[row_idx, : values.numel()] = values

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#f3f4f6")
    fig_width = max(10, min(44, max_cols * 0.035))
    fig_height = max(6, len(row_labels) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(f"Neuron rank within each layer top {max_cols}")
    ax.set_ylabel("Transformer layer")
    ticks = list(range(0, max_cols, max(1, max_cols // 10)))
    if ticks and ticks[-1] != max_cols - 1:
        ticks.append(max_cols - 1)
    ax.set_xticks(ticks, [str(i + 1) for i in ticks], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label=score_label)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_selected_density(
    *,
    layer_rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    if not layer_rows:
        return
    plt = _plt()
    ordered = sorted(layer_rows, key=lambda row: int(row["layer"]))
    layers = [int(row["layer"]) for row in ordered]
    ratios = [float(row["selected_ratio"]) for row in ordered]
    counts = [int(row["selected_neurons"]) for row in ordered]

    fig, ax1 = plt.subplots(figsize=(max(10, len(layers) * 0.35), 5))
    ax1.bar(layers, ratios, color="#2563eb", alpha=0.8)
    ax1.set_xlabel("Transformer layer")
    ax1.set_ylabel("Selected ratio")
    ax1.set_ylim(0, max(0.105, max(ratios) * 1.15))
    ax1.axhline(0.10, color="#dc2626", linestyle="--", linewidth=1.2, label="10% cap")
    ax2 = ax1.twinx()
    ax2.plot(layers, counts, color="#111827", marker="o", linewidth=1.2)
    ax2.set_ylabel("Selected neurons")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_capability_direction(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    if not rows:
        return
    plt = _plt()
    layers = sorted({int(row["layer"]) for row in rows})
    correct = []
    failure = []
    for layer in layers:
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        correct.append(sum(1 for row in layer_rows if row.get("direction") == "correct_high"))
        failure.append(sum(1 for row in layer_rows if row.get("direction") == "failure_high"))
    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.35), 5))
    ax.bar(layers, correct, label="correct_high", color="#16a34a")
    ax.bar(layers, failure, bottom=correct, label="failure_high", color="#ef4444")
    ax.set_title(title)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Selected neurons")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_resolution_classes(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    if not rows:
        return
    plt = _plt()
    classes = ["tool_call", "request_for_info", "cannot_answer"]
    colors = {"tool_call": "#2563eb", "request_for_info": "#f59e0b", "cannot_answer": "#7c3aed"}
    layers = sorted({int(row["layer"]) for row in rows})
    bottoms = [0] * len(layers)
    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.35), 5))
    for cls in classes:
        values = [sum(1 for row in rows if int(row["layer"]) == layer and row.get("best_class") == cls) for layer in layers]
        ax.bar(layers, values, bottom=bottoms, label=cls, color=colors[cls])
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_title(title)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Selected neurons")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
