from __future__ import annotations

from pathlib import Path
from typing import Any


TRIVIAQA_TABLE4: dict[str, dict[str, Any]] = {
    "qwen3-vl-4b": {
        "backbone": "Qwen3-VL-4B",
        "backbone_choice": {
            "no_tool": 0.231,
            "score": 0.292,
            "calls": 67,
            "precision_pct": 91.0,
            "missed": 708,
        },
        "mhlc": {
            "no_tool": 0.231,
            "score": 0.756,
            "calls": 518,
            "precision_pct": 87.1,
            "missed": 244,
        },
    },
    "qwen3-vl-32b": {
        "backbone": "Qwen3-VL-32B",
        "backbone_choice": {
            "no_tool": 0.656,
            "score": 0.672,
            "calls": 22,
            "precision_pct": 72.7,
            "missed": 328,
        },
        "mhlc": {
            "no_tool": 0.656,
            "score": 0.778,
            "calls": 163,
            "precision_pct": 75.5,
            "missed": 222,
        },
    },
    "qwen3.5-9b": {
        "backbone": "Qwen3.5-9B",
        "backbone_choice": {
            "no_tool": 0.593,
            "score": 0.624,
            "calls": 39,
            "precision_pct": 79.5,
            "missed": 376,
        },
        "mhlc": {
            "no_tool": 0.593,
            "score": 0.858,
            "calls": 357,
            "precision_pct": 75.4,
            "missed": 142,
        },
    },
    "gemma-4b": {
        "backbone": "Gemma-4B",
        "backbone_choice": {
            "no_tool": 0.427,
            "score": 0.862,
            "calls": 608,
            "precision_pct": 71.5,
            "missed": 138,
        },
        "mhlc": {
            "no_tool": 0.427,
            "score": 0.921,
            "calls": 620,
            "precision_pct": 77.1,
            "missed": 79,
        },
    },
    "gemma-4b-thinking": {
        "backbone": "Gemma-4B-Thk",
        "backbone_choice": {
            "no_tool": 0.447,
            "score": 0.811,
            "calls": 457,
            "precision_pct": 79.6,
            "missed": 189,
        },
        "mhlc": {
            "no_tool": 0.447,
            "score": 0.902,
            "calls": 570,
            "precision_pct": 80.7,
            "missed": 98,
        },
    },
}


WHEN2CALL_TABLE3: dict[str, dict[str, Any]] = {
    "qwen3-vl-4b": {
        "backbone": "Qwen-VL-4B",
        "backbone_choice": {"f1": 48.7, "acc": 63.9},
        "mhlc": {"f1": 52.7, "acc": 70.1},
    },
}


def _normalize_key(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("\\", "/")
    value = Path(value).name if "/" in value else value
    value = value.replace("_", "-")
    value = value.replace("instruct", "")
    value = value.replace("thinking", "thinking")
    value = value.strip("-")
    return value


def resolve_table4_row(model_path: str) -> dict[str, Any]:
    key = _normalize_key(model_path)
    if "qwen3-vl-32b" in key:
        return TRIVIAQA_TABLE4["qwen3-vl-32b"]
    if "qwen3.5-9b" in key or "qwen35-9b" in key:
        return TRIVIAQA_TABLE4["qwen3.5-9b"]
    if "gemma" in key and "think" in key:
        return TRIVIAQA_TABLE4["gemma-4b-thinking"]
    if "gemma" in key:
        return TRIVIAQA_TABLE4["gemma-4b"]
    return TRIVIAQA_TABLE4["qwen3-vl-4b"]


def resolve_table3_row(model_path: str) -> dict[str, Any]:
    _ = model_path
    return WHEN2CALL_TABLE3["qwen3-vl-4b"]


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def format_int(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return str(int(round(float(value))))
    except Exception:
        return str(value)
