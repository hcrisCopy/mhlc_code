from __future__ import annotations

from collections import OrderedDict
from math import floor
from typing import Any


# Mirrors upstream SOURCE_PORTIONS.  We keep it to derive the text-only share
# from the original mixed 120k recipe.
ORIGINAL_SOURCE_PORTIONS: "OrderedDict[str, float]" = OrderedDict(
    [
        ("vqav2", 2),
        ("scienceqa", 0.5),
        ("chartqa", 0.5),
        ("docvqa", 0.5),
        ("screenqa", 0.5),
        ("aokvqa", 2),
        ("ai2d_merged", 2),
        ("infographic_vqa", 2),
        ("groundui", 0.5),
        ("aguvis-stage-1", 1),
        ("aguvis-stage-2", 2),
        ("mm-openr1", 2),
        ("dapo", 2),
        ("triviaqa", 2),
        ("apigen-mt-5k", 4),
    ]
)


TEXT_CAPABILITY_SOURCE_NAMES = ("dapo", "triviaqa", "apigen-mt-5k")
ORIGINAL_MIXED_TOTAL_QA_PAIRS = 120_000


# Dataset identity mirrors upstream SOURCE_CONFIGS for text sources only.
# Prompt strings live in the upstream script and are reused by the wrappers.
CAPABILITY_SOURCE_DATASETS: dict[str, dict[str, Any]] = {
    "dapo": {"dataset_id": "open-r1/DAPO-Math-17k-Processed", "dataset_config": "en", "split": "train"},
    "triviaqa": {"dataset_id": "mandarjoshi/trivia_qa", "dataset_config": "rc", "split": "train"},
    "apigen-mt-5k": {"dataset_id": "Salesforce/APIGen-MT-5k", "dataset_config": None, "split": "train"},
}


BENCHMARK_DATASETS: dict[str, dict[str, Any]] = {
    "triviaqa": {"dataset_id": "mandarjoshi/trivia_qa", "dataset_config": "rc", "split": "validation"},
}


CSV_BENCHMARK_TARGETS = {
    "math": "merged_math.csv",
    "mmlu_pro": "test.csv",
}


PUBLIC_CSV_BENCHMARK_SOURCES: dict[str, dict[str, Any]] = {
    "math": {
        "target_file": CSV_BENCHMARK_TARGETS["math"],
        "sample_size": 1000,
        "candidates": [
            {"dataset_id": "hendrycks/competition_math", "dataset_config": None, "split": "test"},
            {"dataset_id": "qwedsacf/competition_math", "dataset_config": None, "split": "test"},
        ],
    },
    "mmlu_pro": {
        "target_file": CSV_BENCHMARK_TARGETS["mmlu_pro"],
        "sample_size": 1000,
        "candidates": [
            {"dataset_id": "TIGER-Lab/MMLU-Pro", "dataset_config": None, "split": "test"},
        ],
    },
}


WHEN2CALL_REPO = "nvidia/When2Call"
WHEN2CALL_CONFIGS = ["train_sft", "train_pref", "test"]


def allocate_source_counts(total: int, portions: OrderedDict[str, float] | None = None) -> OrderedDict[str, int]:
    portions = portions or ORIGINAL_SOURCE_PORTIONS
    ratio_sum = sum(float(v) for v in portions.values())
    floors: list[tuple[str, int, float]] = []
    used = 0
    for name, portion in portions.items():
        exact = int(total) * float(portion) / ratio_sum
        base = floor(exact)
        floors.append((name, base, exact - base))
        used += base
    remainder = int(total) - used
    floors.sort(key=lambda x: (-x[2], x[0]))
    extras = {name: 0 for name in portions}
    for i in range(remainder):
        extras[floors[i][0]] += 1
    base_map = {name: base for name, base, _ in floors}
    return OrderedDict((name, base_map[name] + extras[name]) for name in portions)


_original_counts = allocate_source_counts(ORIGINAL_MIXED_TOTAL_QA_PAIRS, ORIGINAL_SOURCE_PORTIONS)
TEXT_SOURCE_COUNTS = OrderedDict((name, _original_counts[name]) for name in TEXT_CAPABILITY_SOURCE_NAMES)
TEXT_TOTAL_QA_PAIRS = sum(TEXT_SOURCE_COUNTS.values())
