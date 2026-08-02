from __future__ import annotations

from collections import OrderedDict
from math import floor
from typing import Any


# Mirrors SOURCE_PORTIONS in upstream combined_all_datagen_multimodel.py.
SOURCE_PORTIONS: "OrderedDict[str, float]" = OrderedDict(
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


# Dataset identity mirrors upstream SOURCE_CONFIGS.  Prompt strings live in the
# upstream script and are reused by the processing wrappers.
CAPABILITY_SOURCE_DATASETS: dict[str, dict[str, Any]] = {
    "vqav2": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "vqav2", "split": "train"},
    "scienceqa": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "scienceqa", "split": "train"},
    "chartqa": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "chartqa", "split": "train"},
    "docvqa": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "docvqa", "split": "train"},
    "screenqa": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "screenqa", "split": "train"},
    "aokvqa": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "aokvqa", "split": "train"},
    "ai2d_merged": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "ai2d_merged", "split": "train"},
    "infographic_vqa": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "infographic_vqa", "split": "train"},
    "groundui": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "groundui", "split": "train"},
    "aguvis-stage-1": {"dataset_id": "HuggingFaceM4/FineVision", "dataset_config": "aguvis-stage-1", "split": "train"},
    "aguvis-stage-2": {"dataset_id": "smolagents/aguvis-stage-2", "dataset_config": "android_control", "split": "train"},
    "mm-openr1": {"dataset_id": "lmms-lab/multimodal-open-r1-8k-verified", "dataset_config": None, "split": "train"},
    "dapo": {"dataset_id": "open-r1/DAPO-Math-17k-Processed", "dataset_config": "en", "split": "train"},
    "triviaqa": {"dataset_id": "mandarjoshi/trivia_qa", "dataset_config": "rc", "split": "train"},
    "apigen-mt-5k": {"dataset_id": "Salesforce/APIGen-MT-5k", "dataset_config": None, "split": "train"},
}


BENCHMARK_DATASETS: dict[str, dict[str, Any]] = {
    "mathvista": {"dataset_id": "AI4Math/MathVista", "dataset_config": None, "split": "testmini"},
    "mathverse": {"dataset_id": "AI4Math/MathVerse", "dataset_config": "testmini", "split": "testmini"},
    "charxiv_reasoning": {"dataset_id": "princeton-nlp/CharXiv", "dataset_config": None, "split": "validation"},
    "simplevqa": {"dataset_id": "m-a-p/SimpleVQA", "dataset_config": None, "split": "test"},
    "triviaqa": {"dataset_id": "mandarjoshi/trivia_qa", "dataset_config": "rc", "split": "validation"},
}


CSV_BENCHMARK_TARGETS = {
    "math": "merged_math.csv",
    "mmlu_pro": "test.csv",
}


SCREENSPOT_REPO = "likaixin/ScreenSpot-Pro"
WHEN2CALL_REPO = "nvidia/When2Call"
WHEN2CALL_CONFIGS = ["train_sft", "train_pref", "test"]


def allocate_source_counts(total: int) -> OrderedDict[str, int]:
    ratio_sum = sum(float(v) for v in SOURCE_PORTIONS.values())
    floors: list[tuple[str, int, float]] = []
    used = 0
    for name, portion in SOURCE_PORTIONS.items():
        exact = int(total) * float(portion) / ratio_sum
        base = floor(exact)
        floors.append((name, base, exact - base))
        used += base
    remainder = int(total) - used
    floors.sort(key=lambda x: (-x[2], x[0]))
    extras = {name: 0 for name in SOURCE_PORTIONS}
    for i in range(remainder):
        extras[floors[i][0]] += 1
    base_map = {name: base for name, base, _ in floors}
    return OrderedDict((name, base_map[name] + extras[name]) for name in SOURCE_PORTIONS)
