from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _replace_dir(src_tmp: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    src_tmp.replace(dst)


def materialize_hf_split(
    *,
    dataset_id: str,
    dataset_config: str | None,
    split: str,
    out_dir: Path,
    cache_dir: Path,
    overwrite: bool = False,
    trust_remote_code: bool = False,
) -> str:
    """Download through datasets, then save the actual dataset to out_dir."""
    from datasets import DatasetDict, load_dataset

    done_marker = out_dir / "_mhlc_materialized_manifest.json"
    if done_marker.exists() and not overwrite:
        return "skipped"

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset_config:
        ds = load_dataset(
            dataset_id,
            dataset_config,
            split=split,
            cache_dir=str(cache_dir),
            trust_remote_code=trust_remote_code,
        )
    else:
        ds = load_dataset(
            dataset_id,
            split=split,
            cache_dir=str(cache_dir),
            trust_remote_code=trust_remote_code,
        )

    DatasetDict({split: ds}).save_to_disk(str(tmp_dir))
    _save_json(
        tmp_dir / "_mhlc_materialized_manifest.json",
        {
            "dataset_id": dataset_id,
            "dataset_config": dataset_config,
            "split": split,
            "format": "datasets.save_to_disk",
            "num_rows": len(ds),
        },
    )
    _replace_dir(tmp_dir, out_dir)
    return "downloaded"


def snapshot_dataset_repo(
    *,
    repo_id: str,
    out_dir: Path,
    cache_dir: Path,
    allow_patterns: list[str] | None = None,
    overwrite: bool = False,
) -> str:
    from huggingface_hub import snapshot_download

    marker = out_dir / "_mhlc_snapshot_manifest.json"
    if marker.exists() and not overwrite:
        return "skipped"
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
    )
    _save_json(
        marker,
        {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "snapshot_path": str(snapshot_path),
            "allow_patterns": allow_patterns,
        },
    )
    return "downloaded"
