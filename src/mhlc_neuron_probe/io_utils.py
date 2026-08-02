from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from mhlc_data_prep.paths import code_root, default_data_root, ensure_mhlc_data_layout, resolve_from_code_root
from mhlc_data_prep.run_utils import clean_path, rel


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_path(path_like: str | Path | None, default: Path) -> Path:
    if path_like is None or str(path_like).strip() == "":
        return default.resolve()
    return resolve_from_code_root(path_like)


def prepare_data_root(path_like: str | Path | None = None) -> Path:
    root = resolve_path(path_like, default_data_root())
    ensure_mhlc_data_layout(root)
    return root


def model_tag(model_path: str | Path) -> str:
    raw = str(model_path).rstrip("/\\")
    name = Path(raw).name if raw else "model"
    if name in {"", ".", ".."}:
        name = raw
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("_") or "model"


def output_dirs(data_root: Path, tag: str, task: str) -> dict[str, Path]:
    return {
        "neurons": data_root / "neurons" / tag / task,
        "features": data_root / "neurons" / tag / task / "feature_shards",
        "trained": data_root / "trained_models" / "neuron_heads" / tag / task,
        "viz": data_root / "visualization" / "neurons" / tag / task,
    }


def maybe_clean(path: Path, data_root: Path, label: str, clean: bool) -> None:
    if clean:
        clean_path(path, allowed_roots=[data_root], label=label)


def should_skip(out_dir: Path, params: dict[str, Any], expected: Iterable[Path], overwrite: bool) -> bool:
    manifest = out_dir / "manifest.json"
    files = [manifest, *list(expected)]
    if overwrite or not all(path.exists() for path in files):
        return False
    try:
        old = read_json(manifest)
    except Exception:
        return False
    if old.get("params") == params:
        print(f"[skip] existing artifact matches manifest: {rel(out_dir)}", flush=True)
        return True
    return False


def ensure_relative_cwd() -> None:
    os.chdir(code_root())

