from __future__ import annotations

import os
from pathlib import Path


def code_root() -> Path:
    """Return the mhlc_code directory.

    Expected layout:
      mhlc_code/
        Multi-Head-Latent-Control/
        src/
      mhlc_data/
      Qwen/
    """
    return Path(__file__).resolve().parents[2]


def upstream_repo_root() -> Path:
    return code_root() / "Multi-Head-Latent-Control"


def resolve_from_code_root(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (code_root() / path).resolve()


def default_data_root() -> Path:
    return (code_root().parent / "mhlc_data").resolve()


def default_qwen_root() -> Path:
    return (code_root().parent / "Qwen").resolve()


def ensure_mhlc_data_layout(data_root: Path) -> None:
    for rel in [
        "data/train",
        "data/benchmarks",
        "data/sources/capability",
        "data/sources/when2call",
        "downloads/hf_runtime_cache",
        "trained_models",
        "eval_outputs",
        "neurons",
        "visualization",
    ]:
        (data_root / rel).mkdir(parents=True, exist_ok=True)


def set_hf_dirs_inside_data_root(data_root: Path) -> dict[str, str]:
    """Force Hugging Face temporary/runtime files under mhlc_data.

    The final materialized datasets are written to data/sources or
    data/benchmarks.  These env vars keep any unavoidable HF runtime cache out
    of ~/.cache/huggingface.
    """
    hf_home = data_root / "downloads" / "hf_runtime_cache"
    values = {
        "HF_HOME": str(hf_home),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
    }
    for key, value in values.items():
        os.environ[key] = value
    return values

