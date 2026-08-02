from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .paths import upstream_repo_root
from .run_utils import rel


def load_upstream_module(relative_path: str, module_name: str) -> ModuleType:
    """Load an upstream MHLC script without editing it."""
    repo = upstream_repo_root()
    script_path = repo / relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Upstream script not found: {rel(script_path)}")

    # Many upstream scripts import sibling modules by plain name.
    for p in [repo, script_path.parent]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {rel(script_path)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
