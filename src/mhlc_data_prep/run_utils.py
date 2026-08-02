from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator

from .paths import code_root


@contextlib.contextmanager
def temporary_argv(argv: list[str]) -> Iterator[None]:
    old = sys.argv[:]
    old_cwd = Path.cwd()
    sys.argv = argv[:]
    try:
        os.chdir(code_root())
        yield
    finally:
        sys.argv = old
        os.chdir(old_cwd)


def rel(path: Path | str) -> str:
    """Return a code-root-relative path for upstream argv construction."""
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return os.path.relpath(p, code_root())
    except ValueError:
        # Different Windows drives cannot be represented as a relative path.
        return str(p)


def clean_path(target: Path, allowed_roots: list[Path], label: str) -> None:
    """Remove an old artifact after checking it is inside an allowed root."""
    resolved = target.resolve()
    resolved_roots = [root.resolve() for root in allowed_roots]
    if not any(root in resolved.parents for root in resolved_roots):
        allowed = ", ".join(rel(root) for root in resolved_roots)
        raise ValueError(f"Refusing to clean {label} outside allowed roots: {rel(resolved)}; allowed={allowed}")
    if not resolved.exists():
        print(f"[clean] no existing {label}: {rel(resolved)}")
        return
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()
    print(f"[clean] removed {label}: {rel(resolved)}")
