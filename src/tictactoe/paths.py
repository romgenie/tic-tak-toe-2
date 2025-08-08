"""Centralized path helpers for data locations and metadata.

Environment-first, with robust fallbacks that still work when installed
as a package or executed from arbitrary CWDs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _find_git_root(start: Path) -> Path | None:
    cur = start
    for _ in range(5):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def repo_root() -> Path:
    """Best-effort repository root.

    Order: env var TTT_REPO_ROOT -> nearest parent containing .git -> CWD.
    Avoids writing under site-packages when installed as a library.
    """
    env = os.getenv("TTT_REPO_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    git_root = _find_git_root(here)
    if git_root is not None:
        return git_root
    # final fallback: current working directory
    return Path.cwd()


def data_raw() -> Path:
    p = os.getenv("TTT_DATA_RAW")
    return Path(p) if p else repo_root() / "data_raw"


def data_clean() -> Path:
    p = os.getenv("TTT_DATA_CLEAN")
    return Path(p) if p else repo_root() / "data_clean"


def ensure_dirs() -> None:
    data_raw().mkdir(parents=True, exist_ok=True)
    data_clean().mkdir(parents=True, exist_ok=True)


def get_git_commit() -> str | None:
    """Return the current git commit hash if available.

    Works when running inside a git repo; returns None otherwise.
    """
    root = repo_root()
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        return out.strip()
    except Exception:
        # Try reading .git/HEAD and ref file
        head = root / ".git" / "HEAD"
        try:
            txt = head.read_text().strip()
            if txt.startswith("ref:"):
                ref_path = txt.split()[1]
                ref_file = root / ".git" / ref_path
                if ref_file.exists():
                    return ref_file.read_text().strip()
            return txt if txt else None
        except Exception:
            return None


def get_git_is_dirty() -> bool | None:
    """Return True if there are uncommitted changes, False if clean, None if unknown.

    Uses `git status --porcelain` when available; falls back to None if not a repo.
    """
    root = repo_root()
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        return len(out.strip()) > 0
    except Exception:
        return None
