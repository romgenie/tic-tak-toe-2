import json
from pathlib import Path

import pytest

from tictactoe.datasets import ExportArgs, run_export


def test_format_both_graceful_without_parquet_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Simulate missing pandas/pyarrow by making importlib.find_spec return None
    import importlib

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package=None):  # type: ignore[override]
        if name in {"pandas", "pyarrow"}:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    out = tmp_path / "exp_both"
    res = run_export(ExportArgs(
        out=out,
        canonical_only=True,
        include_augmentation=False,
        epsilons=[0.1],
        format="both",
    ))

    # CSVs exist, manifest exists
    assert (res / "ttt_states.csv").exists()
    assert (res / "ttt_state_actions.csv").exists()
    manifest = json.loads((res / "manifest.json").read_text())
    assert manifest["parquet_written"] is False
    # Parquet files should not exist
    assert not (res / "ttt_states.parquet").exists()
    assert not (res / "ttt_state_actions.parquet").exists()


def test_format_parquet_raises_without_deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import importlib

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, package=None):  # type: ignore[override]
        if name in {"pandas", "pyarrow"}:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    out = tmp_path / "exp_parquet"
    with pytest.raises(RuntimeError):
        run_export(ExportArgs(
            out=out,
            canonical_only=True,
            include_augmentation=False,
            epsilons=[0.1],
            format="parquet",
        ))

    # No partial outputs should exist
    assert not out.exists() or not any(out.iterdir())
