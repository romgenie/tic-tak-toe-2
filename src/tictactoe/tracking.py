"""
Experiment tracking helpers (optional MLflow backend).

This module is lightweight and only imports MLflow when requested
to avoid adding a hard dependency.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional


@contextmanager
def maybe_mlflow_run(enabled: bool, run_name: str, log_dir: Optional[Path] = None) -> Iterator[None]:
    if not enabled:
        yield None
        return
    try:
        import mlflow  # type: ignore

        if log_dir is not None:
            mlflow.set_tracking_uri((log_dir / "mlruns").as_uri())
        with mlflow.start_run(run_name=run_name):
            yield None
    except Exception:
        # Soft-fail: continue without tracking
        yield None


def log_params(params: Dict[str, object]) -> None:
    try:
        import mlflow  # type: ignore

        mlflow.log_params(params)
    except Exception:
        pass


def log_metrics(metrics: Dict[str, float]) -> None:
    try:
        import mlflow  # type: ignore

        mlflow.log_metrics(metrics)
    except Exception:
        pass


def log_artifact(path: Path, artifact_path: Optional[str] = None) -> None:
    try:
        import mlflow  # type: ignore

        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception:
        pass
