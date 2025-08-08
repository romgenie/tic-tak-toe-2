#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics as stats
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tictactoe.datasets import ExportArgs, run_export
from tictactoe.paths import get_git_commit, get_git_is_dirty
from tictactoe.solver import solve_all_reachable


@dataclass
class RunConfig:
    mode: str  # "min" or "full"
    seeds: int
    canonical_only: bool
    include_augmentation: bool
    epsilons: List[float]
    normalize_to_move: bool
    format: str


def _set_deterministic(seed: int | None) -> None:
    if seed is not None:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
    for var in (
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(root: Path, cfg: RunConfig, artifacts: Dict[str, Any]) -> None:
    # Collect environment metadata
    locks: Dict[str, Any] = {}
    repo = Path(__file__).resolve().parents[1]
    for name in (
        "requirements-lock.txt",
        "environment.yml",
        "conda-linux-64.lock.txt",
        "conda-osx-64.lock.txt",
        "conda-osx-arm64.lock.txt",
        "conda-win-64.lock.txt",
    ):
        p = repo / name
        if p.exists():
            locks[name] = {"path": str(p), "sha256": _sha256_file(p), "size": p.stat().st_size}

    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "git_is_dirty": get_git_is_dirty(),
        "run_config": asdict(cfg),
        "artifacts": artifacts,
        "environment_locks": locks,
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    # Human-readable overview
    lines = [
        "# Run Manifest",
        "",
        f"Commit: {manifest['git_commit']}",
        f"Dirty: {manifest['git_is_dirty']}",
        f"Mode: {cfg.mode}",
        "",
        "## Artifacts",
    ]
    for k, v in artifacts.items():
        try:
            ch = _sha256_file(Path(str(v)))
            lines.append(f"- {k}: {v}  (sha256={ch})")
        except Exception:
            lines.append(f"- {k}: {v}")
    (root / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def _write_metrics_csv(root: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with (root / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (root / "metrics.json").write_text(json.dumps(rows, indent=2))


def _ci95(values: List[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    m = stats.fmean(values)
    s = stats.pstdev(values) if len(values) > 1 else 0.0
    z = 1.96
    half = z * (s / math.sqrt(len(values)))
    return m, half


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run reproducible experiments and collect artifacts")
    ap.add_argument("mode", choices=["min", "full"], help="Run mode: quick smoke (<1 min) or full")
    ap.add_argument("--seed", type=int, default=0)
    ns = ap.parse_args(argv)

    _set_deterministic(ns.seed)

    # Configure run
    if ns.mode == "min":
        cfg = RunConfig(
            mode="min",
            seeds=1,
            canonical_only=True,
            include_augmentation=False,
            epsilons=[0.1],
            normalize_to_move=False,
            format="csv",
        )
    else:
        cfg = RunConfig(
            mode="full",
            seeds=3,
            canonical_only=False,
            include_augmentation=True,
            epsilons=[0.05, 0.1],
            normalize_to_move=True,
            format="both",
        )

    # Create timestamped results dir
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    out_root = Path("results") / ts
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)

    metrics: List[Dict[str, Any]] = []
    artifacts: Dict[str, Any] = {}

    # Timing: repeated solve timings for CI95
    solve_times: List[float] = []
    for i in range(cfg.seeds):
        t0 = time.perf_counter()
        _ = solve_all_reachable()
        t1 = time.perf_counter()
        solve_times.append(t1 - t0)
    m_solve, h_solve = _ci95(solve_times)
    metrics.append({"metric": "solve_mean_s", "value": m_solve})
    metrics.append({"metric": "solve_ci95_half_s", "value": h_solve})

    # Persist run config as YAML (no external deps; minimal writer)
    rc = asdict(cfg)
    yaml_lines = ["# run-config.yaml"] + [f"{k}: {rc[k]}" for k in rc]
    (out_root / "run-config.yaml").write_text("\n".join(yaml_lines) + "\n")

    # Run dataset export (timed)
    export_dir = out_root / ("export_min" if ns.mode == "min" else "export_full")
    t2 = time.perf_counter()
    export_path = run_export(
        ExportArgs(
            out=export_dir,
            canonical_only=cfg.canonical_only,
            include_augmentation=cfg.include_augmentation,
            epsilons=cfg.epsilons,
            normalize_to_move=cfg.normalize_to_move,
            format=cfg.format,
            verbose=False,
            cli_argv=list(sys.argv),
        )
    )
    t3 = time.perf_counter()
    metrics.append({"metric": "export_elapsed_s", "value": t3 - t2})
    artifacts["export_manifest"] = str(export_path / "manifest.json")
    if (export_path / "ttt_states.csv").exists():
        artifacts["states_csv"] = str(export_path / "ttt_states.csv")
    if (export_path / "ttt_state_actions.csv").exists():
        artifacts["state_actions_csv"] = str(export_path / "ttt_state_actions.csv")
    if (export_path / "ttt_states.parquet").exists():
        artifacts["states_parquet"] = str(export_path / "ttt_states.parquet")
    if (export_path / "ttt_state_actions.parquet").exists():
        artifacts["state_actions_parquet"] = str(export_path / "ttt_state_actions.parquet")

    # Minimal metrics example (row counts)
    manifest = json.loads((export_path / "manifest.json").read_text())
    metrics.append({"metric": "rows_states", "value": manifest["row_counts"]["states"]})
    metrics.append(
        {"metric": "rows_state_actions", "value": manifest["row_counts"]["state_actions"]}
    )

    # Simple ablation (full only): canonical_only vs full+augmentation
    if ns.mode == "full":
        ablation_dir = out_root / "export_ablation_canonical"
        ab_start = time.perf_counter()
        abl_path = run_export(
            ExportArgs(
                out=ablation_dir,
                canonical_only=True,
                include_augmentation=False,
                epsilons=cfg.epsilons,
                normalize_to_move=cfg.normalize_to_move,
                format=cfg.format,
                verbose=False,
                cli_argv=list(sys.argv),
            )
        )
        ab_end = time.perf_counter()
        abl_manifest = json.loads((abl_path / "manifest.json").read_text())
        # Write ablation table
        import csv as _csv

        ab_csv = out_root / "ablation.csv"
        with ab_csv.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["scenario", "rows_states", "rows_state_actions", "export_elapsed_s"])
            w.writerow(
                [
                    "full+aug",
                    manifest["row_counts"]["states"],
                    manifest["row_counts"]["state_actions"],
                    t3 - t2,
                ]
            )
            w.writerow(
                [
                    "canonical_only",
                    abl_manifest["row_counts"]["states"],
                    abl_manifest["row_counts"]["state_actions"],
                    ab_end - ab_start,
                ]
            )
        artifacts["ablation_csv"] = str(ab_csv)

    _write_metrics_csv(out_root, metrics)
    _write_manifest(out_root, cfg, artifacts)
    print(f"Artifacts written under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
