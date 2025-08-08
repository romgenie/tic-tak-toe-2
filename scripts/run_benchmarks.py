#!/usr/bin/env python3
from __future__ import annotations

import math
import statistics as stats
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from tictactoe.datasets import ExportArgs, run_export
from tictactoe.solver import solve_all_reachable
from tictactoe.tracking import log_artifact, log_metrics, log_params, maybe_mlflow_run


def ci95(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    m = stats.fmean(values)
    s = stats.pstdev(values) if len(values) > 1 else 0.0
    z = 1.96
    half = z * (s / math.sqrt(len(values)))
    return m, half


@dataclass
class Config:
    seeds: int = 10
    tracking: str = "none"  # or "mlflow"
    log_dir: Path = Path("runs")


def main() -> int:
    cfg = Config()
    with maybe_mlflow_run(cfg.tracking == "mlflow", run_name="benchmarks", log_dir=cfg.log_dir):
        log_params({"seeds": cfg.seeds})
        solve_times: List[float] = []
        export_times: List[float] = []
        for s in range(cfg.seeds):
            t0 = time.perf_counter()
            _ = solve_all_reachable()
            t1 = time.perf_counter()
            solve_times.append(t1 - t0)
            outdir = Path("data_raw") / f"bench_seed_{s:03d}"
            t2 = time.perf_counter()
            run_export(ExportArgs(out=outdir, canonical_only=True, include_augmentation=False, epsilons=[0.1]))
            t3 = time.perf_counter()
            export_times.append(t3 - t2)
            log_artifact(outdir / "manifest.json", artifact_path=f"seed_{s:03d}")
        m_solve, h_solve = ci95(solve_times)
        m_export, h_export = ci95(export_times)
        metrics = {
            "solve_mean_s": m_solve,
            "solve_ci95_half_s": h_solve,
            "export_mean_s": m_export,
            "export_ci95_half_s": h_export,
        }
        log_metrics(metrics)
        bench_md = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"
        try:
            old = bench_md.read_text()
        except Exception:
            old = "# Benchmarks\n\n"
        summary = (
            f"\n## Automated summary (N={cfg.seeds})\n\n"
            f"- solve_all_reachable: mean={m_solve:.4f}s ± {h_solve:.4f}s (95% CI)\n"
            f"- export(canonical-only,csv): mean={m_export:.4f}s ± {h_export:.4f}s (95% CI)\n\n"
        )
        bench_md.write_text(old.rstrip() + "\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
