from pathlib import Path

import pytest

from tictactoe.datasets import ExportArgs, run_export
from tictactoe.solver import solve_all_reachable

try:
    HAS_BENCH = True
except Exception:
    HAS_BENCH = False


@pytest.mark.skipif(not HAS_BENCH, reason="pytest-benchmark not installed")
def test_benchmark_solve_all_reachable(benchmark):
    def _solve():
        return solve_all_reachable()

    solved = benchmark(_solve)
    assert isinstance(solved, dict)
    assert len(solved) > 0


@pytest.mark.skipif(not HAS_BENCH, reason="pytest-benchmark not installed")
def test_benchmark_small_export(tmp_path: Path, benchmark):
    def _export():
        return run_export(
            ExportArgs(
                out=tmp_path / "bench",
                canonical_only=True,
                include_augmentation=False,
                epsilons=[0.1],
            )
        )

    out = benchmark(_export)
    assert (out / "ttt_states.csv").exists()
