import json
import subprocess
import sys
from pathlib import Path

import pytest

from tictactoe.datasets import ExportArgs, run_export


def test_run_export_reproducible(tmp_path: Path):
    out1 = tmp_path / "exp1"
    out2 = tmp_path / "exp2"
    args = ExportArgs(out=out1, canonical_only=True, include_augmentation=False, epsilons=[0.05, 0.1])
    run_export(args)
    args2 = ExportArgs(out=out2, canonical_only=True, include_augmentation=False, epsilons=[0.05, 0.1])
    run_export(args2)

    m1 = json.loads((out1 / "manifest.json").read_text())
    m2 = json.loads((out2 / "manifest.json").read_text())
    # Compare fields except created_at
    def strip(m):
        s = dict(m)
        s.pop("created_at", None)
        return s
    s1, s2 = strip(m1), strip(m2)
    assert s1["row_counts"] == s2["row_counts"]
    assert s1["schema_hash"] == s2["schema_hash"]
    # Compare CSV bytes
    b1 = (out1 / "ttt_states.csv").read_bytes()
    b2 = (out2 / "ttt_states.csv").read_bytes()
    assert b1 == b2
    ba1 = (out1 / "ttt_state_actions.csv").read_bytes()
    ba2 = (out2 / "ttt_state_actions.csv").read_bytes()
    assert ba1 == ba2


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    exe = [sys.executable, "-m", "tictactoe.cli"]
    return subprocess.run(exe + args, cwd=cwd, capture_output=True, text=True)


def test_cli_symmetry_solve_and_export(tmp_path: Path):
    # symmetry
    # Use reachable state: after two moves (X at 0, O at 4), X to move
    r = _run_cli(["symmetry", "--board", "100020000"], cwd=tmp_path)
    assert r.returncode == 0
    assert "canonical_form=" in r.stdout or "canonical_form=" in r.stderr
    # solve
    r = _run_cli(["solve", "--board", "100020000"], cwd=tmp_path)
    assert r.returncode == 0
    s = r.stdout + r.stderr
    assert "value=" in s and "plies=" in s and "optimal=" in s
    # export
    outdir = tmp_path / "cli_out"
    r = _run_cli(["datasets", "export", "--out", str(outdir), "--epsilons", "0.05,0.1", "--normalize-to-move"], cwd=tmp_path)
    assert r.returncode == 0
    assert (outdir / "ttt_states.csv").exists()
    assert (outdir / "ttt_state_actions.csv").exists()
    assert (outdir / "manifest.json").exists()


@pytest.mark.parametrize("bad", ["abc", "012345678", "0123456789", "12345678x"])
def test_cli_error_invalid_boards(tmp_path: Path, bad: str):
    r = _run_cli(["symmetry", "--board", bad], cwd=tmp_path)
    assert r.returncode != 0
    r = _run_cli(["solve", "--board", bad], cwd=tmp_path)
    assert r.returncode != 0


def test_cli_error_unreachable_state(tmp_path: Path):
    # Double winner or illegal counts: X and O both win -> unreachable
    bad = "111222111"  # X wins and counts invalid for O to win too
    r = _run_cli(["symmetry", "--board", bad], cwd=tmp_path)
    assert r.returncode != 0
    r = _run_cli(["solve", "--board", bad], cwd=tmp_path)
    assert r.returncode != 0
