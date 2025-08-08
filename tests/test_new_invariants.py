import json
import subprocess
import sys
from pathlib import Path

import pytest

from tictactoe.datasets import ExportArgs, run_export
from tictactoe.game_basics import get_winner, is_draw
from tictactoe.orchestrator import extract_board_features
from tictactoe.paths import get_git_commit, get_git_is_dirty
from tictactoe.policy import epsilon_policy_distribution
from tictactoe.solver import solve_all_reachable


@pytest.fixture(scope="module")
def solved_map():
    return solve_all_reachable()


def test_reply_branching_factor_zero_when_all_optimal_are_terminal(solved_map):
    # Find a state where any optimal move wins immediately
    for key, sol in solved_map.items():
        b = [int(c) for c in key]
        if get_winner(b) != 0 or is_draw(b):
            continue
        ok = True
        for mv in sol['optimal_moves']:
            child = b[:]
            child[mv] = 1 if b.count(1) == b.count(2) else 2
            if get_winner(child) == 0 and not is_draw(child):
                ok = False
                break
        if ok and sol['optimal_moves']:
            feats = extract_board_features(b, solved_map)
            assert feats['reply_branching_factor'] == 0.0
            return
    pytest.skip("No such state found in solved map (unexpected)")


def test_reachable_counts_snapshot():
    base = json.loads(Path('tests/data/baselines.json').read_text())
    sol = solve_all_reachable()
    reachable = len(sol)
    term_x = term_o = term_d = 0
    nonterm = 0
    for k in sol:
        b = [int(c) for c in k]
        w = get_winner(b)
        if w != 0:
            if w == 1:
                term_x += 1
            else:
                term_o += 1
        elif is_draw(b):
            term_d += 1
        else:
            nonterm += 1
    assert {
        'reachable': reachable,
        'nonterminal': nonterm,
        'terminal': {'x': term_x, 'o': term_o, 'draw': term_d},
    } == base


ess = [0.0, 0.1, 0.3, 0.6, 1.0]
@pytest.mark.parametrize("eps", ess)
@pytest.mark.parametrize("key", [
    '000000000', # opening
    '100020000', # simple midgame
])
def test_epsilon_policy_monotonicity_and_uniformity(eps, key, solved_map):
    b = [int(c) for c in key]
    sol = solved_map[key]
    pols = [epsilon_policy_distribution(b, sol, e) for e in ess]
    # mass on optimal set should not increase as epsilon increases
    opt = set(sol['optimal_moves'])
    masses = []
    for p in pols:
        masses.append(sum(p[i] for i in opt))
    for i in range(1, len(masses)):
        assert masses[i] <= masses[i-1] + 1e-9
    if eps == 1.0:
        legal = [i for i,c in enumerate(b) if c == 0]
        assert all(abs(pols[-1][i] - (1.0/len(legal))) < 1e-9 for i in legal)


def test_manifest_parquet_written_flag_and_paths(tmp_path: Path):
    out = run_export(ExportArgs(out=tmp_path/"d", canonical_only=True, include_augmentation=False, epsilons=[0.1], format='csv'))
    m = json.loads((out/"manifest.json").read_text())
    assert m['parquet_written'] is False
    assert m['files']['states_parquet'] is None
    assert m['files']['state_actions_parquet'] is None


def test_paths_git_helpers_in_temp_repo(tmp_path: Path, monkeypatch):
    # Initialize a git repo and test commit/dirtiness
    subprocess.run(["git","init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path/"README.md").write_text("test")
    subprocess.run(["git","add","README.md"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git","-c","user.email=a@b.c","-c","user.name=n","commit","-m","init"], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setenv("TTT_REPO_ROOT", str(tmp_path))
    c = get_git_commit()
    d = get_git_is_dirty()
    assert c and len(c) >= 7
    assert d is False
    # make dirty
    (tmp_path/"README.md").write_text("dirty")
    d2 = get_git_is_dirty()
    assert d2 is True


def test_cli_help_smoke(tmp_path: Path):
    exe = [sys.executable, "-m", "tictactoe.cli"]
    for args in (
        ["--help"],
        ["datasets","--help"],
        ["datasets","export","--help"],
        ["symmetry","--help"],
        ["solve","--help"],
        ["tactics","--help"],
    ):
        r = subprocess.run(exe + args, cwd=tmp_path, capture_output=True, text=True)
        assert r.returncode == 0
        assert r.stdout or r.stderr
