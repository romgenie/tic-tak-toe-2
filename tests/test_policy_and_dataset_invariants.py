from collections import defaultdict
import json
from pathlib import Path
import math

import pytest

from tictactoe.solver import solve_all_reachable
from tictactoe.policy import build_policy_targets, epsilon_policy_distribution
from tictactoe.orchestrator import generate_state_action_dataset
from tictactoe.game_basics import get_winner, is_draw
from tictactoe.symmetry import ALL_SYMS, transform_board, apply_action_transform, symmetry_info


@pytest.fixture(scope="module")
def solved_map():
    return solve_all_reachable()


def _first_nonterminal(solved_map):
    for k, sol in solved_map.items():
        b = [int(c) for c in k]
        if get_winner(b) == 0 and not is_draw(b):
            return b, sol
    raise RuntimeError("No nonterminal state found")


def test_policy_targets_sum_to_one_over_legal(solved_map):
    b, sol = _first_nonterminal(solved_map)
    pol = build_policy_targets(b, sol, lambda_temp=0.5, q_temp=1.0)
    legal = [i for i, v in enumerate(b) if v == 0]
    for name in ["policy_optimal_uniform", "policy_soft_dtt", "policy_soft_q"]:
        v = pol[name]
        assert all((i not in legal and (v[i] == 0.0 or v[i] is None)) or (i in legal) for i in range(9))
        s = sum(v[i] for i in legal)
        assert math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize("eps", [0.0, 0.25, 1.0])
def test_epsilon_policy_distribution_properties(solved_map, eps):
    b, sol = _first_nonterminal(solved_map)
    pol = epsilon_policy_distribution(b, sol, eps)
    legal = [i for i, v in enumerate(b) if v == 0]
    s = sum(pol[i] for i in legal)
    assert math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    if eps == 0.0:
        opts = set(sol['optimal_moves'])
        vals = {pol[i] for i in legal if i in opts}
        assert len(vals) == 1  # uniform among optimal
        for i in legal:
            if i not in opts:
                assert pol[i] == 0.0
    if eps == 1.0:
        vals = {pol[i] for i in legal}
        assert len(vals) == 1  # uniform among legal


def test_generate_state_action_dataset_terminal_states_emit_no_rows(solved_map):
    rows = generate_state_action_dataset(solved_map, include_augmentation=False)
    # All rows should correspond to non-terminal parent boards
    assert len(rows) > 0
    # pick random subset to check
    for r in rows[:100]:
        b = [int(c) for c in r['board_state']]
        assert get_winner(b) == 0 and not is_draw(b)


def test_reward_done_flags_and_nonterminal_consistency(solved_map):
    rows = generate_state_action_dataset(solved_map, include_augmentation=False)
    for r in rows[:200]:
        b = [int(c) for c in r['board_state']]
        p = 1 if b.count(1) == b.count(2) else 2
        i = r['action']
        child = b[:]
        child[i] = p
        w = get_winner(child)
        done = int(w != 0 or is_draw(child))
        reward = 1 if w == p else (-1 if w != 0 else 0)
        assert r['done'] == done
        assert r['reward_if_terminal'] == reward
        if done == 0:
            assert reward == 0


def test_symmetry_augmentation_action_remap_and_canonical(solved_map):
    rows_aug = generate_state_action_dataset(solved_map, include_augmentation=True)
    # Index rows by board_state to find orbits
    by_board = defaultdict(list)
    for r in rows_aug:
        by_board[r['board_state']].append(r)
    # Check a handful of boards
    checked = 0
    for k, lst in by_board.items():
        b = [int(c) for c in k]
        if get_winner(b) != 0 or is_draw(b):
            continue
        info = symmetry_info(b)
        # For each symmetry, there should exist transformed rows with action mapped
        for op in ALL_SYMS:
            tb = transform_board(b, op)
            tkey = ''.join(map(str, tb))
            if tkey not in by_board:
                continue
            # Pick a base row and ensure mapped action exists in target list
            base_row = lst[0]
            a = base_row['action']
            a_t = apply_action_transform(a, op)
            assert any(r['action'] == a_t for r in by_board[tkey])
            # Canonical form matches transformed board's canonical
            tcanon = symmetry_info(tb)['canonical_form']
            for r in by_board[tkey]:
                assert r['canonical_form'] == tcanon
        checked += 1
        if checked >= 5:
            break


def test_include_augmentation_increases_row_count(solved_map):
    rows_base = generate_state_action_dataset(solved_map, include_augmentation=False)
    rows_aug = generate_state_action_dataset(solved_map, include_augmentation=True)
    assert len(rows_aug) >= len(rows_base)
