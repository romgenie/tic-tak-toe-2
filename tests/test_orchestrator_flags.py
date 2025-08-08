import pytest

from tictactoe.solver import solve_all_reachable
from tictactoe.orchestrator import generate_state_action_dataset


def test_canonical_only_reduces_rows():
    solved = solve_all_reachable()
    rows_all = generate_state_action_dataset(solved, canonical_only=False)
    rows_canon = generate_state_action_dataset(solved, canonical_only=True)
    assert len(rows_canon) > 0
    assert len(rows_canon) <= len(rows_all)


def test_include_augmentation_increases_rows():
    solved = solve_all_reachable()
    rows_no_aug = generate_state_action_dataset(solved, include_augmentation=False)
    rows_aug = generate_state_action_dataset(solved, include_augmentation=True)
    assert len(rows_aug) >= len(rows_no_aug)
