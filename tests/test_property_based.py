from typing import List

import pytest
try:
    from hypothesis import given, strategies as st  # type: ignore
    HAS_HYP = True
except ModuleNotFoundError:  # pragma: no cover - test infra
    HAS_HYP = False
    import pytest as _pytest  # type: ignore
    _pytest.skip("Hypothesis not installed", allow_module_level=True)

from tictactoe.game_basics import is_valid_state, get_winner
from tictactoe.symmetry import ALL_SYMS, transform_board, apply_action_transform


@given(st.lists(st.integers(min_value=0, max_value=2), min_size=9, max_size=9))
def test_is_valid_state_invariants_random(board: List[int]):
    x = board.count(1)
    o = board.count(2)
    w = get_winner(board)
    valid = is_valid_state(board)
    # invalid counts must be False
    if not (x == o or x == o + 1):
        assert valid is False
    # double winner is invalid
    def count_wins(p: int) -> int:
        wins = 0
        for pat in ([0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]):
            if all(board[i] == p for i in pat):
                wins += 1
        return wins
    if count_wins(1) > 0 and count_wins(2) > 0:
        assert valid is False
    # if there is a winner, piece counts align with move order
    if valid and w in (1, 2):
        if w == 1:
            assert x == o + 1
        if w == 2:
            assert x == o


@given(st.integers(min_value=0, max_value=8), st.sampled_from(ALL_SYMS))
def test_apply_action_transform_bijective(idx: int, op: str):
    j = apply_action_transform(idx, op)
    # Ensure mapping is a permutation: 9 unique images
    imgs = {apply_action_transform(i, op) for i in range(9)}
    assert len(imgs) == 9
    assert 0 <= j <= 8


@given(st.lists(st.integers(min_value=0, max_value=2), min_size=9, max_size=9))
def test_rot90_four_times_identity(board: List[int]):
    b = board
    for _ in range(4):
        b = transform_board(b, 'rot90')
    assert b == board


def test_symmetry_composition_closure_on_actions():
    # For any two ops a,b there exists c in ALL_SYMS such that applying a then b equals c
    # Check via action mappings across all i
    for a in ALL_SYMS:
        for b in ALL_SYMS:
            composed = [apply_action_transform(apply_action_transform(i, a), b) for i in range(9)]
            found = False
            for c in ALL_SYMS:
                if composed == [apply_action_transform(i, c) for i in range(9)]:
                    found = True
                    break
            assert found, f"composition {a}∘{b} not closed"
