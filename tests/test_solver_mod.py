from ttt.solver import solve_state


def test_empty_board_is_draw_under_perfect_play():
    empty = tuple([0] * 9)
    sol = solve_state(empty)
    assert sol['value'] == 0
    # should have at least one optimal move
    assert isinstance(sol['optimal_moves'], tuple) and len(sol['optimal_moves']) > 0
    # q_values has 9 entries; all legal on empty board
    qv = list(sol['q_values'])
    assert len(qv) == 9
    assert all(v is not None for v in qv)
