from tictactoe.solver import solve_state


def test_empty_board_draw_and_full_optimal_moves():
    s = solve_state(tuple([0]*9))
    assert s['value'] == 0
    assert isinstance(s['optimal_moves'], tuple) and len(s['optimal_moves']) == 9
