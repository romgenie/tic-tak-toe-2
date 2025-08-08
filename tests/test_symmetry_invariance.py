from tictactoe.solver import solve_state
from tictactoe.symmetry import ALL_SYMS, apply_action_transform, transform_board


def test_value_and_optimal_moves_invariant_under_symmetry():
    # pick a few boards
    boards = [
        [0,0,0,0,0,0,0,0,0],
        [1,0,0,0,2,0,0,0,0],
        [1,2,0,0,1,0,0,0,0],
    ]
    for b in boards:
        base = solve_state(tuple(b))
        for op in ALL_SYMS:
            tb = transform_board(b, op)
            ts = solve_state(tuple(tb))
            assert ts['value'] == base['value']
            # remap optimal moves and compare as sets
            mapped = {apply_action_transform(i, op) for i in base['optimal_moves']}
            assert set(ts['optimal_moves']) == mapped
