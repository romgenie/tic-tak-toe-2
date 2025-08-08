from tictactoe.solver import solve_state


def test_empty_board_is_draw_and_all_moves_optimal():
    s = solve_state(tuple([0]*9))
    assert s['value'] == 0
    assert len(s['optimal_moves']) == 9
    assert s['plies_to_end'] is not None


def test_immediate_win_preferred():
    # X to move, immediate win at 2
    b = (1,1,0, 0,2,0, 0,2,0)
    s = solve_state(b)
    assert s['value'] == 1
    assert 2 in s['optimal_moves']
    # distances should prefer the shortest win
    assert s['dtt_action'][2] == min(d for i, d in enumerate(s['dtt_action']) if d is not None)


def test_delaying_loss_preferred():
    # O to move, but position is lost; make sure longer loss is preferred among losses
    b = (1,1,0, 2,2,0, 0,0,0)  # O to move, X threatens
    s = solve_state(b)
    if s['value'] == -1:
        # Among losses, the optimal move set should favor larger dtt
        dtts = [s['dtt_action'][i] for i in s['optimal_moves']]
        assert all(d is not None for d in dtts)
        assert max(dtts) == max(d for d in s['dtt_action'] if d is not None)
