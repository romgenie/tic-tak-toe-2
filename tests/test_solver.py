import os
import sys

# Add src to path for direct import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "ttt_module",
    os.path.join(SRC, "01_generate_game_states_enhanced.py"),
)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore
is_draw = module.is_draw
solve_state = module.solve_state


def test_terminal_positions_values():
    # X three in a row
    x_win = (1, 1, 1, 0, 0, 0, 0, 0, 0)
    res = solve_state(x_win)
    assert res['value'] == -1
    assert res['plies_to_end'] == 0

    # O three in a row
    o_win = (2, 2, 2, 0, 0, 0, 0, 0, 0)
    res = solve_state(o_win)
    assert res['value'] == -1
    assert res['plies_to_end'] == 0

    # Draw full board no winner
    draw = (1, 1, 2, 2, 2, 1, 1, 2, 1)
    assert is_draw(list(draw))
    res = solve_state(draw)
    assert res['value'] == 0
    assert res['plies_to_end'] == 0


def test_initial_state_is_draw_under_perfect_play():
    start = tuple([0]*9)
    res = solve_state(start)
    assert res['value'] == 0

