import os
import sys
import math
import importlib.util


# Add src to path and load the monolith like other tests do
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

spec = importlib.util.spec_from_file_location(
    "ttt_module",
    os.path.join(SRC, "01_generate_game_states_enhanced.py"),
)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)  # type: ignore

solve_state = module.solve_state
build_policy_targets = module.build_policy_targets
transform_board = module.transform_board
apply_action_transform = module.apply_action_transform


def test_policy_normalization_and_symmetry():
    # A mid-game state with some legal moves
    board = [1,0,2, 0,1,0, 0,0,2]
    sol = solve_state(tuple(board))
    pols = build_policy_targets(board, sol)

    legal = [i for i, v in enumerate(board) if v == 0]
    for _, dist in pols.items():
        s = sum(dist[i] for i in legal)
        assert math.isclose(s, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        for i, v in enumerate(board):
            if v != 0:
                assert dist[i] == 0.0

    # Symmetry consistency for soft_q: distribution should permute under transforms
    t = 'rot90'
    tboard = transform_board(board, t)
    sol_t = solve_state(tuple(tboard))
    pol_t = build_policy_targets(tboard, sol_t)['policy_soft_q']
    pol = pols['policy_soft_q']

    for i in range(9):
        j = apply_action_transform(i, t)
        assert math.isclose(pol[i], pol_t[j], rel_tol=1e-6, abs_tol=1e-6)
