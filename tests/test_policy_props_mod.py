from ttt.policy_targets import build_policy_targets
from ttt.solver import solve_state


def test_policy_distributions_sum_to_one_on_legal_moves():
    board = [0]*9
    sol = solve_state(tuple(board))
    targets = build_policy_targets(board, sol)
    for key in ['policy_optimal_uniform', 'policy_soft_dtt', 'policy_soft_q']:
        pol = targets[key]
        assert abs(sum(pol) - 1.0) < 1e-6
        for i, p in enumerate(pol):
            if board[i] != 0:
                assert p == 0.0
            else:
                assert p >= 0.0
