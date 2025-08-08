import math
from pathlib import Path

from tictactoe.solver import solve_all_reachable
from tictactoe.orchestrator import extract_board_features, generate_state_action_dataset
from tictactoe.game_basics import deserialize_board


def test_state_policy_mass_and_q_values_legality():
    solved = solve_all_reachable()
    # sample a handful of states deterministically
    keys = list(solved.keys())[:50]
    for k in keys:
        board = deserialize_board(k)
        feats = extract_board_features(board, solved, epsilons=[0.05, 0.1])
        legal = [feats[f"legal_{i}"] == 1 for i in range(9)]
        # q_value_i must be in {-1,0,+1} on legal moves; None on illegal
        for i in range(9):
            q = feats[f"q_value_{i}"]
            if legal[i]:
                assert q in (-1, 0, 1)
            else:
                assert q is None
        # policies sum to 1 over legal moves when reachable
        totals = {
            "policy_uniform": 0.0,
            "policy_soft": 0.0,
            "policy_soft_q": 0.0,
        }
        for i in range(9):
            if legal[i]:
                for name in totals:
                    v = feats.get(f"{name}_{i}")
                    if v is not None:
                        totals[name] += v
        if feats["reachable_from_start"]:
            for name, total in totals.items():
                assert math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_best_implies_max_q_value():
    solved = solve_all_reachable()
    keys = list(solved.keys())[:50]
    for k in keys:
        s = solved[k]
        q = s["q_values"]
        legal = [i for i, v in enumerate(q) if v is not None]
        if not legal:
            continue
        max_q = max(q[i] for i in legal)
        for i in legal:
            if i in s["optimal_moves"]:
                assert q[i] == max_q
