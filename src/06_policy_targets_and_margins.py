"""
Teaching module 06: Policy targets and margins.
"""
import argparse
from ttt.policy_targets import build_policy_targets, compute_action_ranks, difficulty_score
from ttt.solver import solve_state
from ttt.game_basics import serialize_board


def demo() -> None:
    b = [1,0,2, 0,1,0, 0,0,2]
    sol = solve_state(tuple(b))
    pol = build_policy_targets(b, sol)
    ranks, vreg, dreg = compute_action_ranks(sol)
    print(serialize_board(b))
    print("policy:", pol)
    print("ranks:", ranks)
    print("difficulty:", difficulty_score(sol))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 06: Policy targets and margins")
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()
    if args.demo:
        demo()
