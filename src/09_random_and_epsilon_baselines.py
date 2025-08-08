"""
Teaching module 09: Random and epsilon-optimal baselines.

Run:
  python -m src.09_random_and_epsilon_baselines --demo
"""
import argparse
from ttt.baselines import random_policy_expectation, epsilon_optimal_expectation


def demo() -> None:
    start = tuple([0] * 9)
    rv = random_policy_expectation(start)
    ev = epsilon_optimal_expectation(start, 0.1)
    print("random exp (value, length):", rv)
    print("epsilon=0.1 exp (value, length):", ev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 09: Baselines & trajectories")
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()
    if args.demo:
        demo()
