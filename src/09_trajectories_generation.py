"""
Teaching module 09: Trajectory generation with different teacher policies.
"""
import argparse
import json
import os
from ttt.trajectories import generate_trajectories
from ttt.solver import solve_all_reachable


def main():
    ap = argparse.ArgumentParser(description="Module 09: Trajectories")
    ap.add_argument('--policy', choices=['optimal','random','epsilon'], default='optimal')
    ap.add_argument('--epsilon', type=float, default=0.1)
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='data_raw/trajectories.jsonl')
    ap.add_argument('--preview', type=int, default=2)
    args = ap.parse_args()

    # Ensure solved map is computed and cached by solver
    solved = solve_all_reachable()
    games = generate_trajectories(solved, policy=args.policy, epsilon=args.epsilon, max_games=args.games, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        for g in games:
            f.write(json.dumps(g) + "\n")

    print(f"Wrote {len(games)} games to {args.out}")
    for i, g in enumerate(games[:max(0, args.preview)]):
        print(f"Game {i+1} (len={len(g)}): {g[:3]} ...")


if __name__ == "__main__":
    main()
