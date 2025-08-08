"""
Teaching module 03: Minimax and perfect play.

Tie-break policy:
- Win > Draw > Loss
- Wins and draws: prefer fewer plies to termination
- Losses: prefer more plies (delay)

Run:
  python -m src.03_minimax_perfect_play --demo
"""
import argparse
from ttt.solver import solve_state
from ttt.game_basics import serialize_board


def demo() -> None:
    boards = [
        [0]*9,
        [1,0,0, 0,0,0, 0,0,0],
        [1,2,0, 0,0,0, 0,0,0],
    ]
    for b in boards:
        s = solve_state(tuple(b))
        print(serialize_board(b), s['value'], s['plies_to_end'], s['optimal_moves'])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 03: Minimax and perfect play")
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()
    if args.demo:
        demo()
