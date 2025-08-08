"""
Teaching module 04: Tactics and patterns.

Run:
  python -m src.04_tactics_and_patterns --demo
"""
import argparse
from ttt.tactics import immediate_winning_moves, fork_moves
from ttt.game_basics import serialize_board


def demo() -> None:
    board = [1,0,1, 0,2,0, 0,0,2]
    print(serialize_board(board))
    print("X immediate wins:", immediate_winning_moves(board, 1))
    print("O immediate wins:", immediate_winning_moves(board, 2))
    print("X forks:", fork_moves(board, 1))
    print("O forks:", fork_moves(board, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 04: Tactics and patterns")
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()
    if args.demo:
        demo()
