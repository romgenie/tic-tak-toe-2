"""
Teaching module 01: Game basics for Tic‑Tac‑Toe.

What you learn here:
- How to represent a game state as numbers.
- How to check for a winner or a draw.
- What makes a state valid (consistent with the rules).

Why it matters:
- Clear, consistent state representation is the foundation for any ML pipeline.
- Simple, deterministic rules are perfect for building intuition.

Run:
  python -m src.01_game_basics --demo
"""
from typing import List
import argparse
from ttt.game_basics import (
    WIN_PATTERNS,
    serialize_board,
    deserialize_board,
    get_winner,
    is_draw,
    is_valid_state,
    get_piece_counts,
    current_player,
)


def demo() -> None:
    examples = [
        [1, 1, 1, 0, 0, 0, 0, 0, 0],  # X win
        [2, 2, 2, 0, 0, 0, 0, 0, 0],  # O win
        [1, 1, 2, 2, 2, 1, 1, 2, 1],  # draw
        [0] * 9,                       # start
    ]
    for b in examples:
        s = serialize_board(b)
        w = get_winner(b)
        print(f"Board {s} winner={w} draw={is_draw(b)} valid={is_valid_state(b)} current={current_player(b)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 01: Game basics")
    ap.add_argument('--demo', action='store_true', help='Print demo examples and explanations')
    args = ap.parse_args()
    if args.demo:
        demo()
