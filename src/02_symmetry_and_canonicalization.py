"""
Teaching module 02: Symmetry and canonicalization.

Why it matters:
- Many boards are the “same shape.” Recognizing this reduces data and speeds learning.

Run:
  python -m src.02_symmetry_and_canonicalization --demo
"""
import argparse
from typing import List
from ttt.symmetry import ALL_SYMS, transform_board, symmetry_info, apply_action_transform
from ttt.game_basics import serialize_board


def demo() -> None:
    board = [1,0,2, 0,1,0, 2,0,0]
    print("Board:", serialize_board(board))
    print("All symmetries:")
    for k in ALL_SYMS:
        tb = transform_board(board, k)
        print(f"  {k}: {serialize_board(tb)}")
    sym = symmetry_info(board)
    print("Canonical:", sym['canonical_form'], "via", sym['canonical_op'])
    print("Orbit size:", sym['orbit_size'])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 02: Symmetry and canonicalization")
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()
    if args.demo:
        demo()
