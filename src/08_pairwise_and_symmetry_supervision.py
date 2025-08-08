"""
Teaching module 08: Pairwise preferences and symmetry supervision.

Run:
  python -m src.08_pairwise_and_symmetry_supervision --demo
"""
import argparse
from ttt.solver import solve_all_reachable
from ttt.supervision import generate_pairwise_preferences, generate_symmetry_pairs


def demo() -> None:
    solved = solve_all_reachable()
    pairs = generate_pairwise_preferences(solved)
    spairs = generate_symmetry_pairs(solved)
    print(f"pairwise rows: {len(pairs)}  symmetry pairs: {len(spairs)}")
    # print a couple of samples
    print("sample pairwise:", pairs[0] if pairs else None)
    print("sample symmetry:", spairs[0] if spairs else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 08: Supervision datasets")
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()
    if args.demo:
        demo()
