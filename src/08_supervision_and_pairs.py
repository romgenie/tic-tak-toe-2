"""
Teaching module 08: Supervision datasets (pairwise preferences, symmetry pairs).
"""
import argparse
from ttt.supervision import generate_pairwise_preferences, generate_symmetry_pairs
from ttt.solver import solve_all_reachable


def main():
    ap = argparse.ArgumentParser(description="Module 08: Supervision datasets")
    ap.add_argument('--preview', type=int, default=5, help='How many examples to print')
    args = ap.parse_args()

    solved = solve_all_reachable()
    pairs = generate_pairwise_preferences(solved)
    symm = generate_symmetry_pairs(solved)

    print(f"pairwise_preferences: {len(pairs)} rows")
    print(f"symmetry_pairs: {len(symm)} rows")

    n = max(0, min(args.preview, len(pairs)))
    if n:
        print("\nSample pairwise preferences:")
        for r in pairs[:n]:
            print(r)
    n2 = max(0, min(args.preview, len(symm)))
    if n2:
        print("\nSample symmetry pairs:")
        for r in symm[:n2]:
            print(r)


if __name__ == "__main__":
    main()
