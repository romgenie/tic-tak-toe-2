"""
Teaching module 05: Positional features and control.
"""
import argparse
from ttt.features_positional import (
    calculate_line_threats,
    calculate_connectivity,
    calculate_control_metrics,
    calculate_pattern_strength,
    calculate_cell_line_potentials,
    calculate_game_phase,
)
from ttt.game_basics import serialize_board


def demo() -> None:
    b = [1,0,2, 0,1,0, 0,0,2]
    print(serialize_board(b))
    print("phase:", calculate_game_phase(b))
    print("control:", calculate_control_metrics(b))
    print("x threats:", calculate_line_threats(b, 1))
    print("o threats:", calculate_line_threats(b, 2))
    print("x conn:", calculate_connectivity(b, 1))
    print("o conn:", calculate_connectivity(b, 2))
    print("x patterns:", calculate_pattern_strength(b, 1))
    print("o patterns:", calculate_pattern_strength(b, 2))
    print("cell lines:", calculate_cell_line_potentials(b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 05: Positional features and control")
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()
    if args.demo:
        demo()
