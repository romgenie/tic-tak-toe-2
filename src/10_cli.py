"""
Module 10: CLI aggregator for the teaching pathway.
"""
import argparse
import json
import sys
from pathlib import Path
from ttt.datasets import run_export, ExportArgs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ttt-cli", description="Teaching pathway CLI")
    sub = ap.add_subparsers(dest="cmd")

    p_export = sub.add_parser("export", help="Generate datasets (modular pipeline)")
    p_export.add_argument("--out", default="data_raw")
    # Export uses modular pipeline; advanced knobs are available via src/07_datasets_export.py

    # quick demos for teaching modules
    sub.add_parser("demo-01", help="Run module 01 demo")
    sub.add_parser("demo-02", help="Run module 02 demo")
    sub.add_parser("demo-03", help="Run module 03 demo")
    sub.add_parser("demo-04", help="Run module 04 demo")
    sub.add_parser("demo-05", help="Run module 05 demo")
    sub.add_parser("demo-06", help="Run module 06 demo")
    sub.add_parser("demo-08", help="Run module 08 demo")
    sub.add_parser("demo-09", help="Run module 09 demo")

    args = ap.parse_args(argv)

    if args.cmd == "export":
        out_path = Path(args.out)
        run_export(ExportArgs(out=out_path))
        print(json.dumps({"out": str(out_path)}))
        return 0
    elif args.cmd and args.cmd.startswith("demo-"):
        mod_map = {
            'demo-01': 'src.01_game_basics',
            'demo-02': 'src.02_symmetry_and_canonicalization',
            'demo-03': 'src.03_minimax_perfect_play',
            'demo-04': 'src.04_tactics_and_patterns',
            'demo-05': 'src.05_features_positional_control',
            'demo-06': 'src.06_policy_targets_and_margins',
            'demo-08': 'src.08_pairwise_and_symmetry_supervision',
            'demo-09': 'src.09_random_and_epsilon_baselines',
        }
        mod_name = mod_map[args.cmd]
        __import__(mod_name)
        mod = sys.modules[mod_name]
        mod.demo()
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
