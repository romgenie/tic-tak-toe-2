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

    p_export = sub.add_parser("export", help="Generate datasets (delegates to monolith)")
    p_export.add_argument("--out", default="data_raw")
    p_export.add_argument("--no-augmentation", action="store_true")
    p_export.add_argument("--no-canonical-only", action="store_true")
    p_export.add_argument("--all-states", action="store_true")
    p_export.add_argument("--csv", action="store_true")
    p_export.add_argument("--no-npz", action="store_true")
    p_export.add_argument("--lambda-temp", type=float, default=0.5)
    p_export.add_argument("--q-temp", type=float, default=1.0)
    p_export.add_argument("--epsilons", type=str, default="0.1")
    p_export.add_argument("--states-canonical-only", action="store_true")
    p_export.add_argument("--seed", type=int, default=42)
    p_export.add_argument("--use-orchestrator", action="store_true", help="Use modular pipeline instead of monolith")

    args = ap.parse_args(argv)

    if args.cmd == "export":
        # Note: extra flags currently only affect the monolith path; orchestrator uses defaults.
        out_path = Path(args.out)
        run_export(ExportArgs(out=out_path, use_orchestrator=args.use_orchestrator))
        print(json.dumps({"out": str(out_path), "use_orchestrator": bool(args.use_orchestrator)}))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
