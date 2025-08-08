from __future__ import annotations
import argparse
from pathlib import Path
from .datasets import run_export, ExportArgs
from .symmetry import symmetry_info
from .solver import solve_state
from .tactics import immediate_winning_moves, fork_moves
from .game_basics import is_valid_state
import logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ttt", description="Tic-tac-toe CLI")
    sub = p.add_subparsers(dest="cmd")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # datasets group
    p_ds = sub.add_parser("datasets", help="Dataset utilities")
    g = p_ds.add_subparsers(dest="subcmd")

    p_export = g.add_parser("export", help="Export datasets (CSV by default; use --format parquet|both; requires pandas+pyarrow for parquet)")
    p_export.add_argument("--out", type=Path, default=Path("data_raw"), help="Output directory (default: data_raw)")
    p_export.add_argument("--canonical-only", action="store_true", help="Only export canonical states and skip augmentation")
    aug = p_export.add_mutually_exclusive_group()
    aug.add_argument("--include-augmentation", dest="include_augmentation", action="store_true", default=True, help="Include symmetry augmentation (default: on)")
    aug.add_argument("--no-augmentation", dest="include_augmentation", action="store_false", help="Disable symmetry augmentation")
    p_export.add_argument("--epsilons", type=str, default="0.1", help='Comma-separated epsilons, e.g. "0.05,0.1,0.2"')
    p_export.add_argument("--normalize-to-move", action="store_true", help="Compute state features with side-to-move normalized to X")
    p_export.add_argument("--format", choices=["csv", "parquet", "both"], default="csv", help="Export format: csv (default), parquet, or both (parquet requires pandas+pyarrow)")

    # demo: symmetry
    p_sym = sub.add_parser("symmetry", help="Show symmetry info for a board (string of 9 digits, 0=empty,1=X,2=O)")
    p_sym.add_argument("--board", help="Board string, e.g., 100020200 (omit with --stdin)")
    p_sym.add_argument("--stdin", action="store_true", help="Read many boards from stdin and stream CSV output")

    # demo: solve
    p_sol = sub.add_parser("solve", help="Solve a board via perfect play from side-to-move")
    p_sol.add_argument("--board", help="Board string, e.g., 100020200 (omit with --stdin)")
    p_sol.add_argument("--stdin", action="store_true", help="Read many boards from stdin and stream CSV output")

    # demo: tactics
    p_tac = sub.add_parser("tactics", help="List immediate wins and forks for side-to-move")
    p_tac.add_argument("--board", required=True, help="Board string, e.g., 100020200")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(ns, "verbose", False) else logging.INFO,
                        format="[%(levelname)s] %(message)s")

    if ns.cmd == "datasets" and ns.subcmd == "export":
        eps = [float(x) for x in ns.epsilons.split(',') if x.strip()]
        for e in eps:
            if e < 0.0 or e > 1.0:
                logging.error("Epsilon out of range [0,1]: %s", e)
                return 2
        if ns.verbose:
            logging.info("parsed_epsilons=%s", eps)
        out = run_export(ExportArgs(
            out=ns.out,
            canonical_only=ns.canonical_only,
            include_augmentation=ns.include_augmentation,
            epsilons=eps,
            normalize_to_move=ns.normalize_to_move,
            verbose=ns.verbose,
            cli_argv=list(argv) if argv is not None else None,
            format=ns.format,
        ))
        logging.info("Exported datasets to: %s", out)
        return 0

    if ns.cmd == "symmetry":
        import sys as _sys
        if ns.stdin:
            import csv as _csv
            r = _sys.stdin
            w = _csv.writer(_sys.stdout)
            w.writerow(["board","canonical_form","orbit_size","canonical_op"]) 
            for line in r:
                raw = line.strip()
                if not raw:
                    continue
                if len(raw) != 9 or any(c not in "012" for c in raw):
                    continue
                b = [int(c) for c in raw]
                if not is_valid_state(b):
                    continue
                info = symmetry_info(b)
                w.writerow([raw, info['canonical_form'], info['orbit_size'], info['canonical_op']])
            return 0
        else:
            raw = (ns.board or "").strip()
            if len(raw) != 9 or any(c not in "012" for c in raw):
                logging.error("Invalid board string. Must be 9 chars of 0/1/2.")
                return 2
            b = [int(c) for c in raw]
            if not is_valid_state(b):
                logging.error("Board is not a valid reachable state.")
                return 2
            info = symmetry_info(b)
            logging.info("canonical_form=%s orbit_size=%d op=%s", info['canonical_form'], info['orbit_size'], info['canonical_op'])
            return 0

    if ns.cmd == "solve":
        import sys as _sys
        if ns.stdin:
            import csv as _csv
            r = _sys.stdin
            w = _csv.writer(_sys.stdout)
            w.writerow(["board","value","plies_to_end","optimal_moves"]) 
            for line in r:
                raw = line.strip()
                if not raw:
                    continue
                if len(raw) != 9 or any(c not in "012" for c in raw):
                    continue
                b_list = [int(c) for c in raw]
                if not is_valid_state(b_list):
                    continue
                res = solve_state(tuple(b_list))
                w.writerow([raw, res['value'], res['plies_to_end'], ' '.join(map(str,res['optimal_moves']))])
            return 0
        else:
            raw = (ns.board or "").strip()
            if len(raw) != 9 or any(c not in "012" for c in raw):
                logging.error("Invalid board string. Must be 9 chars of 0/1/2.")
                return 2
            b_list = [int(c) for c in raw]
            if not is_valid_state(b_list):
                logging.error("Board is not a valid reachable state.")
                return 2
            b = tuple(b_list)
            res = solve_state(b)
            logging.info("value=%s plies=%s optimal=%s", res['value'], res['plies_to_end'], list(res['optimal_moves']))
            return 0

    if ns.cmd == "tactics":
        raw = ns.board.strip()
        if len(raw) != 9 or any(c not in "012" for c in raw):
            logging.error("Invalid board string. Must be 9 chars of 0/1/2.")
            return 2
        b = [int(c) for c in raw]
        if not is_valid_state(b):
            logging.error("Board is not a valid reachable state.")
            return 2
        p = 1 if b.count(1) == b.count(2) else 2
        logging.info("to_move=%d wins=%s forks=%s", p, immediate_winning_moves(b,p), fork_moves(b,p))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
