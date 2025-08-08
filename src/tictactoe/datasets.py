"""
Datasets helpers for Tic-Tac-Toe using the modular pipeline.

This module builds state and state-action datasets by composing solver and
feature orchestration utilities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import csv
import importlib.util
import json
import hashlib
import logging
from datetime import datetime, timezone
from collections import Counter

from .solver import solve_all_reachable
from .symmetry import symmetry_info
from .orchestrator import extract_board_features, generate_state_action_dataset
from .paths import get_git_commit, get_git_is_dirty


@dataclass
class ExportArgs:
    out: Path
    canonical_only: bool = False
    include_augmentation: bool = True
    epsilons: List[float] = field(default_factory=lambda: [0.1])
    normalize_to_move: bool = False
    verbose: bool = False
    cli_argv: List[str] | None = None
    format: str = "csv"  # one of: "csv", "parquet", "both"


DATASET_VERSION = "1.1.0"


def _schema_hash(rows: List[Dict[str, Any]]) -> str:
    keys = sorted({k for r in rows for k in r.keys()})
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_export(args: ExportArgs) -> Path:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(levelname)s] %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    # Generate data with modular pipeline
    logging.info("Solving all reachable states…")
    solved = solve_all_reachable()
    logging.info("Solved %d states", len(solved))
    rows_states: List[Dict[str, Any]] = []
    term_counts = {"x": 0, "o": 0, "draw": 0}
    for key in solved.keys():
        board = [int(c) for c in key]
        # accumulate terminal split from raw boards
        from .game_basics import get_winner, is_draw
        w = get_winner(board)
        if w == 1:
            term_counts["x"] += 1
        elif w == 2:
            term_counts["o"] += 1
        elif is_draw(board):
            term_counts["draw"] += 1
        if args.canonical_only:
            canon = symmetry_info(board)['canonical_form']
            if key != canon:
                continue
        rows_states.append(
            extract_board_features(
                board,
                solved,
                epsilons=args.epsilons,
                normalize_to_move=args.normalize_to_move,
            )
        )
    logging.info("Building state-action rows%s…",
                 " with augmentation" if args.include_augmentation and not args.canonical_only else "")
    rows_sa = generate_state_action_dataset(
        solved,
        include_augmentation=args.include_augmentation,
        canonical_only=args.canonical_only,
        epsilons=args.epsilons,
    )

    # Prepare output paths
    states_csv = args.out / 'ttt_states.csv'
    sa_csv = args.out / 'ttt_state_actions.csv'
    states_parquet_path = args.out / 'ttt_states.parquet'
    sa_parquet_path = args.out / 'ttt_state_actions.parquet'

    def write_csv(path: Path, rows: List[Dict[str, Any]]):
        fieldnames = set()
        for r in rows:
            fieldnames.update(r.keys())
        fnames = sorted(fieldnames)
        # Deterministic row order: primary by board_state, secondary by action if present
        def _sort_key(r: Dict[str, Any]):
            return (str(r.get('board_state', '')),
                    int(r.get('action', -1)) if isinstance(r.get('action', None), int) else -1,
                    str(r.get('canonical_form', '')))
        rows_sorted = sorted(rows, key=_sort_key)
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)

    # Write according to requested format
    fmt = (args.format or "csv").lower()
    if fmt not in {"csv", "parquet", "both"}:
        raise ValueError(f"Unknown export format: {args.format}")
    wrote_csv = False
    wrote_parquet = False

    if fmt in {"csv", "both"}:
        write_csv(states_csv, rows_states)
        write_csv(sa_csv, rows_sa)
        wrote_csv = True
        logging.info("Wrote CSVs: %s (%d rows), %s (%d rows)",
                     states_csv, len(rows_states), sa_csv, len(rows_sa))

    if fmt in {"parquet", "both"}:
        have_pandas = importlib.util.find_spec('pandas') is not None
        have_pyarrow = importlib.util.find_spec('pyarrow') is not None
        if have_pandas and have_pyarrow:
            try:
                import pandas as pd  # type: ignore
                df_b = pd.DataFrame(rows_states)
                df_sa = pd.DataFrame(rows_sa)
                df_b.to_parquet(states_parquet_path)
                df_sa.to_parquet(sa_parquet_path)
                logging.info("Wrote Parquet files to %s", args.out)
                wrote_parquet = True
            except Exception as e:
                logging.warning("Failed to write Parquet files: %s: %s", type(e).__name__, e)
        else:
            msg = (
                "Parquet dependencies not available (install pandas and pyarrow). "
                "Use pip install .[parquet] to enable parquet support."
            )
            if fmt == "parquet":
                # Strict: user asked only for parquet; fail early before writing any files
                raise RuntimeError(msg)
            else:
                # Graceful degrade for 'both': keep CSVs already written, log warning, and continue
                logging.warning("%s Proceeding with CSV only; manifest will record parquet_written=false.", msg)
                wrote_parquet = False

    # Write manifest with metadata for reproducibility
    # Environment & versions
    py_info: Dict[str, Any] = {
        "python_version": None,
        "packages": {},
    }
    try:
        import sys
        py_info["python_version"] = sys.version.split(" ")[0]
    except Exception:
        pass
    for pkg in ["numpy", "pandas", "pyarrow"]:
        try:
            spec = importlib.util.find_spec(pkg)
            if spec is not None:
                mod = __import__(pkg)  # type: ignore
                ver = getattr(mod, "__version__", None)
                if ver:
                    py_info["packages"][pkg] = ver
        except Exception:
            continue

    # Orbit size distribution
    orbit_counts = Counter([r.get('orbit_size', None) for r in rows_states])
    if None in orbit_counts:
        orbit_counts.pop(None)

    # Checksums for files we wrote
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    files: Dict[str, Any] = {
        "states_csv": str(states_csv) if wrote_csv else None,
        "state_actions_csv": str(sa_csv) if wrote_csv else None,
        "states_parquet": str(states_parquet_path) if wrote_parquet else None,
        "state_actions_parquet": str(sa_parquet_path) if wrote_parquet else None,
    }
    checksums: Dict[str, Any] = {}
    for label, p in files.items():
        if p is not None:
            try:
                checksums[label] = sha256_file(Path(p))
            except Exception:
                checksums[label] = None

    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            "canonical_only": args.canonical_only,
            "include_augmentation": args.include_augmentation,
            "epsilons": args.epsilons,
            "normalize_to_move": args.normalize_to_move,
            "format": args.format,
        },
        "git_commit": get_git_commit(),
        "git_is_dirty": get_git_is_dirty(),
        "python": py_info,
        "cli_argv": args.cli_argv,
        "row_counts": {
            "states": len(rows_states),
            "state_actions": len(rows_sa),
        },
        "terminal_split": term_counts,
        "orbit_size_distribution": dict(sorted(orbit_counts.items())),
        "schema_hash": {
            "states": _schema_hash(rows_states) if rows_states else None,
            "state_actions": _schema_hash(rows_sa) if rows_sa else None,
        },
        "files": files,
        "checksums": checksums,
        "parquet_written": wrote_parquet,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info("Wrote manifest.json with metadata and schema hashes")

    # Emit JSON Schemas
    schema_dir = args.out / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    def infer_schema(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        for k in sorted({kk for r in rows for kk in r.keys()}):
            # Determine type based on first non-null value
            t = "string"
            for r in rows:
                v = r.get(k, None)
                if v is None:
                    continue
                if isinstance(v, bool):
                    t = "boolean"
                elif isinstance(v, int):
                    t = "integer"
                elif isinstance(v, float):
                    t = "number"
                else:
                    t = "string"
                break
            props[k] = {"type": [t, "null"]}
        return {"$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": props,
                "additionalProperties": False}

    (schema_dir / "ttt_states.schema.json").write_text(json.dumps(infer_schema(rows_states), indent=2))
    (schema_dir / "ttt_state_actions.schema.json").write_text(json.dumps(infer_schema(rows_sa), indent=2))
    logging.info("Wrote JSON Schemas to %s", schema_dir)

    return args.out
