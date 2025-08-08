"""
Datasets helpers for Tic-Tac-Toe using the modular pipeline.

This module builds state and state-action datasets by composing solver and
feature orchestration utilities. No references to the archived monolith remain.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import csv
import importlib.util

from .solver import solve_all_reachable
from .feature_orchestrator import extract_board_features, generate_state_action_dataset


@dataclass
class ExportArgs:
    out: Path


def run_export(args: ExportArgs) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)

    # Generate data with modular pipeline
    solved = solve_all_reachable()
    rows_states: List[Dict[str, Any]] = []
    for key in solved.keys():
        board = [int(c) for c in key]
        rows_states.append(extract_board_features(board, solved))
    rows_sa = generate_state_action_dataset(solved, include_augmentation=True)

    # Write CSV always using stdlib; avoid heavy deps in user envs
    states_csv = args.out / 'ttt_states.csv'
    sa_csv = args.out / 'ttt_state_actions.csv'

    def write_csv(path: Path, rows: List[Dict[str, Any]]):
        fieldnames = set()
        for r in rows:
            fieldnames.update(r.keys())
        fnames = sorted(fieldnames)
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(states_csv, rows_states)
    write_csv(sa_csv, rows_sa)

    # Optionally write Parquet if pandas+pyarrow are available
    if importlib.util.find_spec('pandas') and importlib.util.find_spec('pyarrow'):
        try:
            import pandas as pd  # type: ignore
            df_b = pd.DataFrame(rows_states)
            df_sa = pd.DataFrame(rows_sa)
            df_b.to_parquet(args.out / 'ttt_states.parquet')
            df_sa.to_parquet(args.out / 'ttt_state_actions.parquet')
        except Exception:
            pass

    return args.out
