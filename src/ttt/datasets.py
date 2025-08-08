"""
Datasets helpers for Tic-Tac-Toe.

Current implementation delegates to the monolith generator for dataset creation
(to avoid duplication during the refactor) and provides a small API to run it
and discover artifacts.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import importlib.util
import shutil
import csv


@dataclass
class ExportArgs:
    out: Path
    use_orchestrator: bool = False  # new: prefer modular pipeline over monolith


def _import_by_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def run_export(args: ExportArgs) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    # Try common locations for data_raw
    candidates = [
        Path.cwd() / 'data_raw',
        Path(__file__).resolve().parents[2] / 'data_raw',  # repo root
        Path(__file__).resolve().parents[1] / 'data_raw',  # src/data_raw
    ]
    data_raw = next((p for p in candidates if p.exists()), None)
    if not data_raw:
        data_raw = args.out  # fallback

    if not args.use_orchestrator:
        # Defer to monolith and move artifacts
        monolith_path = Path(__file__).resolve().parents[1] / '01_generate_game_states_enhanced.py'
        mod = _import_by_path(monolith_path)
        # monolith writes into data_raw itself
        mod.main(['--out', str(data_raw)])
    else:
        # Use modular pipeline to generate artifacts into data_raw
        from .solver import solve_all_reachable
        from .feature_orchestrator import extract_board_features, generate_state_action_dataset
        sol_map = solve_all_reachable()
        # Boards dataset
        rows_b = []
        for key in sol_map.keys():
            board = [int(c) for c in key]
            rows_b.append(extract_board_features(board, sol_map))
        # State-action dataset
        rows_sa = generate_state_action_dataset(sol_map, include_augmentation=True)

        # Try pandas for convenience; fallback to stdlib CSV if unavailable
        try:
            import pandas as pd  # type: ignore
            df_b = pd.DataFrame(rows_b)
            df_b.to_parquet(data_raw / 'ttt_states.parquet')
            df_b.to_csv(data_raw / 'ttt_states.csv', index=False)
            df_sa = pd.DataFrame(rows_sa)
            df_sa.to_parquet(data_raw / 'ttt_state_actions.parquet')
            df_sa.to_csv(data_raw / 'ttt_state_actions.csv', index=False)
        except Exception:
            # CSV-only fallback
            def write_csv(path: Path, rows):
                fieldnames = set()
                for r in rows:
                    fieldnames.update(r.keys())
                fnames = sorted(fieldnames)
                with path.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fnames)
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
            write_csv(data_raw / 'ttt_states.csv', rows_b)
            write_csv(data_raw / 'ttt_state_actions.csv', rows_sa)

    # Move common artifacts to args.out if not already there
    for fname in [
        'ttt_states.parquet', 'ttt_states.csv',
        'ttt_state_actions.parquet', 'ttt_state_actions.csv',
    ]:
        src = data_raw / fname
        if src.exists():
            dst = args.out / fname
            if src != dst:
                shutil.copy2(src, dst)
    return args.out
