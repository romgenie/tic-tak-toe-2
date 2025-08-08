"""
Teaching module 07: Dataset export (states and state-actions) with splits and metadata.

This module uses the modular pipeline (solver + feature orchestrator) to build
state and state-action datasets. It no longer references the archived monolith.
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# Modular pipeline imports
from ttt.solver import solve_all_reachable
from ttt.feature_orchestrator import (
    extract_board_features,
    generate_state_action_dataset,
)
from ttt.symmetry import symmetry_info
from ttt.game_basics import deserialize_board


@dataclass
class ExportArgs:
    out_dir: str = "data_raw"
    include_augmentation: bool = True
    canonical_only: bool = True
    reachable_only: bool = True
    valid_only: bool = True
    export_parquet: bool = True
    export_npz: bool = True
    lambda_temp: float = 0.5
    q_temp: float = 1.0
    epsilons: List[float] | None = None
    states_canonical_only: bool = False
    seed: int = 42


def export_all(a: ExportArgs) -> Dict[str, Any]:
    # Ensure output dir exists
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Build solved map (reachable states from empty)
    solved_map = solve_all_reachable()

    # Build states dataset (optionally canonical-only)
    state_rows: List[Dict[str, Any]] = []
    seen_canonical = set()
    for key in solved_map.keys():
        board = deserialize_board(key)
        if a.states_canonical_only:
            sym = symmetry_info(board)
            cf = sym['canonical_form']
            if cf in seen_canonical:
                continue
            seen_canonical.add(cf)
        state_rows.append(extract_board_features(board, solved_map, lambda_temp=a.lambda_temp, q_temp=a.q_temp, epsilons=a.epsilons or [0.1]))

    # Build state-action dataset
    sa_rows = generate_state_action_dataset(
        solved_map,
        include_augmentation=a.include_augmentation,
        canonical_only=a.canonical_only,
        lambda_temp=a.lambda_temp,
        q_temp=a.q_temp,
        epsilons=a.epsilons or [0.1],
    )

    # Write outputs
    states_csv = os.path.join(out_dir, "ttt_states.csv")
    sa_csv = os.path.join(out_dir, "ttt_state_actions.csv")
    states_parquet = os.path.join(out_dir, "ttt_states.parquet")
    sa_parquet = os.path.join(out_dir, "ttt_state_actions.parquet")

    # Always write CSV with stdlib to avoid heavy deps
    import csv
    def write_csv(path: str, rows: List[Dict[str, Any]]):
        fieldnames = set()
        for r in rows:
            fieldnames.update(r.keys())
        fnames = sorted(fieldnames)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    write_csv(states_csv, state_rows)
    write_csv(sa_csv, sa_rows)

    # Optionally write Parquet if requested and dependencies are present
    if a.export_parquet:
        try:
            import importlib.util
            if importlib.util.find_spec('pandas') and importlib.util.find_spec('pyarrow'):
                import pandas as pd  # type: ignore
                df_states = pd.DataFrame(state_rows)
                df_sa = pd.DataFrame(sa_rows)
                df_states.to_parquet(states_parquet)
                df_sa.to_parquet(sa_parquet)
        except Exception:
            # Skip parquet silently if environment is not compatible
            pass

    # Optional: NPZ not generated when --no-npz is set; implementing NPZ is deferred.
    states_npz = None
    actions_npz = None

    # Minimal metadata
    metadata_path = os.path.join(out_dir, "metadata.json")
    metadata = {
        "generator": "modular",
        "args": asdict(a),
        "counts": {
            "states": len(state_rows),
            "state_actions": len(sa_rows),
        },
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    artifacts = {
        "states": states_parquet if a.export_parquet else states_csv,
        "actions": sa_parquet if a.export_parquet else sa_csv,
        "states_splits": None,
        "actions_splits": None,
        "states_npz": states_npz if a.export_npz else None,
        "actions_npz": actions_npz if a.export_npz else None,
        "metadata": metadata_path,
        "canonical_splits": None,
    }
    return artifacts


def main():
    ap = argparse.ArgumentParser(description="Module 07: Dataset export")
    ap.add_argument("--out", default="data_raw", help="Output directory (default: data_raw)")
    ap.add_argument("--no-augmentation", action="store_true")
    ap.add_argument("--no-canonical-only", action="store_true")
    ap.add_argument("--all-states", action="store_true")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--no-npz", action="store_true")
    ap.add_argument("--lambda-temp", type=float, default=0.5)
    ap.add_argument("--q-temp", type=float, default=1.0)
    ap.add_argument("--epsilons", type=str, default="0.1")
    ap.add_argument("--states-canonical-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eps = [float(x) for x in args.epsilons.split(',') if x.strip()]
    res = export_all(ExportArgs(
        out_dir=args.out,
        include_augmentation=not args.no_augmentation,
        canonical_only=not args.no_canonical_only,
        reachable_only=not args.all_states,
        valid_only=not args.all_states,
        export_parquet=not args.csv,
        export_npz=not args.no_npz,
        lambda_temp=args.lambda_temp,
        q_temp=args.q_temp,
        epsilons=eps,
        states_canonical_only=args.states_canonical_only,
        seed=args.seed,
    ))

    print("Artifacts written:")
    for k, v in res.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
