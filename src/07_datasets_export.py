"""
Teaching module 07: Dataset export (states and state-actions) with splits and metadata.

This wrapper reuses the monolithic generator `01_generate_game_states_enhanced.py`
by importing it via file path (safe even though the filename starts with digits).

It simply invokes its `generate_tic_tac_toe_dataset` function and prints the expected
artifact paths for convenience.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List


def _load_monolith():
    """Import the monolith file `01_generate_game_states_enhanced.py` by path."""
    here = os.path.dirname(__file__)
    path = os.path.join(here, "01_generate_game_states_enhanced.py")
    spec = importlib.util.spec_from_file_location("ttt_monolith", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


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
    # Ensure output dir exists (monolith writes into data_raw by default)
    os.makedirs(a.out_dir, exist_ok=True)

    monolith = _load_monolith()
    monolith.generate_tic_tac_toe_dataset(
        include_augmentation=a.include_augmentation,
        canonical_only=a.canonical_only,
        reachable_only=a.reachable_only,
        valid_only=a.valid_only,
        export_parquet=a.export_parquet,
        export_npz=a.export_npz,
        lambda_temp=a.lambda_temp,
        q_temp=a.q_temp,
        states_canonical_only=a.states_canonical_only,
        epsilons=a.epsilons or [0.1],
        seed=a.seed,
    )

    # The monolith writes into 'data_raw' unconditionally; move to --out if needed.
    # Try common locations: current working dir, project root (parent of src), then src/.
    cwd_dir = os.path.abspath(os.path.join(os.getcwd(), "data_raw"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    root_dir = os.path.join(project_root, "data_raw")
    src_dir = os.path.join(os.path.dirname(__file__), "data_raw")
    actual_dir = next((d for d in (cwd_dir, root_dir, src_dir) if os.path.isdir(d)), root_dir)
    requested_dir = os.path.abspath(a.out_dir)
    os.makedirs(requested_dir, exist_ok=True)

    def maybe_move(filename: str) -> str | None:
        src = os.path.join(actual_dir, filename)
        if not os.path.exists(src):
            return None
        dst = os.path.join(requested_dir, filename)
        try:
            # If src and dst are same location, just return
            if os.path.abspath(src) == os.path.abspath(dst):
                return dst
            os.replace(src, dst)
            return dst
        except Exception:
            # Fallback: copy then leave original
            try:
                import shutil
                shutil.copy2(src, dst)
                return dst
            except Exception:
                return None

    artifacts = {
        "states": maybe_move("ttt_states.parquet" if a.export_parquet else "ttt_states.csv"),
        "actions": maybe_move("ttt_state_actions.parquet" if a.export_parquet else "ttt_state_actions.csv"),
        "states_splits": maybe_move("ttt_states_splits.parquet" if a.export_parquet else "ttt_states_splits.csv"),
        "actions_splits": maybe_move("ttt_state_actions_splits.parquet" if a.export_parquet else "ttt_state_actions_splits.csv"),
        "states_npz": maybe_move("ttt_states.npz") if a.export_npz else None,
        "actions_npz": maybe_move("ttt_state_actions.npz") if a.export_npz else None,
        "metadata": maybe_move("metadata.json"),
        "canonical_splits": maybe_move("canonical_splits.json"),
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
