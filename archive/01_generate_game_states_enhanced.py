#!/usr/bin/env python3
"""
Thin shim: the original monolithic generator has been archived under `archive/`.
This shim re-exports its public API so tests and existing imports keep working.
For new workflows, prefer `src/10_cli.py export` or `ttt.datasets` with --use-orchestrator.
"""
from pathlib import Path
import importlib.util
import sys as _sys

_ARCHIVED = Path(__file__).resolve().parents[1] / 'archive' / '01_generate_game_states_enhanced.py'
_SRC_COPY = Path(__file__).resolve()  # this file; try to load original if still present via archive

def _load_module_from(path: Path):
    spec = importlib.util.spec_from_file_location('ttt_monolith_archived', str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

_mod = None
if _ARCHIVED.exists():
    _mod = _load_module_from(_ARCHIVED)
else:
    # Fallback: if someone removed archive, try to load ourselves as a module (no-op)
    _mod = _load_module_from(_SRC_COPY)

# Re-export all public attributes
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith('_')})

def main(argv=None):  # delegate CLI behavior if present
    if hasattr(_mod, 'main'):
        return _mod.main(argv)
    # If archived module expects argparse in __main__, mimic call
    if hasattr(_mod, 'generate_tic_tac_toe_dataset'):
        import argparse
        parser = argparse.ArgumentParser(description='Generate Tic-Tac-Toe datasets (archived monolith)')
        parser.add_argument('--no-augmentation', action='store_true')
        parser.add_argument('--no-canonical-only', action='store_true')
        parser.add_argument('--all-states', action='store_true')
        parser.add_argument('--csv', action='store_true')
        parser.add_argument('--no-npz', action='store_true')
        parser.add_argument('--lambda-temp', type=float, default=0.5)
        parser.add_argument('--q-temp', type=float, default=1.0)
        parser.add_argument('--epsilons', type=str, default='0.1')
        parser.add_argument('--states-canonical-only', action='store_true')
        parser.add_argument('--seed', type=int, default=42)
        args = parser.parse_args(argv)
        eps = [float(x) for x in args.epsilons.split(',') if x.strip()]
        return _mod.generate_tic_tac_toe_dataset(
            include_augmentation=not args.no_augmentation,
            canonical_only=not args.no_canonical_only,
            reachable_only=not args.all_states,
            valid_only=not args.all_states,
            export_parquet=not args.csv,
            export_npz=not args.no_npz,
            lambda_temp=args.lambda_temp,
            q_temp=args.q_temp,
            states_canonical_only=args.states_canonical_only,
            epsilons=eps,
            seed=args.seed,
        )
    raise SystemExit('Archived monolith not found or unsupported entry point.')

if __name__ == '__main__':
    raise SystemExit(main())


"""
Archived module (monolith) placeholder.

This file is intentionally disabled as part of the refactor. Do not import or use it.
Use the modular pipeline instead:
- Teaching: src/07_datasets_export.py
- Library/CLI: src/ttt/datasets.py and src/10_cli.py (export)
"""

raise RuntimeError(
    "archive/01_generate_game_states_enhanced.py has been archived. "
    "Use the modular exporter (src/07_datasets_export.py) or the CLI (src/10_cli.py export)."
)
