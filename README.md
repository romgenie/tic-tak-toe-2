# Tic-Tac-Toe Research Dataset Generator

Exact game-theoretic solver with symmetry-aware canonicalization and rich ML-friendly features. Exports states and state-action datasets to Parquet/CSV and NPZ.

## Quick start

Create a virtual environment and install pinned deps (avoid conda base):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate datasets (defaults: canonical-only, reachable/valid only, Parquet + NPZ):

```bash
python src/01_generate_game_states_enhanced.py
```

Teaching CLI (same outputs, nicer UX):

```bash
python src/10_cli.py export --out data_raw
```

Use the modular orchestrator (no monolith; CSV fallback if pandas/pyarrow unavailable):

```bash
python src/10_cli.py export --out data_raw --use-orchestrator
```

Optional flags:

- `--csv` export to CSV instead of Parquet
- `--no-npz` disable NPZ export
- `--no-augmentation` disable symmetry augmentation in actions
- `--no-canonical-only` include all symmetry variants
- `--states-canonical-only` deduplicate states by canonical form
- `--epsilons 0.05,0.1,0.2` epsilon values for policy/expectations
- `--lambda-temp 0.5` DTT soft policy temperature
- `--q-temp 1.0` Q softmax temperature
- `--seed 42` randomness for splits
- `--log-level DEBUG` verbose logging

## Outputs

- `data_raw/ttt_states.parquet|csv` — per-state features and targets
- `data_raw/ttt_state_actions.parquet|csv` — per-action features and targets
- `data_raw/ttt_states.npz` — numeric-only arrays (+ planes if present)
- `data_raw/ttt_state_actions.npz`
- Split assignments: `*_splits.*` with train/val/test by canonical form

## Key columns (abridged)

- Perfect play: `value_current`, `plies_to_end`, `optimal_moves_count`, `q_value_i`, `dtt_action_i`
- Policies: `policy_uniform_i`, `policy_soft_i`, `policy_soft_q_i`, `epsilon_policy_XXX_i`
- Symmetry: `canonical_form`, `canonical_op`, `canonical_action_map_i`, `orbit_size`, `orbit_index`
- Tactical: `is_immediate_win`, `is_immediate_block`, `creates_fork`, `blocks_opponent_fork`, `is_safe_move`
- Robustness: `winning_reply_count`, `drawing_reply_count`, `losing_reply_count`, `worst_reply_q`, `safe_1ply`, `safe_2ply`
- Difficulty: `difficulty_score`, `reply_branching_factor`, `reply_branching_after_best_reply`, `child_wins/draws/losses`, margins/gaps
- Planes: `x_plane_i`, `o_plane_i`, `empty_plane_i`, `to_move_plane_i`, `legal_plane_i`, `best_plane_i`

## Notes

- The solver derives exact values and tie-breaks via distance-to-termination.
- Canonical split prevents leakage across symmetric states.
- For correct runtime, use the pinned `requirements.txt` (NumPy<2 with matching pandas/pyarrow).
