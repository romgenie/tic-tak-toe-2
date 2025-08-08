# Tic-tac-toe: exact solver, symmetry, datasets, CLI

[![CI](https://github.com/romgenie/tic-tak-toe-2/actions/workflows/ci.yml/badge.svg)](https://github.com/romgenie/tic-tak-toe-2/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/romgenie/tic-tak-toe-2/branch/main/graph/badge.svg)](https://codecov.io/gh/romgenie/tic-tak-toe-2)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.0000000.svg)](https://doi.org/10.5281/zenodo.0000000)
[![Reproducible build](https://img.shields.io/badge/reproducible-locked-blue)](docs/repro.md#reproduce-with-locks)

Docs: <https://romgenie.github.io/tic-tak-toe-2/>

Research-grade, deterministic pipeline for Tic-tac-toe: exact memoized minimax with
tie-breaks; symmetry canonicalization and action remapping; composable dataset
builders; and a small CLI.

Archival: releases are archived on Zenodo with a DOI. After cutting a GitHub release, enable Zenodo for this repo and replace the placeholder DOI badge/link above with the minted DOI for that tag.

## Install

Python 3.11+ is required.

Locked, archival installs (recommended for papers):

```bash
# pip-tools hashed lock
pip install -r requirements-lock.txt --require-hashes
pip install -e .
```

Standard dev install:

```bash
pip install -e .[dev]
```

Optional Parquet support (for `--format parquet|both`):

```bash
pip install .[parquet]
```

Optional MLflow tracking:

```bash
pip install .[tracking]
```

Run tests:

```bash
pytest -q
```

## CLI

```bash
# export datasets (CSV by default)
ttt datasets export --out data_raw -v \
    --epsilons 0.05,0.1 --normalize-to-move --format csv

# request both CSV and Parquet (requires pandas+pyarrow installed via .[parquet])
ttt datasets export --out data_raw -v --format both --canonical-only

# show symmetry info (reachable example)
ttt symmetry --board 100020000 -v

# solve a board
ttt solve --board 100020000

# tactics preview
ttt tactics --board 100020200
```

Input validation: board strings must be length 9 of digits 0,1,2 and represent a
reachable state (X starts; counts are legal; no double-winners). Invalid inputs
exit with a non-zero code.

## Datasets

Two CSVs are emitted under the chosen output directory:

- `ttt_states.csv`: one row per reachable state (optionally canonical only).
- `ttt_state_actions.csv`: one row per legal action from each state (with optional
    symmetry augmentation via index remapping; no recomputation).

Parquet files are written only when requested (`--format parquet|both`) and when `pandas` + `pyarrow` are installed. If you request `--format both` without the deps, the export will still succeed with CSVs and `manifest.json` noting `parquet_written=false`.

### Schema highlights

- Common keys: `board_state` (string of 9), `canonical_form`, `orbit_size`.
- State features include `current_player`, `value_current`, `plies_to_end`,
    per-cell `q_value_i`/`dtt_action_i` (None on illegal), and policy distributions
    computed on the normalized board when `normalize_to_move=True`.
        Extended columns (deterministic):
        - Control (symmetric): `x_control_score`, `o_control_score`, `x_control_percentage`,
            `o_control_percentage`, `control_difference`.
        - Threats by player: `x_row_threats`, `x_col_threats`, `x_diag_threats`, `x_total_threats` (and `o_*`).
        - Connectivity by player: `*_connected_pairs`, `*_total_connections`, `*_isolated_pieces`,
            `*_cluster_count`, `*_largest_cluster`.
        - Pattern strength by player: `*_open_lines`, `*_semi_open_lines`, `*_blocked_lines`, `*_potential_lines`.
        - Tactics counts: `x_two_in_row_open`, `o_two_in_row_open`.
        - Cell potentials: `x_cell_open_lines_i`, `o_cell_open_lines_i` for i in 0..8.
        - Game phase: `game_phase` in {opening, midgame, endgame}.
- State-action rows include `action`, `q_value` in {-1,0,+1}, `dtt_action` (plies),
  `optimal_action` (0/1), `policy_optimal_uniform`, `policy_soft_dtt`,
  `policy_soft_q`, and per-epsilon columns `epsilon_policy_XXX`.

Determinism: epsilon policies are analytic; there’s no RNG. Re-running exports with
the same args produces identical outputs. A `manifest.json` is written alongside
exports with args, git commit, dirty flag, python & package versions, row counts, and schema hashes.

### dtt_action (distance to termination)

`dtt_action` is the number of plies until the game ends after taking that action
and then playing optimally on both sides.

Tie-breaks use `dtt_action`:

- Among wins and draws, prefer a smaller `dtt_action`.
- Among losses, prefer a larger `dtt_action` (delay the loss).

### normalize_to_move

When enabled in state feature extraction, piece labels are swapped so the
side-to-move is always X=1. Policies are computed on this normalized board, but
q-values and legality still align, because moves are indexed by cell and
normalization only swaps labels 1<->2.

Terminal states: state-action rows are emitted only for legal (non-terminal)
actions. Epsilon policies assign mass only to legal actions.

### Reference snapshot for archival

For long-term reproducibility, publish a small reference export with its `manifest.json` as a GitHub release asset (and optionally deposit alongside the code on Zenodo). Use the provided target:

```bash
make reproduce-small
```

This writes `data_raw/small/*` and verifies checksums and schema hashes.

Reference snapshot assets are attached to GitHub Releases (see DOI badge) with checksums and a manifest binding to the exact commit and environment locks.

## Reproduce with locks

See docs: docs/repro.md#reproduce-with-locks. We publish both pip hashed locks (`requirements-lock.txt`) and conda-lock files for major platforms.

## Supply chain & provenance

CI generates and uploads a CycloneDX SBOM (sbom.json). Dataset manifests record commit SHA, dirty flag, and lock file hashes. Release assets include the SBOM and lock files.

## Paths & environment

Data directories are configurable via env vars:

- `TTT_DATA_RAW` (defaults to `data_raw/` in repo)
- `TTT_DATA_CLEAN` (defaults to `data_clean/` in repo)

The repo root can be overridden with `TTT_REPO_ROOT`. Otherwise, the nearest
parent containing `.git/` is used; falling back to the current working directory (never site-packages).

## Contributing & CI

This repo ships with formatting (`black`), linting (`ruff`), type-checking (`mypy`),
and tests (`pytest`, `hypothesis`). GitHub Actions runs these on pushes and PRs, and also
exports a small dataset to verify determinism and uploads the CSVs/manifest as artifacts.

Docs: build locally with MkDocs Material or serve live reload:

```bash
make docs-serve
```

## Citation

Please cite via `CITATION.cff`. Example:

> Author(s). Tic-tac-toe: exact solver, symmetry, datasets, and CLI. Version ${version}. DOI: 10.5281/zenodo.xxxxxxx

## License

Code: MIT — see `LICENSE`.

Data: unless otherwise stated, generated datasets are licensed under CC BY 4.0.
