# Datasets

This page serves as the dataset card for the exported Tic-tac-toe datasets.

Motivation: Provide an exact, exhaustive reference dataset derived from a perfect-play solver, suitable for benchmarking learning methods, verifying invariants, and reproducing published results.

Composition: Two tables are exported.

- ttt_states: one row per reachable state (optionally canonical-only). Includes solver value, plies-to-end, optimal move mask, symmetry info, and rich features.
- ttt_state_actions: one row per legal action in each non-terminal state; includes Q-values, DTT action-tier, optimality mask, and policy targets (uniform-optimal, soft by decision-time tier, and soft by Q).


Generation: Rows are generated via the in-repo solver and feature pipeline. See Reproducibility for commands and stable checksums.

Preprocessing: Only deterministic numerical transforms; no randomness. Optional symmetry augmentation can expand state-action coverage.

Features: Positional control, threats, connectivity, pattern strength, phase, and per-cell open-line counts. Policies obey probability constraints and are zero outside legal moves.

Potential biases: Tic-tac-toe is trivial; policy mass concentrates on optimal moves. Augmentation changes class balance across action rows.

Licensing: MIT; see LICENSE.

Intended uses: Teaching, benchmarking, verifying RL targets. Misuses: Treating counts as representative of stochastic games or non-perfect-play agents.

Determinism & Manifest: The manifest.json includes row counts, checksums, orbit_size_distribution, terminal_split, parquet_written, and schema hashes. The schemas are emitted under export/schema/*.schema.json.

## Parquet outputs

Parquet is written only when explicitly requested via `--format parquet` or `--format both` and when the optional dependencies are installed:

- Install extras: `pip install .[parquet]` (installs pandas and pyarrow).
- Example: `python -m tictactoe.cli datasets export --out data_raw/example --format both --canonical-only`.

If the dependencies are missing and you request `--format parquet`, the command fails early with a clear error. If you request `--format both`, the export gracefully degrades to CSV-only, writes `manifest.json`, and sets `parquet_written=false`.

## Dataset Card

- License: MIT (matches repository license).
- Intended use: benchmarking algorithms and verifying invariants; educational demos of perfect play and symmetry.
- Misuse: inferring properties of stochastic or non-perfect agents; overfitting to trivial game dynamics.
- Determinism: fully deterministic; re-running with identical args produces byte-identical CSVs; manifest contains schema hashes and checksums.
- Reproducibility: see `docs/repro.md` for exact commands and environment notes.
- Reachable counts: captured in `manifest.json` (`row_counts`, `terminal_split`, `orbit_size_distribution`).
- Signatures: per-file SHA-256 in `manifest.json/checksums`.
