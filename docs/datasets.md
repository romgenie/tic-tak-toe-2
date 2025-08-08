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
