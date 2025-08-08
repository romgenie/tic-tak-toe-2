# Tic-Tac-Toe Dataset Card

- Name: ttt_states / ttt_state_actions
- Version: 1.1.0
- License: Code is MIT; generated datasets are licensed under CC BY 4.0 unless otherwise noted.

## Provenance

Data are generated deterministically by the packaged perfect-play solver over all reachable Tic-Tac-Toe states. No external data sources are used.

## Generation Pipeline

- Enumerate all reachable states starting from an empty board.
- Compute solver values (win/draw/loss), optimal moves, depth-to-terminal (DTT), and q-values.
- Derive features: symmetry metadata, positional heuristics, policy targets, and difficulty signals.
- Optionally add symmetry augmentation for state-action records.
- Export CSV and optionally Parquet with a manifest (checksums, schema hash, versions).

## Schema

See schema JSON files under each export directory (schema/*.schema.json). Schemas include a `$schema` field and are versioned implicitly via the `dataset_version` field in the export manifest.

## Splits

No train/val/test splits are predefined; the dataset enumerates the full state space. Consumers may define splits deterministically by hashing the canonical form.

## Integrity

Each export writes a manifest.json with file checksums (sha256) and schema hashes. Use `scripts/verify_export.py` to verify integrity.

## Ethical and Legal

No personal or sensitive data. Fully synthetic.

## Reproducibility

`make reproduce-all` regenerates the reference artifacts deterministically.
