# Reproducibility

This project aims to be fully reproducible end-to-end: solver, datasets, and checksums.

## Quick start

1) Create a Python 3.11+ environment and install the package in editable mode with dev extras.
2) Export the small canonical dataset (CSV) and verify integrity.

Example using the included Makefile target:

    make reproduce-small

This runs the CLI export with CSV outputs and then executes `scripts/verify_export.py` to check checksums, row counts, and schema hashes.

## Manual commands

If you prefer to run the steps manually:

    python -m tictactoe.cli datasets export --out data_clean/small --format csv --canonical-only
    python scripts/verify_export.py data_clean/small

Expected outputs (tree abbreviated):

    data_clean/small/
    ├── manifest.json
    ├── schema/
    │   ├── ttt_states.schema.json
    │   └── ttt_state_actions.schema.json
    ├── ttt_states.csv
    └── ttt_state_actions.csv

The verify script will fail with a non-zero exit code on any mismatch.

## Notes on Parquet

Parquet export is optional and requires pandas + pyarrow. Install extras with:

    pip install .[parquet]

Use `--format parquet` or `--format both` to write Parquet. If you request `--format parquet` without the dependencies installed, the CLI will raise a clear error. If you request `--format both` without the dependencies, the export will gracefully degrade to CSV-only and still write `manifest.json` with `parquet_written=false`.

## Determinism

CSV rows are written in a deterministic order (sorted by state, then action) to ensure byte-for-byte reproducibility. The manifest records SHA-256 checksums for every exported artifact.

## Environment pinning

For strict archival, use `environment.yml` or `requirements.txt` along with Python 3.11/3.12. Our CI tests against multiple NumPy versions (1.26 and 2.x). For long-term reproduction, we recommend capturing the exact `pip freeze` (or conda env export) with the manifest.
 
