# Reproducibility Guide

This project is designed for deterministic, research-grade reproducibility.

## Environment

- Python: 3.11 (CI also tests 3.10 and 3.12)
- NumPy: 1.26.x and 2.x supported
- Optional: pandas + pyarrow for Parquet

Install with lock (archival):

```bash
pip install -r requirements-lock.txt --require-hashes
pip install -e .[dev]
# Parquet (optional)
pip install .[parquet]
```

Conda (using conda-lock):

```bash
# create from platform-specific lock (example for macOS arm64)
conda create -n ttt --file conda-osx-arm64.lock.txt
conda activate ttt
pip install -e .
```

## Deterministic flags

We set seeds and environment variables to avoid nondeterministic BLAS threading:

- PYTHONHASHSEED
- MKL_NUM_THREADS
- OPENBLAS_NUM_THREADS
- OMP_NUM_THREADS

The CLI exposes `--deterministic` and `--seed`.

## One-shot pipeline

```bash
make reproduce-all
```

This exports datasets (CSV + Parquet), runs multi-seed benchmarks with confidence intervals, and builds the docs.

This project aims to be fully reproducible end-to-end: solver, datasets, and checksums.

## Quick start

1) Create a Python 3.11+ environment and install the package in editable mode with dev extras.
2) Export the small canonical dataset (CSV) and verify integrity.

Example using the included Makefile target:

```bash
make reproduce-small
```

This runs the CLI export with CSV outputs and then executes `scripts/verify_export.py` to check checksums, row counts, and schema hashes.

## Manual commands

If you prefer to run the steps manually:

```bash
python -m tictactoe.cli datasets export --out data_clean/small --format csv --canonical-only
python scripts/verify_export.py data_clean/small
```

Expected outputs (tree abbreviated):

```text
data_clean/small/
├── manifest.json
├── schema/
│   ├── ttt_states.schema.json
│   └── ttt_state_actions.schema.json
├── ttt_states.csv
└── ttt_state_actions.csv
```

The verify script will fail with a non-zero exit code on any mismatch.

## Notes on Parquet

Parquet export is optional and requires pandas + pyarrow. Install extras with:

```bash
pip install .[parquet]
```

Use `--format parquet` or `--format both` to write Parquet. If you request `--format parquet` without the dependencies installed, the CLI will raise a clear error. If you request `--format both` without the dependencies, the export will gracefully degrade to CSV-only and still write `manifest.json` with `parquet_written=false`.

## Determinism

CSV rows are written in a deterministic order (sorted by state, then action) to ensure byte-for-byte reproducibility. The manifest records SHA-256 checksums for every exported artifact.

## Reproduce with locks

Two paths are supported:

- pip with cryptographic hashes: `pip install -r requirements-lock.txt --require-hashes`
- conda with conda-lock: create env from `conda-<platform>.lock.txt` (generated and published in releases)

Set env vars for determinism before running:

```bash
export PYTHONHASHSEED=0
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

## Provenance

CI publishes artifacts (datasets, benchmark JSON, SBOM) tied to the commit SHA and lock file hashes. Each dataset export includes a `manifest.json` with commit, dirty flag, environment lock identifiers, and checksums.
 

## Archival and DOIs

We include a `.zenodo.json` descriptor in the repo. After cutting a GitHub release, enable Zenodo for this repository to mint a DOI and update the README badge to point to the release DOI. Attach the small reference export produced by `make reproduce-small` as a release asset to preserve artifacts and checksums alongside the code.

