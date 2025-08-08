# Changelog

## 1.1.0

- Fix reply_branching_factor terminal handling.
- Manifest parquet_written flag and correct Parquet paths.
- CLI flags: mutually exclusive augmentation; --format; --stdin streaming; epsilon validation.
- Docs site (MkDocs) and CI workflows.
- JSON Schemas and checksums in manifest.

### Maintenance & CI

- CI: Fix NumPy matrix pins to ["1.26.*", "2.*"] and install via `pip install "numpy==${{ matrix.numpy }}"`.
- CI: Add parquet smoke jobs for `--format parquet` and `--format both` with `.[parquet]` extras.
- Packaging: Align requirements.txt to `numpy<3.0` (matching pyproject).

### Behavior changes

- Datasets: `--format both` now gracefully degrades when pandas/pyarrow are missing — CSVs + manifest are written with `parquet_written=false`.
- Datasets: `--format parquet` without deps raises early with a clear error and does not write partial files.
- Paths: `repo_root()` now prefers CWD when not inside a git repo (avoids site-packages); env var `TTT_REPO_ROOT` still takes precedence.

### Docs & CLI

- CLI help: export subcommand now states “CSV by default; use --format parquet|both (requires pandas+pyarrow)”.
- Docs: Fix `docs/repro.md` commands to `python -m tictactoe.cli datasets export ...`.
- Docs: Clarify Parquet instructions in README and `docs/datasets.md`; add dataset card details.
- Docs: Expand `docs/benchmarks.md` with methodology and reproduction guidance.

## 1.2.0 - 2025-08-08

- Pre-commit: add ruff, black, isort, mypy, nbstripout, whitespace and EOF fixers. Add .editorconfig.
- CLI: global `--seed`, `--deterministic`, `--version`, and `--info` flags.
- Tracking: optional MLflow integration via `tictactoe.tracking`; new `scripts/run_benchmarks.py` with CI-friendly stats.
- CI: matrix expanded to OS x Python (3.10–3.12), pip cache, coverage gate set to 90%.
- Make: `reproduce-all` target regenerates datasets, runs benchmarks, and builds docs deterministically.
- Docs: add `docs/data_card.md` and update nav; add `docs/repro.md` guide.
- Packaging: bump version to 1.2.0; add `tracking` extra for `mlflow`.
