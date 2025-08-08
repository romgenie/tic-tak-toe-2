# Benchmarks

This page documents typical performance for solver and export operations and how to reproduce locally.

## Methodology

- Hardware: note CPU model and RAM. Example: Apple M2 Pro, 16GB RAM.
- OS: macOS 14.x or Ubuntu 22.04.
- Python: 3.11 or 3.12.
- NumPy: tested on 1.26.x and 2.x.
- Environment: `pip install -r requirements-lock.txt --require-hashes && pip install -e .` (for archival) or `pip install .[dev]` for local dev. Optionally add `.[parquet]` for Parquet runs.

We use `pytest-benchmark` to time hot paths. Benchmarks are stable and avoid tiny timers via `--benchmark-min-time=0.1`.

### Deterministic protocol

To keep CPU threading and hashing deterministic across runs, set:

```bash
export PYTHONHASHSEED=0
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Then run the benchmarks with a warmup:

```bash
pytest -q -k benchmark --benchmark-min-time=0.1 --benchmark-warmup=on --benchmark-warmup-iterations=10
```

## Reproduce locally

```bash
pip install -r requirements-lock.txt --require-hashes
pip install -e .[dev]
# optional for parquet
pip install .[parquet]

pytest -q -k benchmark --benchmark-min-time=0.1
```

## Sample results (indicative)

- Perfect-play solver (enumerate all reachable states, memoized): typically <100 ms.
- Dataset export (CSV, canonical-only, no augmentation): <500 ms.
- Dataset export (both CSV+Parquet): slightly higher due to serialization overhead.

`pytest-benchmark` output example:

```text
--------------------------------------------------------------------------------------- benchmark: 2 tests --------------------------------------------------------------------------------------
Name (time in ms)                      Min       Max      Mean   StdDev    Median      IQR  Outliers  OPS (Kops/s)  Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------
test_solver_full_enumeration        45.000    60.000    50.000     3.00    49.500    4.000       2;0           20.0      20           1
test_export_small_csv              180.000   260.000   200.000    15.00   198.000   20.000       1;1            5.0      10           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------
```

Numbers vary slightly by machine and Python/NumPy version but are consistently within the same order of magnitude.

## Notes

- All pipelines are pure Python with vectorized components where applicable.
- CSV order is deterministic, which can add minimal sorting cost but is essential for reproducibility.

## CI artifacts

Automated runs publish raw `pytest-benchmark` JSON and a small summary tied to:

- Git commit
- OS/arch and Python version
- NumPy version
- Environment lock identifiers (hash of requirements-lock.txt and/or conda-lock)

See the CI job artifacts for the current branch. The JSON can be downloaded and compared using `pytest-benchmark compare`.
