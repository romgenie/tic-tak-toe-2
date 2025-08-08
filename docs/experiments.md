# Experiments & Reproducibility

Use Make targets:

- Quick smoke (< 1 minute): make reproduce-min
- Full pipeline (longer): make reproduce

Artifacts are written to results/YYYY-MM-DD_HHMM/ with:

- MANIFEST.json and MANIFEST.md (config + provenance, lock file hashes)
- metrics.csv and metrics.json
- figures/ and logs/
- export_*/ with dataset CSV/Parquet and manifest.json

Determinism:

- Environment seeds set via PYTHONHASHSEED and BLAS threads forced to 1
- No RNG in solver or dataset generation; epsilon policies are analytic
- Re-running with identical config yields identical outputs (checksums in manifest)

