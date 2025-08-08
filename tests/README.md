# Tests philosophy

- Core invariants and determinism first.
- Optional suites: property-based (hypothesis), benchmarks (pytest-benchmark). These are auto-skipped if dependencies are missing.

Run:

```bash
pytest -q
```

With property-based and benchmark extras:

```bash
pip install .[dev]
pytest -q
```
