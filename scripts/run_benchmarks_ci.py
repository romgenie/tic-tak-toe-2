#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tictactoe.paths import get_git_commit


def main() -> int:
    # Deterministic env
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    repo = Path(__file__).resolve().parents[1]
    benches_dir = repo / "data_raw" / "benchmarks" / get_git_commit()
    benches_dir.mkdir(parents=True, exist_ok=True)

    # Run pytest-benchmark and get raw JSON
    raw_path = benches_dir / "pytest-benchmark.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-k",
        "benchmark",
        "--benchmark-min-time=0.1",
        "--benchmark-warmup=on",
        "--benchmark-warmup-iterations=10",
        "--benchmark-json",
        str(raw_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=repo)

    # Minimal manifest tying commit and environment lock identifiers
    env_lock_info = {}

    def file_sha256(p: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    for name in ["requirements-lock.txt", "environment.yml"]:
        p = repo / name
        if p.exists():
            env_lock_info[name] = {
                "path": str(p),
                "sha256": file_sha256(p),
                "size": p.stat().st_size,
            }

    manifest = {
        "commit": get_git_commit(),
        "artifacts": {
            "pytest_benchmark_json": str(raw_path),
        },
        "environment_locks": env_lock_info,
    }
    (benches_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote benchmark artifacts to", benches_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
