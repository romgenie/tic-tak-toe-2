#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


def main() -> int:
    results = Path("results")
    if not results.exists():
        print("No results directory found")
        return 0
    latest = sorted(results.iterdir())[-1]
    metrics_csv = latest / "metrics.csv"
    if not metrics_csv.exists():
        print("No metrics.csv found in latest run")
        return 0
    rows = list(csv.DictReader(metrics_csv.open()))
    lines = [
        "\n## Latest run metrics\n",
        "| metric | value |",
        "|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['metric']} | {r['value']} |")
    results_md = Path("docs") / "results.md"
    try:
        prev = results_md.read_text()
    except Exception:
        prev = "# Results\n\n"
    results_md.write_text(prev.rstrip() + "\n" + "\n".join(lines) + "\n")
    print(f"Updated {results_md}")

    # Append ablation table if present
    ab_csv = latest / "ablation.csv"
    if ab_csv.exists():
        rows = list(csv.DictReader(ab_csv.open()))
        lines = [
            "\n## Latest ablation\n",
            "| scenario | rows_states | rows_state_actions | export_elapsed_s |",
            "|---|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['scenario']} | {r['rows_states']} | {r['rows_state_actions']} | {r['export_elapsed_s']} |"
            )
        abl_md = Path("docs") / "ablations.md"
        try:
            prev = abl_md.read_text()
        except Exception:
            prev = "# Ablations\n\n"
        abl_md.write_text(prev.rstrip() + "\n" + "\n".join(lines) + "\n")
        print(f"Updated {abl_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
