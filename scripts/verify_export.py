#!/usr/bin/env python3
"""
Verify a Tic-tac-toe dataset export directory.

Checks performed:
- manifest.json exists and is parseable
- Row counts in manifest are positive (>0)
- Files listed in manifest exist (if not None)
- SHA256 checksums of files match manifest.checksums
- Row counts in CSVs match manifest.row_counts
- schema_hash matches headers-derived schema (sorted column names)
- schema/*.schema.json files exist and have basic shape (object with properties)

Exit codes:
 0 on success, non-zero on any validation failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def schema_hash_from_csv_header(path: Path) -> str:
    with path.open('r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
    keys = sorted(header)
    payload = "\n".join(keys).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open('r', newline='') as f:
        reader = csv.reader(f)
        # subtract header
        return sum(1 for _ in reader) - 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify Tic-tac-toe dataset export")
    ap.add_argument("out", type=Path, help="Export directory (contains manifest.json)")
    ns = ap.parse_args(argv)
    out = ns.out
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        print(f"ERROR: failed to parse manifest: {e}", file=sys.stderr)
        return 2

    ok = True
    files: Dict[str, Any] = manifest.get("files", {}) or {}
    checksums: Dict[str, Any] = manifest.get("checksums", {}) or {}

    # Manifest row counts must be positive
    rc = manifest.get("row_counts", {}) or {}
    if not isinstance(rc.get("states", None), int) or rc.get("states", 0) <= 0:
        print("ERROR: manifest.row_counts.states must be a positive integer", file=sys.stderr)
        ok = False
    if not isinstance(rc.get("state_actions", None), int) or rc.get("state_actions", 0) <= 0:
        print("ERROR: manifest.row_counts.state_actions must be a positive integer", file=sys.stderr)
        ok = False
    # Validate file existence and checksums
    for label, p in files.items():
        if p is None:
            continue
        fp = Path(p)
        if not fp.exists():
            print(f"ERROR: missing file listed in manifest: {label} -> {fp}", file=sys.stderr)
            ok = False
            continue
        want = checksums.get(label)
        have = sha256_file(fp)
        if want and want != have:
            print(f"ERROR: checksum mismatch for {label}: manifest={want} computed={have}", file=sys.stderr)
            ok = False

    # Validate row counts if CSVs were written
    states_csv = files.get("states_csv")
    sa_csv = files.get("state_actions_csv")
    if states_csv:
        n = count_csv_rows(Path(states_csv))
        if rc.get("states") != n:
            print(f"ERROR: states row count mismatch: manifest={rc.get('states')} actual={n}", file=sys.stderr)
            ok = False
    if sa_csv:
        n = count_csv_rows(Path(sa_csv))
        if rc.get("state_actions") != n:
            print(f"ERROR: state_actions row count mismatch: manifest={rc.get('state_actions')} actual={n}", file=sys.stderr)
            ok = False

    # Validate schema hash using CSV headers as proxy for schema
    sh = manifest.get("schema_hash", {}) or {}
    if states_csv:
        have = schema_hash_from_csv_header(Path(states_csv))
        want = sh.get("states")
        if want and want != have:
            print(f"ERROR: schema_hash(states) mismatch: manifest={want} computed={have}", file=sys.stderr)
            ok = False
    if sa_csv:
        have = schema_hash_from_csv_header(Path(sa_csv))
        want = sh.get("state_actions")
        if want and want != have:
            print(f"ERROR: schema_hash(state_actions) mismatch: manifest={want} computed={have}", file=sys.stderr)
            ok = False

    # Ensure schemas exist and have basic shape
    schema_dir = out / "schema"
    states_schema_path = schema_dir / "ttt_states.schema.json"
    sa_schema_path = schema_dir / "ttt_state_actions.schema.json"
    if not states_schema_path.exists() or not sa_schema_path.exists():
        print("ERROR: schema files not found under export/schema/", file=sys.stderr)
        ok = False
    else:
        try:
            s_states = json.loads(states_schema_path.read_text())
            s_sa = json.loads(sa_schema_path.read_text())
            for label, sch in (("states", s_states), ("state_actions", s_sa)):
                if not isinstance(sch, dict):
                    print(f"ERROR: {label} schema is not a JSON object", file=sys.stderr)
                    ok = False
                    continue
                if sch.get("type") != "object":
                    print(f"ERROR: {label} schema .type must be 'object'", file=sys.stderr)
                    ok = False
                props = sch.get("properties")
                if not isinstance(props, dict) or not props:
                    print(f"ERROR: {label} schema .properties must be a non-empty object", file=sys.stderr)
                    ok = False
        except Exception as e:
            print(f"ERROR: failed to parse schema JSON: {e}", file=sys.stderr)
            ok = False

    if not ok:
        return 1
    print("OK: export verified", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
