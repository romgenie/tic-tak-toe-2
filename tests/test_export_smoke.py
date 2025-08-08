import json
from pathlib import Path

from tictactoe.datasets import ExportArgs, run_export


def test_export_creates_csv_and_manifest(tmp_path: Path):
    out = tmp_path / "exp"
    res = run_export(
        ExportArgs(out=out, canonical_only=True, include_augmentation=False, epsilons=[0.1])
    )
    assert (res / "ttt_states.csv").exists()
    assert (res / "ttt_state_actions.csv").exists()
    manifest = res / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["dataset_version"]
    assert data["row_counts"]["states"] > 0
    assert data["row_counts"]["state_actions"] > 0
