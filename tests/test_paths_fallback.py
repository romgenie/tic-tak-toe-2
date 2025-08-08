from pathlib import Path

from tictactoe.paths import data_clean, data_raw, repo_root


def test_repo_root_prefers_cwd_when_no_git_and_no_env(tmp_path: Path, monkeypatch):
    # Ensure no env overrides
    monkeypatch.delenv("TTT_REPO_ROOT", raising=False)
    monkeypatch.delenv("TTT_DATA_RAW", raising=False)
    monkeypatch.delenv("TTT_DATA_CLEAN", raising=False)

    # Simulate running in a directory with no .git present
    monkeypatch.chdir(tmp_path)
    # Also monkeypatch the helper to ensure it doesn't find a git root
    import tictactoe.paths as P

    monkeypatch.setattr(P, "_find_git_root", lambda start: None)

    rr = repo_root()
    assert rr == tmp_path

    # data dirs should resolve under CWD
    assert data_raw() == tmp_path / "data_raw"
    assert data_clean() == tmp_path / "data_clean"
