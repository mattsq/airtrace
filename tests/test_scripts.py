import subprocess
from types import SimpleNamespace
from typing import List

from scripts.get_models_to_validate import get_changed_files


def _completed_process(lines: List[str]) -> SimpleNamespace:
    return SimpleNamespace(stdout="\n".join(lines))


def test_get_changed_files_handles_missing_base(monkeypatch):
    calls = []

    def fake_run(args, capture_output, text, check):  # type: ignore[override]
        calls.append(list(args))
        if args[:3] == ["git", "merge-base", "origin/main"]:
            raise subprocess.CalledProcessError(returncode=1, cmd=args)
        if list(args) == ["git", "diff", "--name-only"]:
            return _completed_process(["README.md", "src/airtrace/models/mamba2.py"])
        if list(args) == ["git", "diff", "--name-only", "--cached"]:
            return _completed_process(["configs/model/mamba2_ar.yaml"])
        if list(args) == ["git", "ls-files", "--others", "--exclude-standard"]:
            return _completed_process(["tests/models/test_mamba2.py"])
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    files = get_changed_files()

    assert {
        "README.md",
        "src/airtrace/models/mamba2.py",
        "configs/model/mamba2_ar.yaml",
        "tests/models/test_mamba2.py",
    } == files
    assert ["git", "diff", "--name-only"] in calls
    assert ["git", "diff", "--name-only", "--cached"] in calls


def test_get_changed_files_uses_merge_base(monkeypatch):
    def fake_run(args, capture_output, text, check):  # type: ignore[override]
        if args[:3] == ["git", "merge-base", "origin/main"]:
            return _completed_process(["abc123"])
        if list(args) == ["git", "diff", "--name-only", "abc123", "HEAD"]:
            return _completed_process(["src/airtrace/models/gru.py"])
        if list(args) == ["git", "ls-files", "--others", "--exclude-standard"]:
            return _completed_process([])
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    files = get_changed_files()

    assert files == {"src/airtrace/models/gru.py"}
