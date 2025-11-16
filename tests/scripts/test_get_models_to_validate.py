"""Tests for the model selection helper used in CI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Set

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "src/scripts/get_models_to_validate.py"
    spec = importlib.util.spec_from_file_location("get_models_to_validate", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for mypy
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


@pytest.fixture()
def script_module():
    """Load a fresh copy of the script module for each test."""

    return _load_module()


def test_get_changed_files_uses_previous_commit_when_base_matches_head(monkeypatch, script_module):
    """Ensure we diff against HEAD^ when the base ref already equals HEAD."""

    def fake_run_git(args):
        # Only the merge-base command should reach this stub in the test scenario.
        assert args[:2] == ["git", "merge-base"]
        return SimpleNamespace(stdout="basehash\n")

    def fake_git_diff(base: str, head: str) -> Set[str]:
        if (base, head) == ("basehash", "HEAD"):
            return set()
        if (base, head) == ("parent", "HEAD"):
            return {"configs/model/mamba2_ar.yaml", "src/airtrace/models/mamba2.py"}
        raise AssertionError(f"Unexpected diff request: {(base, head)}")

    hashes = {"HEAD": "abc123", "origin/main": "abc123", "HEAD^": "parent"}

    monkeypatch.setattr(script_module, "_run_git_command", fake_run_git)
    monkeypatch.setattr(script_module, "_git_diff_names", fake_git_diff)
    monkeypatch.setattr(script_module, "_get_commit_hash", lambda ref: hashes.get(ref))
    monkeypatch.setattr(script_module, "_list_all_tracked_files", lambda: set())
    monkeypatch.setattr(script_module, "_get_untracked_files", lambda: set())

    changed = script_module.get_changed_files(base_ref="origin/main")
    assert "configs/model/mamba2_ar.yaml" in changed
    assert "src/airtrace/models/mamba2.py" in changed


def test_get_changed_files_lists_tracked_files_without_parent(monkeypatch, script_module):
    """If HEAD has no parent commit we fall back to the entire tracked tree."""

    monkeypatch.setattr(
        script_module,
        "_run_git_command",
        lambda args: SimpleNamespace(stdout="initial\n"),
    )
    monkeypatch.setattr(script_module, "_git_diff_names", lambda *_: set())
    monkeypatch.setattr(
        script_module,
        "_get_commit_hash",
        lambda ref: "initial" if ref in {"HEAD", "origin/main"} else None,
    )
    monkeypatch.setattr(
        script_module,
        "_list_all_tracked_files",
        lambda: {"configs/model/new.yaml", "src/airtrace/models/new.py"},
    )
    monkeypatch.setattr(script_module, "_get_untracked_files", lambda: {"README.md"})

    changed = script_module.get_changed_files(base_ref="origin/main")
    assert changed == {
        "configs/model/new.yaml",
        "src/airtrace/models/new.py",
        "README.md",
    }
