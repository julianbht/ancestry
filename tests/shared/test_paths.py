"""Tests for the ANCESTRY_DATA_DIR override in src/pipeline/shared/paths.py.

DATA_DIR is resolved at import time, so each case reloads the module under a
controlled environment. A teardown reload restores the default so other tests
see the normal data/ root.
"""

import importlib

import pytest

import pipeline.shared.paths as paths_mod


@pytest.fixture
def reload_paths(monkeypatch):
    def _reload(value: str | None):
        if value is None:
            monkeypatch.delenv("ANCESTRY_DATA_DIR", raising=False)
        else:
            monkeypatch.setenv("ANCESTRY_DATA_DIR", value)
        importlib.reload(paths_mod)
        return paths_mod

    yield _reload
    monkeypatch.delenv("ANCESTRY_DATA_DIR", raising=False)
    importlib.reload(paths_mod)


def test_default_is_repo_data(reload_paths):
    paths = reload_paths(None)
    assert paths.DATA_DIR == paths.PROJECT_ROOT / "data"


def test_relative_override_is_anchored_at_repo_root(reload_paths):
    paths = reload_paths("quickstart/data")
    assert paths.DATA_DIR == paths.PROJECT_ROOT / "quickstart" / "data"


def test_absolute_override_used_verbatim(reload_paths, tmp_path):
    paths = reload_paths(str(tmp_path))
    assert paths.DATA_DIR == tmp_path


def test_derived_dirs_follow_the_root(reload_paths):
    paths = reload_paths("quickstart/data")
    assert paths.STEPS_DIR == paths.DATA_DIR / "steps"
    assert paths.DEBUG_DIR == paths.DATA_DIR / "debug"


def test_rel_falls_back_for_paths_outside_repo(reload_paths, tmp_path):
    paths = reload_paths(None)
    outside = tmp_path / "somewhere" / "x.jpg"
    assert paths.rel(outside) == outside  # not relative to repo -> returned as-is
