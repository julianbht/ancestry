"""Fixtures for the gramps step's tests.

The step resolves its paths from module-level constants derived from DATA_DIR at
import time, so the fixture repoints those constants rather than the env var —
re-importing the modules mid-session would be far more fragile. The tree itself
is built by tree_fixture.build_tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gramps.tree_fixture import Tree, build_tree


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tree:
    built = build_tree(tmp_path)

    monkeypatch.setattr("pipeline.gramps.plan.DATA_DIR", built.root)
    monkeypatch.setattr("pipeline.gramps.media.MEDIA_DIR", built.media_dir)
    monkeypatch.setattr("pipeline.gramps.step.MEDIA_DIR", built.media_dir)
    monkeypatch.setattr("pipeline.gramps.step.EXPORT_FILE", built.export)
    monkeypatch.setattr("pipeline.gramps.step.AUGMENTED_FILE", built.augmented)
    monkeypatch.setattr("pipeline.gramps.step.PORTRAITS_DIR", built.portraits_dir)
    monkeypatch.setattr("pipeline.gramps.step.GROUND_TRUTH_FILE", built.ground_truth)
    monkeypatch.setattr("pipeline.gramps.step.STATE_FILE", built.state_file)

    return built
