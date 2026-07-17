"""Shared utilities for reading and writing pipeline state files."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.shared.paths import DATA_DIR

STATE_DIR = DATA_DIR / "state"
VERSION = 1

State = dict[str, Any]


def load(state_file: Path) -> State:
    """Load a state file, returning an empty state dict if it doesn't exist."""
    if state_file.exists():
        with state_file.open() as f:
            return json.load(f)  # type: ignore[no-any-return]
    return {"version": VERSION, "processed": {}}


def save(state: State, state_file: Path) -> None:
    # The file's own parent, not STATE_DIR: they're the same for every step, but
    # honouring the argument means a caller pointing elsewhere (a test) works.
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w") as f:
        json.dump(state, f, indent=2)


def is_done(state: State, key: str) -> bool:
    return state["processed"].get(key, {}).get("status") == "done"


def mark_done(state: State, key: str, output: list[str]) -> None:
    state["processed"][key] = {
        "status": "done",
        "output": output,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


def mark_failed(state: State, key: str, error: str) -> None:
    state["processed"][key] = {
        "status": "failed",
        "output": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
