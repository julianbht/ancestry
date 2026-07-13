"""Path helpers for Label Studio helpers."""

from pathlib import Path
from typing import Iterable

from pipeline.shared.paths import DATA_DIR, PROJECT_ROOT


def resolve_path(path: Path) -> Path:
    """Resolve a config-relative path. Absolute paths pass through. A ``data/…``
    path resolves under the (redirectable) data root, so annotation outputs honour
    ANCESTRY_DATA_DIR (e.g. write into quickstart/data). Anything else (``config/…``)
    anchors at the repo root, keeping public config where it belongs."""
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        return DATA_DIR / Path(*parts[1:])
    return PROJECT_ROOT / path


def resolve_paths(paths: Iterable[Path]) -> list[Path]:
    return [resolve_path(path) for path in paths]


def relativize(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
