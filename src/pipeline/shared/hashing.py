"""Content hashing for provenance.

A derived artifact (e.g. a face embedding) is a pure function of its input
bytes. Recording the SHA-256 of those bytes lets a later reader detect when an
upstream step has been re-run and changed the input — turning a silently-stale
artifact into a detectable mismatch, rather than relying on a path that stays
the same across re-runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pipeline.shared.paths import PROJECT_ROOT


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes (streamed, so large files are fine)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_crop_image(crop_image: str) -> str | None:
    """SHA-256 of a face crop referenced by its project-relative path string (the
    `crop_image` value stored in face_crop / face_recognition sidecars), or None
    if the file is missing or unreadable."""
    try:
        return sha256_file(PROJECT_ROOT / crop_image)
    except OSError:
        return None
