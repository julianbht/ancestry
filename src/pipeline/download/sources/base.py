"""The PhotoSource interface: a pluggable, pull-only way to get raw photos into
`data/raw/`.

A source only ever *brings raw photos in* — it never uploads and never touches
derived artifacts or curated data. That line is what keeps ingest (this step)
separate from backup (`scripts/sync_private.py`). To add a new source, implement
this Protocol and add a branch to `pipeline.download.step._build_source`.
"""

from __future__ import annotations

from typing import Protocol

from pipeline.shared.paths import DATA_DIR

RAW_DIR = DATA_DIR / "raw"


class PhotoSource(Protocol):
    def fetch(self, max_files: int | None, ignore_state: bool) -> None:
        """Ensure raw photos are present in RAW_DIR. Idempotent: re-running only
        fetches what's missing.

        Args:
            max_files: cap on files fetched this run (None = no cap); for smoke tests.
            ignore_state: re-fetch even files already fetched / already on disk.
        """
        ...
