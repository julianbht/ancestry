"""Local photo source: photos are already present in `data/raw/`.

The no-fetch case — used for the committed dummy/quickstart tree and for the
"drop new jpgs in locally, then push to R2" workflow. It only reports what's
already there so a run has a clear starting point.
"""

from __future__ import annotations

from loguru import logger

from pipeline.download.sources.base import RAW_DIR
from pipeline.shared.paths import rel


class LocalSource:
    def fetch(self, max_files: int | None, ignore_state: bool) -> None:
        count = (
            sum(1 for p in RAW_DIR.rglob("*") if p.suffix.lower() == ".jpg")
            if RAW_DIR.exists()
            else 0
        )
        logger.info(
            f"Done — local: no fetch; {count} photo(s) already present in {rel(RAW_DIR)}"
        )
