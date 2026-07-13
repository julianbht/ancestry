"""Config schemas for the download (ingest) step."""

import json
from typing import Annotated, Literal

from pydantic import Field, HttpUrl

from pipeline.shared.config import StrictConfig, _validate
from pipeline.shared.paths import CURATED_DIR


class NextcloudShare(StrictConfig):
    token: str
    description: str


class SharesConfig(StrictConfig):
    nextcloud_base_url: HttpUrl
    shares: Annotated[list[NextcloudShare], Field(min_length=1)]


class DownloadConfig(StrictConfig):
    # Where raw photos come from:
    #   nextcloud — legacy WebDAV shares, fetched as encrypted zips and extracted
    #   r2        — pull raw photos from Cloudflare R2 (the canonical store)
    #   local     — already present in data/raw (dummy data / drop-locally workflow)
    source: Literal["nextcloud", "r2", "local"]
    # Maximum number of files to fetch in a single run (null = no limit).
    # Useful for smoke testing.
    max_files_to_download: Annotated[int | None, Field(ge=1)]
    # Re-fetch even files already fetched (ignore recorded state / existing local files).
    ignore_state: bool


def load_shares() -> SharesConfig:
    """Load and validate data/curated/download/shares.json (only needed for source=nextcloud).

    Private (the share tokens are effectively access credentials), so it lives
    under the data root, not in the public config/ tree — like the download
    step's step.yaml, which stays public."""
    shares_file = CURATED_DIR / "download" / "shares.json"
    if not shares_file.exists():
        raise FileNotFoundError(f"Shares config not found: {shares_file}")
    raw = json.loads(shares_file.read_text())
    return _validate(raw, SharesConfig, shares_file)
