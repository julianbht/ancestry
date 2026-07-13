"""Download (ingest) step: ensure raw photos are present in data/raw/.

The step is a thin selector over pluggable, pull-only PhotoSources (see
pipeline/download/sources/). Which one runs is set by `source` in
config/download/step.<env>.yaml: `nextcloud` (legacy WebDAV+zip), `r2` (pull from
Cloudflare R2), or `local` (already present). Getting raw *out* to R2 is the
backup tool's job (scripts/sync_private.py), not this step's.
"""

from loguru import logger

from pipeline.download.config import DownloadConfig, load_shares
from pipeline.download.sources.base import PhotoSource
from pipeline.download.sources.local import LocalSource
from pipeline.download.sources.nextcloud import NextcloudSource
from pipeline.download.sources.r2 import R2Source
from pipeline.shared.config import load as load_config
from pipeline.shared.log import setup


def _build_source(config: DownloadConfig) -> PhotoSource:
    if config.source == "nextcloud":
        return NextcloudSource(load_shares())
    if config.source == "r2":
        return R2Source()
    return LocalSource()  # "local"


def run() -> None:
    setup("download")

    try:
        config = load_config("download", DownloadConfig)
        source = _build_source(config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return

    logger.info(
        f"Ingest source: {config.source} — "
        f"max_files={config.max_files_to_download or 'unlimited'}, "
        f"ignore_state={config.ignore_state}"
    )
    source.fetch(config.max_files_to_download, config.ignore_state)


if __name__ == "__main__":
    run()
