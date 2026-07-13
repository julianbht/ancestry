"""Shared logging setup for all pipeline steps.

Each pipeline step calls setup() once at startup, which configures loguru
with a colored stderr sink and a timestamped log file for that run.

Usage:
    from pipeline.shared.log import setup, logger

    setup("download")
    logger.info("Starting download step")
"""

import sys
from datetime import datetime

from loguru import logger

from pipeline.shared.paths import DATA_DIR

LOG_DIR = DATA_DIR / "logs"


def setup(step: str) -> None:
    """Configure loguru for a pipeline run.

    Replaces any existing handlers with a colored stderr sink and a
    timestamped file sink. Each call creates a new log file, so every
    run (and every step in a full pipeline run) gets its own file.

    Args:
        step: Name of the pipeline step (e.g. "download", "frame_crop").
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"{timestamp}_{step}.log"

    logger.remove()
    logger.add(sys.stderr, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    logger.add(log_file, colorize=False, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

    # Banner so each step stands out clearly when steps run back-to-back in a
    # full pipeline run (uv run python -m pipeline).
    logger.info("=" * 60)
    logger.info(f" STEP: {step} ".center(60, "="))
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
