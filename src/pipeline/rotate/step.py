"""
Rotate step: fix photos with incorrect orientation.

Reads data/curated/rotate/rotations.csv (filename, apparent_rotation) and rotates
those photos into data/steps/rotate/, preserving the subdirectory structure from
data/raw/. Only files listed in the CSV are processed.

apparent_rotation is how the photo appears to a viewer (e.g. 90 = tilted right).
The step applies exif_transpose() first, then rotates by apparent_rotation CCW
to produce a correctly oriented image with no EXIF orientation tag.
"""

import csv
from pathlib import Path

from loguru import logger
from PIL import Image, ImageOps

from pipeline.shared import state as state_lib
from pipeline.shared.log import setup
from pipeline.shared.paths import CURATED_DIR, DATA_DIR, STEPS_DIR

RAW_DIR = DATA_DIR / "raw"
ROTATED_DIR = STEPS_DIR / "rotate"
ROTATIONS_CSV = CURATED_DIR / "rotate" / "rotations.csv"
STATE_FILE = state_lib.STATE_DIR / "rotate.json"


def _load_rotations() -> dict[str, int]:
    if not ROTATIONS_CSV.exists():
        return {}
    with ROTATIONS_CSV.open(newline="") as f:
        return {
            row["filename"]: int(row["apparent_rotation"])
            for row in csv.DictReader(f)
        }


def _build_filename_index() -> dict[str, Path | None]:
    """Map filename -> path for all JPEGs in RAW_DIR. Marks duplicates as None."""
    index: dict[str, Path | None] = {}
    for path in RAW_DIR.rglob("*.jpg"):
        name = path.name
        if name in index:
            logger.warning(f"Duplicate filename '{name}' in raw directory — skipping both")
            index[name] = None
        else:
            index[name] = path
    return index


def run() -> None:
    setup("rotate")

    rotations = _load_rotations()
    if not rotations:
        logger.info(f"No rotations defined in {ROTATIONS_CSV} — nothing to do")
        return

    logger.info(f"Loaded {len(rotations)} rotation(s) from {ROTATIONS_CSV}")

    filename_index = _build_filename_index()
    state = state_lib.load(STATE_FILE)

    total_rotated = 0
    total_skipped = 0
    total_failed = 0

    for filename, apparent_rotation in rotations.items():
        raw_path = filename_index.get(filename)
        if raw_path is None:
            logger.warning(f"'{filename}' not found in raw directory — skipping")
            continue

        key = raw_path.relative_to(RAW_DIR).as_posix()

        if state_lib.is_done(state, key):
            logger.debug(f"Skipping {filename} (already rotated)")
            total_skipped += 1
            continue

        out_path = ROTATED_DIR / raw_path.relative_to(RAW_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(raw_path)
            img = ImageOps.exif_transpose(img)
            img = img.rotate(apparent_rotation, expand=True)
            img.save(out_path, quality=95, subsampling=0)

            state_lib.mark_done(state, key, [str(out_path)])
            state_lib.save(state, STATE_FILE)
            logger.success(f"Rotated {filename} ({apparent_rotation} degrees CCW)")
            total_rotated += 1
        except Exception as e:
            state_lib.mark_failed(state, key, str(e))
            state_lib.save(state, STATE_FILE)
            logger.error(f"Failed to rotate {filename}: {e}")
            total_failed += 1

    logger.info(
        f"Done — {total_rotated} rotated, {total_skipped} skipped, {total_failed} failed"
    )


if __name__ == "__main__":
    run()
