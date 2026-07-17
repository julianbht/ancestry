"""Hand-cropped portraits under data/gramps/portraits/.

These are curated by hand and are the highest-priority image for a person
everywhere they are shown (the web viewer's profile picture, the Gramps profile
thumbnail). Dropping a correctly-named file into the directory is all it takes
to give someone a portrait — no pipeline run, no annotation.

Filename convention::

    <person_id>-<anything>.<ext>     e.g. I0004-hergoss-julian.png

Only the leading Gramps person id is significant; the rest of the stem is there
for humans browsing the folder. Matching on the id rather than on the person's
name means a portrait keeps working through a rename or a spelling fix in the
family tree.
"""

from __future__ import annotations

import re
from pathlib import Path

from pipeline.shared.paths import GRAMPS_DIR

PORTRAITS_DIR = GRAMPS_DIR / "portraits"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Gramps person ids are "I" + digits. Anchored at the stem's start and followed
# by a separator (or nothing), so "I0004-hergoss-julian" matches but a stray
# "Ilse.png" does not.
_ID_PREFIX = re.compile(r"^(I\d+)(?:[-_.]|$)")


def person_id_from_filename(path: Path) -> str | None:
    """The Gramps person id a portrait filename is for, or None if it has none."""
    match = _ID_PREFIX.match(path.stem)
    return match.group(1) if match else None


def load_curated_portraits(directory: Path = PORTRAITS_DIR) -> dict[str, Path]:
    """Map person id -> hand-cropped portrait file.

    Files that don't start with a person id are ignored, so the directory can
    hold notes or work-in-progress crops without breaking the lookup. When two
    files claim the same id the alphabetically first wins — arbitrary, but
    stable across runs and machines, which matters because this feeds a
    deterministic Gramps export.
    """
    portraits: dict[str, Path] = {}
    if not directory.is_dir():
        return portraits
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        person_id = person_id_from_filename(path)
        if person_id is not None:
            portraits.setdefault(person_id, path)
    return portraits
