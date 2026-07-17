"""Upright, EXIF-baked copies of every image the Gramps export references.

The raw smartphone photos store their pixels sideways with an EXIF orientation
tag, while the annotations (bbox_xywh / image_size) are in the EXIF-corrected
*display* space. Gramps renders the raw pixel buffer and ignores the orientation
tag, so pointing a media object straight at the raw file shows it sideways and
the face regions no longer line up.

Every media object therefore points at a copy under data/gramps/media/ with the
orientation applied to the pixels and the tag stripped, so pixels and regions
share one coordinate space. Baking is a plain re-encode for images that carry no
orientation tag (steps/rotate/ output, hand-cropped portraits), which keeps the
invariant simple: media src always resolves under MEDIA_DIR, never to the
original.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from pipeline.shared.paths import GRAMPS_DIR

MEDIA_DIR = GRAMPS_DIR / "media"

# Existing Gramps handles are "_" + 28 hex chars; ours match that shape so they
# are indistinguishable from Gramps' own in the file.
_HANDLE_HEX_LEN = 28


def media_handle(media_key: str) -> str:
    """Deterministic Gramps handle for a media key.

    Same key -> same handle on every run and every machine, so re-merging a
    fresh export never produces a different tree for the same inputs.
    """
    digest = hashlib.sha1(media_key.encode("utf-8")).hexdigest()
    return "_" + digest[:_HANDLE_HEX_LEN]


def media_path(media_key: str) -> Path:
    """Where the baked copy for a media key lives on disk."""
    return MEDIA_DIR / media_key


def mime_type(path: Path) -> str:
    """The mime the media <object> declares. Gramps needs it to pick a viewer."""
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def bake_upright(source: Path, dest: Path, jpeg_quality: int) -> None:
    """Write an upright, EXIF-free copy of `source` to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        upright = ImageOps.exif_transpose(img)  # bake orientation into pixels
        # quality/subsampling are JPEG-only; PIL ignores them for PNG, which is
        # lossless anyway, so one call covers both kinds of input.
        upright.save(dest, quality=jpeg_quality, subsampling=0)
