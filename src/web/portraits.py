"""Resolve a homepage portrait image for a person.

Resolution order (first hit wins):
  1. A cached crop we made earlier.
  2. The human `is_portrait` face from ground truth, cropped from its source
     photo with a little margin and cached.
  3. A curated portrait from data/gramps/portraits/ matched by name.
  4. None → the template renders an initials placeholder.

Cropping/caching is the only I/O here; the repository supplies the
`PortraitSource` so this service stays unaware of how labels are stored.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from pipeline.shared.paths import DATA_DIR, GRAMPS_DIR

from web.models import Person
from web.repository import PortraitSource

GRAMPS_PORTRAITS_DIR = GRAMPS_DIR / "portraits"
CACHE_DIR = DATA_DIR / "web_cache" / "portraits"

# Fraction of the face box added on every side so the portrait isn't cropped to
# the eyebrows. The ground-truth boxes are tight to the face.
_MARGIN = 0.35


def _fold(text: str) -> str:
    """Normalise a name fragment for filename matching (umlauts, case, punctuation)."""
    table = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}
    text = text.lower()
    for src, dst in table.items():
        text = text.replace(src, dst)
    return "".join(ch for ch in text if ch.isalnum())


class PortraitService:
    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._curated = self._index_curated()

    @staticmethod
    def _index_curated() -> dict[str, Path]:
        """Map folded 'surname+given' -> curated portrait file."""
        index: dict[str, Path] = {}
        if not GRAMPS_PORTRAITS_DIR.is_dir():
            return index
        for path in GRAMPS_PORTRAITS_DIR.iterdir():
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            # filenames look like 'boldt-frieda.png' -> surname 'boldt', given 'frieda'
            key = _fold(path.stem.replace("-", ""))
            index.setdefault(key, path)
        return index

    def resolve(self, person: Person, source: PortraitSource | None) -> Path | None:
        """Return a servable image path, or None for the placeholder."""
        cached = CACHE_DIR / f"{person.id}.jpg"
        if cached.exists():
            return cached
        if source is not None:
            crop = self._crop_and_cache(source, cached)
            if crop is not None:
                return crop
        return self._curated_for(person)

    def _curated_for(self, person: Person) -> Path | None:
        given_first = person.given.split()[0] if person.given else ""
        key = _fold(person.surname + given_first)
        return self._curated.get(key)

    @staticmethod
    def _crop_and_cache(source: PortraitSource, dest: Path) -> Path | None:
        if not source.image_path.exists():
            return None
        x, y, w, h = source.bbox_xywh
        mx, my = w * _MARGIN, h * _MARGIN
        with Image.open(source.image_path) as raw:
            # frame_crop ran on cv2.imread pixels, which honour EXIF orientation;
            # the ground-truth bbox is in that upright space. PIL doesn't apply
            # EXIF on open, so we transpose to match before cropping — otherwise
            # raw smartphone photos come out sideways.
            img = ImageOps.exif_transpose(raw)
            left = max(0, int(x - mx))
            top = max(0, int(y - my))
            right = min(img.width, int(x + w + mx))
            bottom = min(img.height, int(y + h + my))
            crop = img.crop((left, top, right, bottom)).convert("RGB")
            crop.save(dest, "JPEG", quality=88)
        return dest
