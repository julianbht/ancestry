"""Resolve a homepage portrait image for a person.

Resolution order (first hit wins):
  1. A hand-cropped portrait from data/gramps/portraits/, matched by person id.
     Always wins: dropping a correctly-named file in there is how you fix
     someone's portrait, and it must not be second-guessed by a cached crop we
     made earlier. The Gramps step (pipeline.gramps) applies the same priority,
     so the viewer and the family tree agree on everyone's face.
  2. A cached crop we made earlier.
  3. The human `is_portrait` face from ground truth, cropped from its source
     photo with a little margin and cached.
  4. None → the template renders an initials placeholder.

Cropping/caching is the only I/O here; the repository supplies the
`PortraitSource` so this service stays unaware of how labels are stored.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from pipeline.gramps.portraits import load_curated_portraits
from pipeline.shared.paths import DATA_DIR

from web.models import Person
from web.repository import PortraitSource

CACHE_DIR = DATA_DIR / "web_cache" / "portraits"

# Fraction of the face box added on every side so the portrait isn't cropped to
# the eyebrows. The ground-truth boxes are tight to the face.
_MARGIN = 0.35


class PortraitService:
    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._curated = load_curated_portraits()

    def resolve(self, person: Person, source: PortraitSource | None) -> Path | None:
        """Return a servable image path, or None for the placeholder."""
        curated = self._curated.get(person.id)
        if curated is not None:
            return curated
        cached = CACHE_DIR / f"{person.id}.jpg"
        if cached.exists():
            return cached
        if source is not None:
            return self._crop_and_cache(source, cached)
        return None

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
