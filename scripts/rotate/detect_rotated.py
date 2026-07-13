"""
Detect photos with incorrect orientation using EXIF tags and write data/curated/rotate/rotations.csv.

Saves before/after previews to data/rotation_preview/ for visual confirmation:
  before/ - originals as they appear (sideways/upside-down)
  after/  - corrected versions (should look upright)

EXIF heuristic (specific to the phone used for these scans):
  EXIF=3 -> appears tilted 90 degrees to the right -> apparent_rotation=90
  EXIF=8 -> appears upside down                    -> apparent_rotation=180

The generated CSV can be edited manually to add photos from other phones
that cannot be detected automatically via EXIF.
"""

import csv
import shutil
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_FILE = PROJECT_ROOT / "data" / "curated" / "rotate" / "rotations.csv"
PREVIEW_BEFORE = PROJECT_ROOT / "data" / "debug" / "rotation_preview" / "before"
PREVIEW_AFTER = PROJECT_ROOT / "data" / "debug" / "rotation_preview" / "after"

# EXIF orientation value -> how the image appears to a human viewer (degrees tilted)
EXIF_TO_APPARENT: dict[int, int] = {
    3: 90,   # appears tilted 90 degrees to the right
    8: 180,  # appears upside down
}


def main() -> None:
    detected: list[tuple[Path, int]] = []

    for path in sorted(RAW_DIR.rglob("*.jpg")):
        try:
            img = Image.open(path)
            orientation = img.getexif().get(274, 1)
            if orientation in EXIF_TO_APPARENT:
                detected.append((path, EXIF_TO_APPARENT[orientation]))
        except Exception as e:
            print(f"ERROR: {path.name}: {e}")

    print(f"Detected {len(detected)} photos needing rotation")

    # Load existing CSV so manual additions are preserved
    existing: dict[str, int] = {}
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open(newline="") as f:
            for row in csv.DictReader(f):
                existing[row["filename"]] = int(row["apparent_rotation"])
        print(f"Loaded {len(existing)} existing entries from {CONFIG_FILE}")

    added = 0
    for path, apparent in detected:
        if path.name not in existing:
            existing[path.name] = apparent
            added += 1

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "apparent_rotation"])
        for filename, apparent in sorted(existing.items()):
            writer.writerow([filename, apparent])

    print(f"Wrote {len(existing)} entries to {CONFIG_FILE} ({added} new)")

    # Save before/after previews for visual confirmation
    PREVIEW_BEFORE.mkdir(parents=True, exist_ok=True)
    PREVIEW_AFTER.mkdir(parents=True, exist_ok=True)

    for path, apparent in detected:
        shutil.copy2(path, PREVIEW_BEFORE / path.name)

        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.rotate(apparent, expand=True)
        img.save(PREVIEW_AFTER / path.name, quality=95, subsampling=0)

    print(f"\nPreviews saved to data/tmp/rotation_preview/")
    print("  before/ - originals (sideways/upside-down)")
    print("  after/  - corrected (should look upright)")


if __name__ == "__main__":
    main()
