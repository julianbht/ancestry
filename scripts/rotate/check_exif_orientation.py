"""Scan all raw photos and report EXIF orientation tags that are not 'normal' (1)."""

from pathlib import Path
from PIL import Image

ORIENTATION_LABELS = {
    1: "normal",
    2: "mirrored horizontal",
    3: "rotated 180°",
    4: "mirrored vertical",
    5: "mirrored horizontal + rotated 90° CCW",
    6: "rotated 90° CW",
    7: "mirrored horizontal + rotated 90° CW",
    8: "rotated 90° CCW",
}

raw_dir = Path(__file__).parent / "data" / "raw"
jpgs = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.JPG")) + list(raw_dir.rglob("*.jpeg"))

print(f"Found {len(jpgs)} JPEG files\n")

non_normal = []
for path in sorted(jpgs):
    try:
        img = Image.open(path)
        exif = img.getexif()
        orientation = exif.get(274, 1)
        if orientation != 1:
            label = ORIENTATION_LABELS.get(orientation, f"unknown ({orientation})")
            non_normal.append((path, orientation, label))
            print(f"{path.relative_to(raw_dir)}  ->  {label}")
    except Exception as e:
        print(f"ERROR reading {path.name}: {e}")

print(f"\n{len(non_normal)} files with non-normal orientation out of {len(jpgs)} total.")
