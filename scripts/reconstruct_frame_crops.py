"""Reconstruct frame_crop JPEG images from raw images and sidecar JSONs.

The frame_crop step was run on the cluster; its output sidecars (.json) have
been imported locally but the JPEG crops were not transferred. This script
reproduces every frame crop by reading crop_rect_xywh from each sidecar and
cropping directly from the corresponding raw (or rotated) source image.

For frames where no crop was found (crop_found=false), the sidecar's
crop_rect_xywh still covers the full image — that behaviour is preserved here.

Usage:
    uv run python scripts/reconstruct_frame_crops.py
"""

import argparse
import json
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).parent.parent
FRAME_CROP_DIR = PROJECT_ROOT / "data" / "steps" / "frame_crop"


def reconstruct(jpeg_quality: int, force: bool) -> None:
    sidecar_paths = sorted(FRAME_CROP_DIR.rglob("*.json"))
    print(f"Found {len(sidecar_paths)} frame_crop sidecar(s)")

    n_written = 0
    n_skipped = 0
    n_no_source = 0
    n_failed = 0

    for sidecar_path in sidecar_paths:
        data = json.loads(sidecar_path.read_text())
        crop_image_path = PROJECT_ROOT / data["crop_image"]

        if not force and crop_image_path.exists():
            n_skipped += 1
            continue

        raw_path = PROJECT_ROOT / data["source_image"]
        if not raw_path.exists():
            print(f"  SKIP (raw image missing): {raw_path.relative_to(PROJECT_ROOT)}")
            n_no_source += 1
            continue

        img = cv2.imread(str(raw_path))
        if img is None:
            print(f"  SKIP (could not read): {raw_path.relative_to(PROJECT_ROOT)}")
            n_no_source += 1
            continue

        x, y, w, h = data["crop_rect_xywh"]
        x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        crop = img[y : y + h, x : x + w]

        if crop.size == 0:
            print(f"  WARN (empty crop): {sidecar_path.relative_to(PROJECT_ROOT)}")
            n_failed += 1
            continue

        crop_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_image_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        n_written += 1

    print(
        f"\nDone — {n_written} frame crops written, "
        f"{n_skipped} skipped (already exist), "
        f"{n_no_source} skipped (missing source), "
        f"{n_failed} failed (empty crop)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=100,
        help="JPEG quality for saved crops (default: 100)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite crops that already exist",
    )
    args = parser.parse_args()
    reconstruct(args.jpeg_quality, args.force)


if __name__ == "__main__":
    main()
