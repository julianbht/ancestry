"""Reconstruct face_crop JPEG images from raw images and sidecar JSONs.

The face_crop step was run on the cluster; its output sidecars (faces.json)
have been imported locally but the JPEG crops were not transferred. This script
reproduces every face crop by reading box_xyxy_source from each sidecar and
cropping directly from the corresponding raw (or rotated) source image — no SAM
or GPU required.

Box geometry follows the same margin logic as src/pipeline/face_crop/cropping.py.

Usage:
    uv run python scripts/reconstruct_face_crops.py [--margin-frac 0.15]
"""

import argparse
import json
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).parent.parent
FACE_CROP_DIR = PROJECT_ROOT / "data" / "steps" / "face_crop"
FRAME_CROP_DIR = PROJECT_ROOT / "data" / "steps" / "frame_crop"


def _crop_with_margin(
    img: "cv2.typing.MatLike",
    box: tuple[float, float, float, float],
    margin_frac: float,
) -> "cv2.typing.MatLike | None":
    """Crop box from img with margin, clamped to image bounds.

    Replicates the logic in src/pipeline/face_crop/cropping.py so the output
    matches what the step would have written.
    """
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    mx = (x1 - x0) * margin_frac
    my = (y1 - y0) * margin_frac
    cx0 = max(0, int(round(x0 - mx)))
    cy0 = max(0, int(round(y0 - my)))
    cx1 = min(w, int(round(x1 + mx)))
    cy1 = min(h, int(round(y1 + my)))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return img[cy0:cy1, cx0:cx1]


def reconstruct(margin_frac: float, jpeg_quality: int, force: bool) -> None:
    sidecar_paths = sorted(FACE_CROP_DIR.rglob("faces.json"))
    print(f"Found {len(sidecar_paths)} face_crop sidecar(s)")

    n_frames_done = 0
    n_frames_skipped = 0
    n_crops_written = 0
    n_crops_failed = 0
    n_no_source = 0

    for sidecar_path in sidecar_paths:
        data = json.loads(sidecar_path.read_text())
        faces = data.get("faces", [])

        if not faces:
            n_frames_skipped += 1
            continue

        # Check if all crop images already exist (skip unless force)
        crop_paths = [PROJECT_ROOT / f["crop_image"] for f in faces]
        if not force and all(p.exists() for p in crop_paths):
            n_frames_skipped += 1
            continue

        # Resolve the source (raw/rotated) image via the frame_crop sidecar.
        # faces.json source_image → frame_crop/<folder>/<stem>.jpg
        # frame_crop sidecar     → raw source_image
        frame_crop_img_path = PROJECT_ROOT / data["source_image"]
        frame_crop_json_path = frame_crop_img_path.with_suffix(".json")

        if not frame_crop_json_path.exists():
            print(f"  SKIP (no frame_crop sidecar): {frame_crop_json_path.relative_to(PROJECT_ROOT)}")
            n_no_source += 1
            continue

        frame_sidecar = json.loads(frame_crop_json_path.read_text())
        raw_image_path = PROJECT_ROOT / frame_sidecar["source_image"]

        if not raw_image_path.exists():
            print(f"  SKIP (raw image missing): {raw_image_path.relative_to(PROJECT_ROOT)}")
            n_no_source += 1
            continue

        img = cv2.imread(str(raw_image_path))
        if img is None:
            print(f"  SKIP (could not read): {raw_image_path.relative_to(PROJECT_ROOT)}")
            n_no_source += 1
            continue

        frame_ok = True
        for face, crop_path in zip(faces, crop_paths):
            if not force and crop_path.exists():
                continue

            box_source = face.get("box_xyxy_source")
            if box_source is None:
                print(f"  WARN (no box_xyxy_source): {sidecar_path.relative_to(PROJECT_ROOT)} face {face['index']}")
                n_crops_failed += 1
                frame_ok = False
                continue

            crop = _crop_with_margin(img, tuple(box_source), margin_frac)
            if crop is None:
                print(f"  WARN (empty crop): {sidecar_path.relative_to(PROJECT_ROOT)} face {face['index']}")
                n_crops_failed += 1
                frame_ok = False
                continue

            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            n_crops_written += 1

        if frame_ok:
            n_frames_done += 1

    print(
        f"\nDone — {n_frames_done} frames reconstructed, "
        f"{n_crops_written} crops written, "
        f"{n_crops_failed} crop failures, "
        f"{n_no_source} frames skipped (missing source), "
        f"{n_frames_skipped} frames skipped (already complete)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--margin-frac",
        type=float,
        default=0.15,
        help="Margin fraction applied to each face box (default: 0.15, matching prod config)",
    )
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
    reconstruct(args.margin_frac, args.jpeg_quality, args.force)


if __name__ == "__main__":
    main()
