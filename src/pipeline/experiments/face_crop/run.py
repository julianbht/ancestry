"""Validate the face_crop detector against the face_annotation ground truth.

This is a single evaluation pass, not a hyperparameter search: it runs the
configured SAM face detector over every ground-truth image and reports how many
GT faces it recovers and how well its boxes land on them.

The detector runs on the raw/rotated images — the same coordinate space the GT
was annotated in — so predictions and labels compare directly with no quad
mapping. Boxes are matched by overlap coefficient rather than IoU so the GT's
extra hair margin doesn't turn correct detections into false misses
(see metrics.py).

Usage:
    uv run ancestry-face-crop-eval                 # all GT images, default settings
    uv run ancestry-face-crop-eval --limit 10      # quick smoke check
    uv run ancestry-face-crop-eval --overlays all  # dump GT-vs-pred overlays
"""

import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np
from loguru import logger

from pipeline.experiments.face_crop.metrics import (
    Summary,
    load_ground_truth,
    match_boxes,
    summarize,
)
from pipeline.face_crop.config import FaceCropConfig
from pipeline.face_crop.detector import SamFaceDetector
from pipeline.shared.config import load as load_config
from pipeline.shared.log import setup
from pipeline.shared.paths import CURATED_DIR, DATA_DIR, DEBUG_DIR, rel as project_rel

GT_PATH = CURATED_DIR / "face_annotation" / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"
OVERLAY_DIR = DEBUG_DIR / "face_crop_eval"

GT_COLOR = (0, 0, 255)    # red   (BGR) — ground-truth boxes (include hair)
PRED_COLOR = (0, 255, 0)  # green (BGR) — detector boxes


def _resolve_input(image_rel: str) -> Path:
    """The GT's image_rel is relative to data/ and already names the exact image
    the boxes were annotated on (e.g. 'raw/...' or 'steps/rotate/...'), so resolve
    it directly — that keeps predictions in the GT's coordinate space."""
    return DATA_DIR / image_rel


def _rel_key(image_rel: str) -> Path:
    """Folder/file portion of an image_rel, dropping the 'raw' / 'steps/rotate'
    prefix — used to name debug overlay outputs."""
    return Path(*Path(image_rel).parts[-2:])


def _detect_boxes(
    detector: SamFaceDetector, img: np.ndarray, min_area_fraction: float
) -> np.ndarray:
    """Run the detector and return predicted boxes as (N, 4) xyxy, applying the
    same min-area filter the step uses."""
    h, w = img.shape[:2]
    min_area = min_area_fraction * h * w
    boxes = [f.box_xyxy for f in detector.detect(img) if f.area >= min_area]
    return np.array(boxes, dtype=float).reshape(-1, 4)


def _save_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    overlay = img.copy()
    thickness = max(2, int(img.shape[1] / 600))
    for box in gt.reshape(-1, 4):
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), GT_COLOR, thickness)
    for box in pred.reshape(-1, 4):
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), PRED_COLOR, thickness)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _log_summary(s: Summary) -> None:
    logger.info(
        f"Recall: {s.recall:.1%} — found {s.total_matched}/{s.total_gt} GT faces "
        f"({s.total_missed} missed)"
    )
    logger.info(
        f"Images with every GT face found: {s.all_found_rate:.1%} "
        f"({s.n_images_all_found}/{s.n_images})"
    )
    logger.info(
        f"Extra detections (not in GT, often real faces it skipped): {s.total_extra} "
        f"across {s.total_pred} predictions"
    )
    logger.info(f"Mean overlap of matched faces: {s.mean_overlap:.3f}")


@click.command()
@click.option("--limit", type=int, default=None, help="Only evaluate the first N GT images.")
@click.option(
    "--overlap-threshold",
    default=0.7,
    show_default=True,
    help="Minimum overlap coefficient to count a prediction as matching a GT face.",
)
@click.option(
    "--overlays",
    type=click.Choice(["none", "missed", "all"]),
    default="missed",
    show_default=True,
    help="Save GT(red)-vs-pred(green) overlays to data/debug/face_crop_eval/. "
    "'missed' = only images where a GT face was not found.",
)
def main(limit: int | None, overlap_threshold: float, overlays: str) -> None:
    setup("face_crop_eval")

    try:
        config: FaceCropConfig = load_config("face_crop", FaceCropConfig)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    gt_items = load_ground_truth(GT_PATH)
    if limit is not None:
        gt_items = gt_items[:limit]
    logger.info(
        f"Evaluating {len(gt_items)} GT image(s) with prompt={config.sam.prompt!r}, "
        f"score_threshold={config.sam.score_threshold}, overlap_threshold={overlap_threshold}"
    )

    # Clear overlays from any previous run so the directory only ever reflects
    # the current eval (stale files would otherwise accumulate across runs).
    if overlays != "none" and OVERLAY_DIR.exists():
        shutil.rmtree(OVERLAY_DIR)

    detector = SamFaceDetector(config.sam)

    results = []
    rows: list[dict] = []
    for item in gt_items:
        path = _resolve_input(item.image_rel)
        img = cv2.imread(str(path))
        if img is None:
            logger.error(f"Could not read image: {path}")
            sys.exit(1)

        pred = _detect_boxes(detector, img, config.min_area_fraction)
        result = match_boxes(pred, item.boxes_xyxy, overlap_threshold)
        results.append(result)

        rows.append(
            {
                "image_rel": item.image_rel,
                "n_gt": result.n_gt,
                "n_pred": result.n_pred,
                "n_matched": result.n_matched,
                "n_missed": result.n_missed,
                "n_extra": result.n_extra,
                "mean_overlap": round(result.mean_overlap, 4),
            }
        )

        overlay_note = ""
        if overlays == "all" or (overlays == "missed" and not result.all_found):
            overlay_path = OVERLAY_DIR / _rel_key(item.image_rel)
            _save_overlay(img, item.boxes_xyxy, pred, overlay_path)
            overlay_note = f" -> {project_rel(overlay_path)}"

        flag = "" if result.all_found else f"  <-- MISSED {result.n_missed}"
        extra = f" extra={result.n_extra}" if result.n_extra else ""
        logger.info(
            f"{project_rel(path)}: gt={result.n_gt} pred={result.n_pred} "
            f"matched={result.n_matched}{extra} overlap={result.mean_overlap:.2f}{flag}{overlay_note}"
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = RESULTS_DIR / f"face-crop-eval-{timestamp}.csv"
    _write_csv(rows, csv_path)

    logger.info("--- Summary ---")
    _log_summary(summarize(results))
    logger.info(f"Per-image results written to {csv_path}")


if __name__ == "__main__":
    main()
