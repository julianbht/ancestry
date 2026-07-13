"""Metrics for validating face_crop detections against the face_annotation
ground truth.

Predicted face boxes are matched one-to-one against the GT boxes, but the match
score is the **overlap coefficient** (Szymkiewicz–Simpson):

    overlap = intersection_area / min(area_pred, area_gt)

not IoU. The GT boxes were annotated to include hair, so they are systematically
taller than SAM's face boxes. With IoU the larger GT box inflates the union and
drags a perfectly good detection below threshold; the GT face then looks
"missed" even though it was found. Overlap divides by the *smaller* box instead —
since SAM's face box sits inside the hair-inclusive GT box, a correct detection
scores ~1.0 regardless of how much hair the GT added, while a spurious or
duplicate box still scores low. So a high threshold (~0.7) stays meaningful.

After matching: unmatched GT faces are genuinely missed; unmatched predictions
are extras (often real faces the GT annotation skipped).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.shared.boxes import box_overlap


@dataclass
class GTFaceImage:
    image_rel: str  # e.g. "raw/folder/image.jpg" — matches ground_truth.json's image_rel
    image_size: tuple[int, int]  # (width, height) in pixels
    boxes_xyxy: np.ndarray  # (N, 4) float, [x0, y0, x1, y1]


def load_ground_truth(gt_path: Path) -> list[GTFaceImage]:
    data = json.loads(gt_path.read_text())
    items: list[GTFaceImage] = []
    for item in data["items"].values():
        boxes = []
        for face in item["faces"]:
            x, y, w, h = face["bbox_xywh"]
            boxes.append([x, y, x + w, y + h])
        arr = np.array(boxes, dtype=float).reshape(-1, 4)
        size = tuple(item["image_size"])  # type: ignore[assignment]
        items.append(GTFaceImage(item["image_rel"], size, arr))
    return items


@dataclass
class MatchResult:
    """Outcome of matching predicted boxes to GT boxes for one image."""

    n_gt: int
    n_pred: int
    matched_overlaps: list[float]  # overlap score of each matched (pred, gt) pair

    @property
    def n_matched(self) -> int:
        return len(self.matched_overlaps)

    @property
    def n_missed(self) -> int:  # GT faces with no matching prediction
        return self.n_gt - self.n_matched

    @property
    def n_extra(self) -> int:  # predictions matching no GT face (often real, GT-skipped)
        return self.n_pred - self.n_matched

    @property
    def mean_overlap(self) -> float:
        return float(np.mean(self.matched_overlaps)) if self.matched_overlaps else 0.0

    @property
    def all_found(self) -> bool:  # every GT face matched (extras allowed)
        return self.n_missed == 0


def match_boxes(pred: np.ndarray, gt: np.ndarray, threshold: float) -> MatchResult:
    """Greedy one-to-one matching by overlap coefficient: repeatedly take the
    highest-overlap (pred, gt) pair at or above the threshold until none remain."""
    pred = pred.reshape(-1, 4)
    gt = gt.reshape(-1, 4)

    pairs = sorted(
        (
            (box_overlap(pred[i], gt[j]), i, j)
            for i in range(len(pred))
            for j in range(len(gt))
        ),
        reverse=True,
    )

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matched: list[float] = []
    for score, i, j in pairs:
        if score < threshold:
            break
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matched.append(score)

    return MatchResult(n_gt=len(gt), n_pred=len(pred), matched_overlaps=matched)


@dataclass
class Summary:
    """Corpus-level rollup of per-image MatchResults."""

    n_images: int
    total_gt: int
    total_pred: int
    total_matched: int
    total_missed: int
    total_extra: int
    n_images_all_found: int  # images where every GT face was matched
    mean_overlap: float

    @property
    def recall(self) -> float:  # fraction of GT faces found
        return self.total_matched / self.total_gt if self.total_gt else 0.0

    @property
    def all_found_rate(self) -> float:
        return self.n_images_all_found / self.n_images if self.n_images else 0.0


def summarize(results: list[MatchResult]) -> Summary:
    all_overlaps = [o for r in results for o in r.matched_overlaps]
    return Summary(
        n_images=len(results),
        total_gt=sum(r.n_gt for r in results),
        total_pred=sum(r.n_pred for r in results),
        total_matched=sum(r.n_matched for r in results),
        total_missed=sum(r.n_missed for r in results),
        total_extra=sum(r.n_extra for r in results),
        n_images_all_found=sum(r.all_found for r in results),
        mean_overlap=float(np.mean(all_overlaps)) if all_overlaps else 0.0,
    )
