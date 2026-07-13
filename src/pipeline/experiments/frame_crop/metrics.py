import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class GTItem:
    image_rel: str  # e.g. "raw/folder/image.jpg" — matches ground_truth.json's image_rel field
    quad: np.ndarray  # (4, 2) float32, pixel coordinates


def load_ground_truth(gt_path: Path) -> list[GTItem]:
    data = json.loads(gt_path.read_text())
    items = []
    for item in data["items"].values():
        quad = np.array(item["quad"], dtype=np.float32)
        items.append(GTItem(image_rel=item["image_rel"], quad=quad))
    return items


def polygon_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU between two convex quadrilaterals, both (4, 2) float32."""
    pred_r = pred.reshape(-1, 1, 2)
    gt_r = gt.reshape(-1, 1, 2)
    intersection_area, _ = cv2.intersectConvexConvex(pred_r, gt_r)
    pred_area = cv2.contourArea(pred_r)
    gt_area = cv2.contourArea(gt_r)
    union_area = pred_area + gt_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return float(intersection_area / union_area)


def detection_rate_at_iou(ious: list[float], threshold: float = 0.75) -> float:
    if not ious:
        return 0.0
    return float(sum(v >= threshold for v in ious) / len(ious))
