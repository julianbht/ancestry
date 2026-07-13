"""Shared axis-aligned bounding-box geometry helpers."""

import numpy as np


def box_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap coefficient of two axis-aligned boxes (both [x0, y0, x1, y1]):
    intersection over the area of the *smaller* box.

    Use this instead of plain IoU whenever one box is expected to sit fully (or
    almost fully) inside the other — e.g. a tight face-detection box matched
    against a ground-truth box that was annotated with extra hair margin. IoU
    would be dragged down by the larger box's contribution to the union and
    unfairly penalize an otherwise-correct match; overlap stays ~1.0 as long as
    the smaller box is contained in the larger one, regardless of how much
    bigger the larger box is.
    """
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return float(inter / smaller) if smaller > 0 else 0.0
