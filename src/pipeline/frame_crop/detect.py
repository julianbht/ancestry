import cv2
import numpy as np


def quad_from_mask(
    mask: np.ndarray, min_area_fraction: float, max_area_fraction: float
) -> tuple[np.ndarray | None, str]:
    """Fit a rectangle to the largest qualifying contour using minAreaRect.

    Returns (box, reason) where reason is empty on success and descriptive on failure.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no contours found"

    image_area = mask.shape[0] * mask.shape[1]
    largest_frac = cv2.contourArea(max(contours, key=cv2.contourArea)) / image_area
    if largest_frac < min_area_fraction:
        return (
            None,
            f"largest blob {largest_frac:.0%} < min {min_area_fraction:.0%}",
        )

    best_rect_frac: float | None = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) / image_area < min_area_fraction:
            break
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        rect_w, rect_h = rect[1]
        rect_frac = (rect_w * rect_h) / image_area
        if best_rect_frac is None:
            best_rect_frac = rect_frac
        if min_area_fraction <= rect_frac <= max_area_fraction:
            return cv2.boxPoints(rect).astype(np.float32), ""

    if best_rect_frac is not None:
        if best_rect_frac > max_area_fraction:
            return (
                None,
                f"best rect {best_rect_frac:.0%} > max {max_area_fraction:.0%}",
            )
        return (
            None,
            f"best rect {best_rect_frac:.0%} < min {min_area_fraction:.0%}",
        )
    return None, "no suitable contour"
