import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]    # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right: smallest x-y
    rect[3] = pts[np.argmax(diff)] # bottom-left: largest x-y
    return rect


def inner_bounding_rect(quad: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned rectangle fully inside a (possibly rotated) quad.

    For an angled photo, this clips the corners slightly but avoids including
    any background outside the photo border.
    """
    tl, tr, br, bl = order_points(quad)
    x1 = int(np.ceil(max(tl[0], bl[0])))   # inner left: rightmost of left-side corners
    y1 = int(np.ceil(max(tl[1], tr[1])))   # inner top: bottommost of top-side corners
    x2 = int(np.floor(min(tr[0], br[0])))  # inner right: leftmost of right-side corners
    y2 = int(np.floor(min(bl[1], br[1])))  # inner bottom: topmost of bottom-side corners
    return x1, y1, x2 - x1, y2 - y1
