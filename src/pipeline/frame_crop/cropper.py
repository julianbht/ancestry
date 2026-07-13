from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from pipeline.frame_crop.config import FrameCropConfig
from pipeline.frame_crop.geometry import inner_bounding_rect

# Returns the cropped image plus the resolved (x, y, w, h) rect it was cropped
# from, in the source image's own pixel coordinates — callers persist this so
# downstream steps can map their own local detections back into that space
# without recomputing it from the quad and current config.
CropFn = Callable[[np.ndarray, np.ndarray, FrameCropConfig], tuple[np.ndarray, tuple[int, int, int, int]]]


@dataclass(frozen=True)
class Cropper:
    name: str
    apply: CropFn


def _expand_rect_with_margin(
    x: int, y: int, w: int, h: int, margin_px: int, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    x1 = max(0, x - margin_px)
    y1 = max(0, y - margin_px)
    x2 = min(image_w, x + w + margin_px)
    y2 = min(image_h, y + h + margin_px)
    return x1, y1, x2 - x1, y2 - y1


def _crop_inner_rect(
    img: np.ndarray, quad: np.ndarray, config: FrameCropConfig
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = inner_bounding_rect(quad)
    x, y, w, h = _expand_rect_with_margin(
        x, y, w, h, config.margin_px, img.shape[1], img.shape[0]
    )
    if w <= 0 or h <= 0:
        raise ValueError("Inner bounding rect is empty")
    return img[y:y+h, x:x+w], (x, y, w, h)


def _crop_outer_rect(
    img: np.ndarray, quad: np.ndarray, config: FrameCropConfig
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    x, y, w, h = _expand_rect_with_margin(
        x, y, w, h, config.margin_px, img.shape[1], img.shape[0]
    )
    if w <= 0 or h <= 0:
        raise ValueError("Bounding rect is empty after clamping to image bounds")
    return img[y:y+h, x:x+w], (x, y, w, h)


def select_cropper(config: FrameCropConfig) -> Cropper:
    if config.inner_crop:
        return Cropper("inner", _crop_inner_rect)
    return Cropper("outer", _crop_outer_rect)
