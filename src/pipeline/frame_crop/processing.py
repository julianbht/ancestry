from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.frame_crop.config import FrameCropConfig
from pipeline.frame_crop.cropper import select_cropper
from pipeline.frame_crop.debug import save_debug_quad
from pipeline.frame_crop.methods.base import Detection, DetectionMethod


@dataclass(frozen=True)
class CropDecision:
    result: np.ndarray
    crop_found: bool
    detection: Detection
    # (x, y, w, h) of `result` within the source image's own pixel coordinates.
    # Full image bounds when crop_found=False, so "no crop happened" is just
    # the identity rect rather than a special case for downstream consumers.
    crop_rect_xywh: tuple[int, int, int, int]


def crop_image(
    img: np.ndarray,
    method: DetectionMethod,
    config: FrameCropConfig,
    debug_dir: Path,
) -> CropDecision:
    detection = method.detect(img, debug_dir)

    if config.debug.save_quad:
        save_debug_quad(img, detection, config, debug_dir)

    if detection.quad is None:
        h, w = img.shape[:2]
        return CropDecision(
            result=img, crop_found=False, detection=detection, crop_rect_xywh=(0, 0, w, h)
        )

    cropper = select_cropper(config)
    result, rect = cropper.apply(img, detection.quad, config)
    return CropDecision(result=result, crop_found=True, detection=detection, crop_rect_xywh=rect)
