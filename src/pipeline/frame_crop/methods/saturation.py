from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from pydantic import Field

from pipeline.frame_crop.config import AnnotationConfig, MorphologyConfig
from pipeline.frame_crop.debug import _annotate, _save
from pipeline.frame_crop.detect import quad_from_mask
from pipeline.frame_crop.filters import apply_morphology
from pipeline.frame_crop.methods.base import Detection
from pipeline.shared.config import StrictConfig


class SaturationConfig(StrictConfig):
    threshold: Annotated[int, Field(ge=0, le=255)]
    save_output: bool
    annotate: bool
    morphology: MorphologyConfig


class SaturationMethod:
    def __init__(
        self,
        config: SaturationConfig,
        min_area_fraction: float,
        max_area_fraction: float,
        annotation: AnnotationConfig,
    ) -> None:
        self.config = config
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.annotation = annotation

    def detect(
        self, img: np.ndarray, debug_dir: Path | None = None
    ) -> Detection:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        _, mask = cv2.threshold(
            s_channel, self.config.threshold, 255, cv2.THRESH_BINARY_INV
        )

        if debug_dir is not None and self.config.save_output:
            out = (
                _annotate(
                    mask,
                    {"threshold": self.config.threshold},
                    self.annotation.scale,
                    self.annotation.bold,
                )
                if self.config.annotate
                else mask
            )
            _save(debug_dir / "sat_mask.jpg", out)

        if self.config.morphology.enabled:
            mask = apply_morphology(mask, self.config.morphology)
            if debug_dir is not None and self.config.morphology.save_output:
                out = (
                    _annotate(
                        mask,
                        {
                            "kernel_size": self.config.morphology.kernel_size,
                            "iterations": self.config.morphology.iterations,
                        },
                        self.annotation.scale,
                        self.annotation.bold,
                    )
                    if self.config.morphology.annotate
                    else mask
                )
                _save(debug_dir / "sat_closed.jpg", out)

        quad, reason = quad_from_mask(mask, self.min_area_fraction, self.max_area_fraction)
        if quad is not None:
            return Detection(quad=quad, info="saturation")
        return Detection(None, reason)
