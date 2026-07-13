from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from pydantic import Field

from pipeline.frame_crop.config import AnnotationConfig, BlurConfig, MorphologyConfig
from pipeline.frame_crop.debug import _annotate, _save
from pipeline.frame_crop.detect import quad_from_mask
from pipeline.frame_crop.filters import apply_blur, apply_morphology
from pipeline.frame_crop.methods.base import Detection
from pipeline.shared.config import StrictConfig


class CannyConfig(StrictConfig):
    threshold1: Annotated[float, Field(ge=0)]
    threshold2: Annotated[float, Field(ge=0)]
    save_output: bool
    annotate: bool
    save_contours: bool
    blur: BlurConfig
    morphology: MorphologyConfig


def _save_contours(
    edges: np.ndarray,
    debug_dir: Path,
    annotation: AnnotationConfig,
) -> None:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    image_area = edges.shape[0] * edges.shape[1]

    rng = np.random.default_rng(seed=42)
    for c in contours:
        color = tuple(int(x) for x in rng.integers(80, 256, size=3))
        cv2.drawContours(out, [c], -1, color, 3)

    largest_frac = (
        cv2.contourArea(max(contours, key=cv2.contourArea)) / image_area
        if contours else 0.0
    )
    out = _annotate(
        out,
        {"contours": len(contours), "largest": f"{largest_frac:.1%}"},
        annotation.scale,
        annotation.bold,
    )
    _save(debug_dir / "canny_contours.jpg", out)


class CannyMethod:
    def __init__(
        self,
        config: CannyConfig,
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = apply_blur(gray, self.config.blur)

        if debug_dir is not None and self.config.blur.save_output:
            out = (
                _annotate(
                    blurred,
                    {"kernel_size": self.config.blur.kernel_size},
                    self.annotation.scale,
                    self.annotation.bold,
                )
                if self.config.blur.annotate
                else blurred
            )
            _save(debug_dir / "blurred.jpg", out)

        edges = cv2.Canny(blurred, self.config.threshold1, self.config.threshold2)

        if debug_dir is not None and self.config.save_output:
            canny_params = {
                "low": self.config.threshold1,
                "high": self.config.threshold2,
                "blur": (
                    f"k{self.config.blur.kernel_size}"
                    if self.config.blur.enabled
                    else "off"
                ),
            }
            out = (
                _annotate(
                    edges,
                    canny_params,
                    self.annotation.scale,
                    self.annotation.bold,
                )
                if self.config.annotate
                else edges
            )
            _save(debug_dir / "canny_edges.jpg", out)

        if self.config.morphology.enabled:
            edges_before = edges
            edges = apply_morphology(edges, self.config.morphology)
            if debug_dir is not None and self.config.morphology.save_diff:
                added = cv2.subtract(edges, edges_before)
                out = cv2.cvtColor(edges_before, cv2.COLOR_GRAY2BGR)
                out[added > 0] = (0, 220, 0)
                out = _annotate(
                    out,
                    {
                        "original_px": int(np.count_nonzero(edges_before)),
                        "added_px": int(np.count_nonzero(added)),
                        "kernel": self.config.morphology.kernel_size,
                        "iterations": self.config.morphology.iterations,
                    },
                    self.annotation.scale,
                    self.annotation.bold,
                )
                _save(debug_dir / "morphology_diff.jpg", out)
            if debug_dir is not None and self.config.morphology.save_output:
                out = (
                    _annotate(
                        edges,
                        {
                            "kernel_size": self.config.morphology.kernel_size,
                            "iterations": self.config.morphology.iterations,
                        },
                        self.annotation.scale,
                        self.annotation.bold,
                    )
                    if self.config.morphology.annotate
                    else edges
                )
                _save(debug_dir / "canny_closed.jpg", out)

        if debug_dir is not None and self.config.save_contours:
            _save_contours(edges, debug_dir, self.annotation)

        quad, reason = quad_from_mask(edges, self.min_area_fraction, self.max_area_fraction)
        if quad is not None:
            return Detection(quad=quad, info="canny")
        return Detection(None, reason)
