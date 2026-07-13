from pathlib import Path

import numpy as np

from pipeline.frame_crop.methods.base import Detection
from pipeline.frame_crop.processing import crop_image


class _StubMethod:
    """DetectionMethod stub returning a fixed Detection regardless of input."""

    def __init__(self, detection: Detection) -> None:
        self._detection = detection

    def detect(self, img: np.ndarray, debug_dir: Path | None = None) -> Detection:
        return self._detection


def test_crop_image_no_quad_uses_full_image_as_rect(sample_image, make_config):
    method = _StubMethod(Detection(quad=None, info="no quad found"))
    decision = crop_image(sample_image, method, make_config(), Path("unused"))

    assert decision.crop_found is False
    assert decision.result is sample_image
    h, w = sample_image.shape[:2]
    assert decision.crop_rect_xywh == (0, 0, w, h)


def test_crop_image_with_quad_returns_resolved_rect(sample_image, sample_quad, make_config):
    method = _StubMethod(Detection(quad=sample_quad, info="sam", score=0.9))
    decision = crop_image(sample_image, method, make_config(inner_crop=True), Path("unused"))

    assert decision.crop_found is True
    assert decision.crop_rect_xywh == (100, 150, 600, 750)
    assert decision.result.shape[:2] == (750, 600)
