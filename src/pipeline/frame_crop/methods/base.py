from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Result of running a detection method on one image.

    `quad` is the rectangle used for cropping (None when nothing was detected).
    `info` is the method name on success or a failure reason otherwise.
    `score`, `box_xyxy` and `prompt` carry the extra signal some methods provide
    (SAM): the detection confidence, the model's axis-aligned box in original-image
    pixels, and the text prompt that produced the detection. Methods without a
    model (canny, saturation) leave them None.
    """

    quad: np.ndarray | None
    info: str
    score: float | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    prompt: str | None = None


class DetectionMethod(Protocol):
    def detect(self, img: np.ndarray, debug_dir: Path | None = None) -> Detection:
        """Detect the photo quad in img.

        Returns a Detection whose quad is set on success and None on failure
        (with info carrying the reason).
        """
        ...
