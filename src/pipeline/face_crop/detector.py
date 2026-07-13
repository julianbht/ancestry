"""SAM-based face detection.

Wraps the SAM 3 image model with a text prompt (e.g. "face") and returns every
instance it finds above the score threshold. Unlike frame_crop — which keeps the
single best quad — face_crop needs *all* detections, since a frame may contain
many faces.

Detection is kept separate from cropping and I/O: this module only turns an image
into a list of boxes. See cropping.py for turning a box into a crop, and step.py
for orchestration.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.face_crop.config import SamConfig


@dataclass(frozen=True)
class FaceDetection:
    """One detected face: its axis-aligned box in frame pixels and the model
    confidence. Boxes are xyxy (x0, y0, x1, y1)."""

    box_xyxy: tuple[float, float, float, float]
    score: float

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box_xyxy
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class SamFaceDetector:
    """Detects faces in a frame using the SAM 3 image model and a text prompt."""

    def __init__(self, config: SamConfig) -> None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise RuntimeError(
                "SAM3 is not installed. Run: uv run python scripts/setup_sam3.py"
            ) from e

        if not config.checkpoint_path.exists():
            raise RuntimeError(
                f"SAM3 checkpoint not found at {config.checkpoint_path}. "
                "Run: uv run python scripts/setup_sam3.py"
            )

        model = build_sam3_image_model(
            checkpoint_path=str(config.checkpoint_path),
            load_from_HF=False,
            device=config.device,
            eval_mode=True,
        )
        self._processor = Sam3Processor(model)
        self._config = config

    def detect(self, img: np.ndarray) -> list[FaceDetection]:
        """Return all faces with score >= threshold, sorted by score descending."""
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # SAM3's image model force-casts backbone features to bfloat16 internally,
        # assuming inference runs under AMP autocast (as every SAM3 example does).
        # Without this context the bf16 features hit float32 weights:
        # "mat1 and mat2 must have the same dtype" (facebookresearch/sam3#507).
        device_type = "cuda" if "cuda" in self._config.device else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            state = self._processor.set_image(pil_img)
            output = self._processor.set_text_prompt(
                state=state, prompt=self._config.prompt
            )

        scores = output["scores"]  # (N,) float tensor
        boxes = output["boxes"]    # (N, 4) xyxy in original-image pixels

        if boxes is None or len(boxes) == 0:
            return []

        scores_np = scores.cpu().float().numpy()
        boxes_np = boxes.cpu().float().numpy()

        faces = [
            FaceDetection(
                box_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(s),
            )
            for b, s in zip(boxes_np, scores_np)
            if s >= self._config.score_threshold
        ]
        faces.sort(key=lambda f: f.score, reverse=True)
        return faces
