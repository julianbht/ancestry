"""InsightFace embedding method.

Uses FaceAnalysis (buffalo_l or similar model pack) to run detection +
alignment + embedding in one pass on each face crop. The model is loaded once
at construction time; embed() is a pure inference call.

InsightFace is imported lazily so that importing this module on machines
without insightface installed does not raise an ImportError.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from pipeline.face_recognition.methods.base import EmbeddingResult


def _flatten_nested_model_dir(model_dir: Path, model_pack: str) -> None:
    """Some InsightFace model zips (e.g. antelopev2, unlike buffalo_l) wrap
    their .onnx files in an extra <model_pack>/ folder, so they land one level
    too deep for FaceAnalysis's flat glob(model_dir/*.onnx) lookup, which then
    fails its 'detection' in self.models assertion. Flatten it if found."""
    nested = model_dir / model_pack
    if not nested.is_dir() or any(model_dir.glob("*.onnx")):
        return
    logger.info(f"Flattening nested InsightFace model dir: {nested}")
    for onnx_file in nested.glob("*.onnx"):
        onnx_file.rename(model_dir / onnx_file.name)
    nested.rmdir()


class InsightFaceMethod:
    def __init__(
        self,
        model_pack: str,
        models_dir: Path,
        det_size: tuple[int, int] = (640, 640),
        pad_ratio: float = 0.4,
        det_thresh: float = 0.5,
    ) -> None:
        from insightface.app import FaceAnalysis  # noqa: PLC0415
        from insightface.utils import ensure_available  # noqa: PLC0415

        self.name = f"insightface/{model_pack}"
        self._pad_ratio = pad_ratio
        models_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(ensure_available("models", model_pack, root=str(models_dir)))
        _flatten_nested_model_dir(model_dir, model_pack)
        self._app = FaceAnalysis(
            name=model_pack,
            root=str(models_dir),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)

    def embed(self, crop_path: Path) -> EmbeddingResult | None:
        img = cv2.imread(str(crop_path))
        if img is None:
            logger.warning(f"InsightFace: could not read {crop_path}")
            return None
        if self._pad_ratio > 0:
            h, w = img.shape[:2]
            pad = int(max(h, w) * self._pad_ratio)
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        faces = self._app.get(img)
        if not faces:
            logger.debug(f"InsightFace: no face detected in {crop_path}")
            return None
        # Face crops contain one face; if InsightFace somehow finds multiple,
        # take the largest detection (most of the frame = most likely the subject).
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        # The model pack's genderage model runs as part of get(), so age/sex are
        # already computed; det_score comes from detection. Guard each in case a
        # pack omits the genderage model. Face.sex is "M"/"F" or None.
        return EmbeddingResult(
            embedding=best.embedding.astype(np.float32),
            age=int(best.age) if best.age is not None else None,
            gender=best.sex,
            det_score=float(best.det_score) if best.det_score is not None else None,
        )
