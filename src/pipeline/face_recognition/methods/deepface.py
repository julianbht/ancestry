"""DeepFace embedding method.

DeepFace (and TensorFlow under it) are imported lazily so loading this module
does not trigger heavy model initialisation until the first embed() call.

The XLA libdevice workaround is a one-time global setup: some CUDA container
images omit the NVVM libdevice library that XLA's PTX backend requires. Triton
(already in the venv) ships a copy; we symlink it into a stub directory and set
XLA_FLAGS so TensorFlow can find it.
"""

from __future__ import annotations

import os
import site
from pathlib import Path

import numpy as np
from loguru import logger

from pipeline.face_recognition.methods.base import EmbeddingResult
from pipeline.shared.paths import MODELS_DIR

_XLA_CUDA_STUB_DIR = MODELS_DIR / ".cuda_stub"
_xla_configured = False


def _setup_xla_libdevice() -> None:
    global _xla_configured
    if _xla_configured:
        return
    _xla_configured = True

    triton_libdevice: Path | None = None
    for sp in site.getsitepackages():
        candidate = Path(sp) / "triton" / "backends" / "nvidia" / "lib" / "libdevice.10.bc"
        if candidate.exists():
            triton_libdevice = candidate
            break

    if triton_libdevice is None:
        return

    link_dir = _XLA_CUDA_STUB_DIR / "nvvm" / "libdevice"
    link_dir.mkdir(parents=True, exist_ok=True)
    link_path = link_dir / "libdevice.10.bc"
    if not link_path.exists():
        link_path.symlink_to(triton_libdevice)

    flag = f"--xla_gpu_cuda_data_dir={_XLA_CUDA_STUB_DIR}"
    existing = os.environ.get("XLA_FLAGS", "")
    if flag not in existing:
        os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()


class DeepFaceMethod:
    def __init__(self, model_name: str, detector_backend: str, models_dir: Path) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.name = f"deepface/{model_name}/{detector_backend}"
        models_dir.mkdir(parents=True, exist_ok=True)
        # Must be set before the first DeepFace import (lazy-imported on first embed call).
        os.environ["DEEPFACE_HOME"] = str(models_dir)

    def embed(self, crop_path: Path) -> EmbeddingResult | None:
        _setup_xla_libdevice()
        try:
            from deepface import DeepFace  # noqa: PLC0415

            result = DeepFace.represent(
                img_path=str(crop_path),
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend=self.detector_backend,
            )
            # represent() yields only the embedding; age/gender would need a
            # separate analyze() pass, so they stay None here.
            return EmbeddingResult(embedding=np.array(result[0]["embedding"], dtype=np.float32))
        except Exception as e:
            logger.warning(f"DeepFace embedding failed for {crop_path}: {e}")
            return None
