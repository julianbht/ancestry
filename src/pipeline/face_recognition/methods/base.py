from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class EmbeddingResult:
    """An embedding plus any per-face attributes the method computed alongside it.

    InsightFace's model packs run a gender/age model on every crop as part of a
    single get() call, so age/gender/det_score come essentially for free and are
    worth keeping (age_estimation is a separate planned step; these are the
    recognition model's own cheap predictions). Methods that don't produce a
    given attribute leave it None (e.g. DeepFace's represent() yields only the
    embedding)."""

    embedding: np.ndarray
    age: int | None = None
    gender: str | None = None  # "M" or "F"
    det_score: float | None = None


class EmbeddingMethod(Protocol):
    name: str

    def embed(self, crop_path: Path) -> EmbeddingResult | None:
        """Return the embedding (and any extra attributes) for a face crop, or
        None on failure."""
        ...
