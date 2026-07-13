"""Build the in-memory face index the recommender retrieves over.

One row per detected face that has a usable embedding: the L2-normalised
InsightFace vector, the crop image it came from, the source photo, and — when
the face was annotated in the Label Studio ground truth — its person_id.

The embeddings are the ones face_recognition already computed on the cluster
(data/steps/face_recognition/<...>/embeddings.npy), aligned row-for-row with
the recognition.json sidecar. Faces whose recognizer status is "no_embedding"
carry a NaN placeholder row and are dropped here. Ground-truth labels are joined
by reusing face_recognition's own match_ground_truth, so the labelled subset is
identical to the gallery the recognition run was built from.

No model is loaded: this is a pure read of artifacts already on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from pipeline.face_recognition.gallery import match_ground_truth
from pipeline.shared.paths import CURATED_DIR, STEPS_DIR

RECOGNITION_DIR = STEPS_DIR / "face_recognition"
FACE_CROP_DIR = STEPS_DIR / "face_crop"
GROUND_TRUTH_FILE = CURATED_DIR / "face_annotation" / "ground_truth.json"


@dataclass
class FaceIndex:
    """All embedded faces in the collection, ready for similarity search."""

    embeddings: np.ndarray  # (N, D) float32, L2-normalised
    crop_images: list[str]  # posix path of each face crop, length N
    source_photos: list[str]  # "<folder>/<stem>" the face was detected in
    person_ids: list[str | None]  # ground-truth label, None if unlabelled

    def __len__(self) -> int:
        return len(self.crop_images)

    def indices_for_person(self, person_id: str) -> list[int]:
        return [i for i, p in enumerate(self.person_ids) if p == person_id]

    def labeled_indices(self) -> list[int]:
        return [i for i, p in enumerate(self.person_ids) if p is not None]

    def person_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.person_ids:
            if p is not None:
                counts[p] = counts.get(p, 0) + 1
        return counts


def _source_photo(crop_image: str) -> str:
    """'data/steps/face_crop/<folder>/<stem>/face_00.jpg' -> '<folder>/<stem>'."""
    p = Path(crop_image)
    return f"{p.parent.parent.name}/{p.parent.name}"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def build_index(overlap_threshold: float = 0.30) -> FaceIndex:
    """Read every recognition sidecar + embeddings.npy, join ground-truth labels."""
    embeddings: list[np.ndarray] = []
    crop_images: list[str] = []
    sources: list[str] = []

    n_skipped_no_emb = 0
    for sidecar in sorted(RECOGNITION_DIR.rglob("recognition.json")):
        emb_path = sidecar.parent / "embeddings.npy"
        if not emb_path.exists():
            continue  # every face here failed to embed; nothing to index
        data = json.loads(sidecar.read_text())
        matrix = np.load(emb_path)
        for i, rec in enumerate(data["recognitions"]):
            row = matrix[i]
            if rec["status"] == "no_embedding" or not np.all(np.isfinite(row)):
                n_skipped_no_emb += 1
                continue
            crop = Path(rec["crop_image"]).as_posix()
            embeddings.append(row)
            crop_images.append(crop)
            sources.append(_source_photo(crop))

    matrix = _normalize(np.vstack(embeddings))

    # Join ground-truth labels by crop path (reuse face_recognition's matcher).
    crop_to_person: dict[str, str] = {}
    for m in match_ground_truth(GROUND_TRUTH_FILE, FACE_CROP_DIR, overlap_threshold):
        if m.person_id:
            crop_to_person[Path(m.crop_path).as_posix()] = m.person_id
    person_ids = [crop_to_person.get(c) for c in crop_images]

    n_labeled = sum(p is not None for p in person_ids)
    logger.info(
        f"Face index: {len(crop_images)} embedded faces "
        f"({n_skipped_no_emb} dropped without embedding); "
        f"{n_labeled} labelled across {len(set(filter(None, person_ids)))} people"
    )
    return FaceIndex(matrix, crop_images, sources, person_ids)
