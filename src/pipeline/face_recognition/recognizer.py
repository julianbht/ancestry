"""Nearest-neighbour gallery matching against an EmbeddingMethod."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.face_recognition.methods.base import EmbeddingMethod
from pipeline.face_recognition.sidecar import Candidate, RecognitionStatus


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """Distance between two embeddings. Shared with the hpopt leave-one-out eval."""
    if metric == "cosine":
        return _cosine_distance(a, b)
    if metric == "euclidean":
        return _euclidean_distance(a, b)
    if metric == "euclidean_l2":
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        a_n = a / norm_a if norm_a > 0.0 else a
        b_n = b / norm_b if norm_b > 0.0 else b
        return _euclidean_distance(a_n, b_n)
    raise ValueError(f"Unknown distance metric: {metric!r}")


@dataclass
class RecognitionResult:
    person_id: str | None
    distance: float | None  # None only when embedding failed
    status: RecognitionStatus
    # Gallery people ranked by nearest distance, ascending, truncated to top_k.
    # Empty when the embedding failed.
    candidates: list[Candidate]
    # The query embedding, returned so the caller can persist it (embeddings are
    # the expensive artifact). None when embedding failed.
    embedding: np.ndarray | None
    # Per-face attributes the method computed alongside the embedding (None when
    # embedding failed or the method doesn't produce them). See EmbeddingResult.
    age: int | None = None
    gender: str | None = None
    det_score: float | None = None


class FaceRecognizer:
    """Nearest-neighbour face recognizer against a prebuilt embedding gallery.

    For each query face crop the recognizer:
      1. Extracts an embedding via the given EmbeddingMethod.
      2. Finds the gallery entry with the smallest distance across all reference
         embeddings for every person.
      3. If that distance is ≤ threshold → "recognized" with that person_id;
         otherwise → "unknown".
    """

    def __init__(
        self,
        gallery: dict[str, list[np.ndarray]],
        method: EmbeddingMethod,
        distance_metric: str,
        threshold: float,
        top_k: int,
    ) -> None:
        self.gallery = gallery
        self.method = method
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.top_k = top_k

    def recognize(self, crop_path: Path) -> RecognitionResult:
        result = self.method.embed(crop_path)
        if result is None:
            return RecognitionResult(
                person_id=None, distance=None, status=RecognitionStatus.NO_EMBEDDING,
                candidates=[], embedding=None,
            )
        emb = result.embedding

        # Nearest reference distance per person, then rank people by it. gallery is
        # non-empty and every person has >= 1 embedding (build_gallery guarantees
        # both), so ranked is non-empty.
        per_person_best = {
            person_id: min(distance(emb, g, self.distance_metric) for g in embeddings)
            for person_id, embeddings in self.gallery.items()
        }
        ranked = sorted(per_person_best.items(), key=lambda kv: kv[1])
        candidates = [Candidate(person_id=pid, distance=d) for pid, d in ranked[: self.top_k]]

        best_pid, best_dist = ranked[0]
        if best_dist <= self.threshold:
            status, person_id = RecognitionStatus.RECOGNIZED, best_pid
        else:
            status, person_id = RecognitionStatus.UNKNOWN, None
        return RecognitionResult(
            person_id=person_id, distance=best_dist, status=status,
            candidates=candidates, embedding=emb,
            age=result.age, gender=result.gender, det_score=result.det_score,
        )
