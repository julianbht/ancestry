"""Typed output schema for the face_recognition step."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class RecognitionStatus(StrEnum):
    RECOGNIZED = "recognized"
    UNKNOWN = "unknown"
    NO_EMBEDDING = "no_embedding"


class Candidate(BaseModel):
    """One gallery person ranked by nearest distance to the query face. Purely a
    ranking entry — independent of the threshold, so candidates[0] can be a
    near-miss on an UNKNOWN face. The candidates[0]→candidates[1] distance gap is
    the natural ambiguity signal for prioritising human review."""

    person_id: str
    distance: float


class RecognitionEntry(BaseModel):
    """Recognition result for one face crop within a frame."""

    face_index: int
    crop_image: str
    # None when status is UNKNOWN or NO_EMBEDDING
    person_id: str | None
    # Nearest-neighbour distance to the gallery; None when embedding failed
    distance: float | None
    status: RecognitionStatus
    # Top-K gallery people by nearest distance, ascending (candidates[0] is the
    # nearest). person_id/distance/status above are all derived from candidates[0]
    # against the threshold. Empty when the embedding failed (NO_EMBEDDING).
    candidates: list[Candidate]
    # Attributes the recognition model computed alongside the embedding, kept
    # rather than discarded. InsightFace's model pack runs a gender/age model on
    # every crop, so these come for free; None for methods that don't produce
    # them (DeepFace) or when the model pack omits the gender/age model. age is
    # this model's own cheap estimate — distinct from the planned age_estimation
    # step; gender is "M"/"F"; det_score is the detection confidence.
    age: int | None = None
    gender: str | None = None
    det_score: float | None = None
    # Carried from face_crop's faces.json (FaceCropEntry.sha256): the SHA-256 of
    # the crop bytes this face's embedding was computed from. The embedding (and
    # thus distance/candidates and the embeddings.npy row) is a pure function of
    # (those bytes, model), so this is the validity token: if face_crop is re-run
    # and changes the crop, the recorded hash won't match the new one and the
    # stored result is known-stale.
    crop_sha256: str


class RecognitionSidecar(BaseModel):
    """Schema for recognition.json written alongside each frame's results."""

    source_sidecar: str
    model: str
    detector_backend: str
    distance_metric: str
    threshold: float
    # Number of distinct person_ids in the gallery used for this run
    gallery_size: int
    # Project-relative path to the query embeddings (embeddings.npy), shape
    # [n_faces, dim] aligned by position to `recognitions` (row i ↔
    # recognitions[i]); failed embeddings are a row of NaN. None when no face in
    # this frame embedded successfully. Tied to `model` — invalid if the model
    # changes. These are the expensive artifact: with them, any re-ranking,
    # threshold/metric change, or unknown-clustering is a recompute, never a re-embed.
    embeddings_file: str | None
    recognitions: list[RecognitionEntry]
