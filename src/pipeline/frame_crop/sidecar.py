"""Typed schema for frame_crop's per-image sidecar JSON.

Written next to each crop as <crop>.json. Kept separate from Detection (the
in-memory dataclass produced by each detection method) so the wire format has
its own explicit, validated schema rather than being implicit in however
Detection happens to be shaped.
"""

from __future__ import annotations

from pydantic import BaseModel

from pipeline.frame_crop.methods.base import Detection


class DetectionSidecar(BaseModel):
    info: str
    score: float | None
    box_xyxy: tuple[float, float, float, float] | None
    prompt: str | None
    quad: list[list[float]] | None

    @classmethod
    def from_detection(cls, detection: Detection) -> "DetectionSidecar":
        return cls(
            info=detection.info,
            score=detection.score,
            box_xyxy=detection.box_xyxy,
            prompt=detection.prompt,
            quad=detection.quad.tolist() if detection.quad is not None else None,
        )


class FrameCropSidecar(BaseModel):
    """Schema for <crop>.json.

    crop_rect_xywh is the resolved crop rect in source_image's own pixel
    coordinates (full image bounds when crop_found=False) — downstream steps
    add it to their own locally-detected boxes to express them in that same
    shared coordinate space.
    """

    source_image: str
    crop_image: str
    crop_found: bool
    crop_rect_xywh: tuple[int, int, int, int]
    detection: DetectionSidecar
    # SHA-256 of crop_image's bytes, recorded when frame_crop wrote it. The frame's
    # content identity: face_crop re-hashes the frame it runs on and compares it to
    # this, so a frame_crop re-run that changes the frame is a detectable mismatch
    # (face_crop throws) rather than silently feeding stale geometry downstream.
    sha256: str
