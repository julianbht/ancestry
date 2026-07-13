"""Typed schema for face_crop's per-frame faces.json sidecar.

Kept separate from FaceDetection (the in-memory dataclass the detector
produces) so the wire format has its own explicit, validated schema rather
than being implicit in however FaceDetection happens to be shaped.
"""

from __future__ import annotations

from pydantic import BaseModel


class FaceCropEntry(BaseModel):
    """One detected face within a frame.

    box_xyxy is in this frame's own local pixel coordinates; box_xyxy_source
    is the same box shifted into the source image's pixel coordinates (the
    same space ground_truth.json's bbox_xywh is in), via frame_crop's
    crop_rect_xywh offset.
    """

    index: int
    box_xyxy: tuple[float, float, float, float]
    score: float
    box_xyxy_source: tuple[float, float, float, float]
    crop_image: str
    # SHA-256 of crop_image's bytes, recorded when face_crop wrote it. The crop's
    # content identity: a downstream step (face_recognition, age_estimation) carries
    # this alongside its derived artifact, so a face_crop re-run that changes the
    # crop is detectable (re-hash won't match) rather than silently stale.
    sha256: str


class FaceCropSidecar(BaseModel):
    """Schema for faces.json, written next to each frame's face crops."""

    source_image: str
    # SHA-256 of source_image's bytes (the frame_crop output this ran on). Carried
    # from frame_crop's FrameCropSidecar.sha256 after face_crop re-hashed the frame
    # and confirmed it matched. The validity token for the input frame: if frame_crop
    # is re-run and changes the frame while face_crop's state still says done, this
    # recorded hash no longer matches the frame on disk, so the faces.json (and its
    # crops) is detectably stale rather than silently so.
    source_sha256: str
    image_size: tuple[int, int]  # (width, height)
    prompt: str
    face_count: int
    faces: list[FaceCropEntry]
