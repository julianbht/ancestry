"""Read the human face labels that feed the Gramps merge.

Turns data/curated/face_annotation/ground_truth.json into flat, typed records.
The JSON nests faces under their source photo; every consumer here wants "all
labelled faces" instead, so the nesting is flattened once, at the edge.

Coordinates are passed through untouched: each bbox_xywh is already in the pixel
space of its item's `image_rel` (`image_size` gives that image's dimensions), so
the media file and the region denominators are always the same image and there
is no pipeline geometry to replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.shared.paths import CURATED_DIR

GROUND_TRUTH_FILE = CURATED_DIR / "face_annotation" / "ground_truth.json"


@dataclass(frozen=True)
class AnnotatedFace:
    """One human-labelled face, resolved to the person it belongs to."""

    person_id: str
    image_rel: str  # source photo, relative to DATA_DIR
    image_size: tuple[int, int]  # (width, height) of that photo
    bbox_xywh: tuple[float, float, float, float]  # pixels, in image_rel's space
    is_portrait: bool


@dataclass(frozen=True)
class FaceAnnotations:
    faces: list[AnnotatedFace]
    unlabelled_count: int  # faces with no person_id — detected, never named


def load_face_annotations(path: Path = GROUND_TRUTH_FILE) -> FaceAnnotations:
    """Load every labelled face, sorted by photo for a deterministic result."""
    data = json.loads(path.read_text(encoding="utf-8"))
    faces: list[AnnotatedFace] = []
    unlabelled = 0

    for key in sorted(data["items"]):
        item = data["items"][key]
        width, height = item["image_size"]
        for face in item["faces"]:
            person_id = face.get("person_id")
            if not person_id:
                unlabelled += 1
                continue
            faces.append(
                AnnotatedFace(
                    person_id=person_id,
                    image_rel=item["image_rel"],
                    image_size=(width, height),
                    bbox_xywh=tuple(face["bbox_xywh"]),
                    is_portrait=bool(face.get("is_portrait")),
                )
            )

    return FaceAnnotations(faces=faces, unlabelled_count=unlabelled)
