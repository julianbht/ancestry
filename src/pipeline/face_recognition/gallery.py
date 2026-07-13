"""Match ground-truth annotations to face_crop detections, and build the
face-recognition gallery from the matched labeled faces.

For each annotated face in ground_truth.json, we find the best-matching
face_crop detection via overlap coefficient on source-image coordinates
(match_ground_truth). build_gallery() then takes the labeled subset (faces
with a person_id) and extracts a DeepFace embedding for each, producing a
dict mapping person_id -> list of embeddings (one per matched reference crop).

Overlap coefficient (intersection / area of the smaller box), not IoU, is used
for the same reason pipeline/experiments/face_crop/metrics.py uses it to score
SAM detections against this same ground truth: GT boxes include hair margin
that face_crop's boxes don't, so plain IoU would unfairly penalize an
otherwise-correct match. See pipeline/shared/boxes.py.

Frames whose face_crop sidecar is missing or unparseable are skipped with a
warning — this covers the small number of old sidecars that predate the
box_xyxy_source field.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from pipeline.face_recognition.methods.base import EmbeddingMethod
from pipeline.shared.boxes import box_overlap


def _xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


@dataclass
class MatchedFace:
    """A ground-truth-annotated face resolved to its face_crop detection."""

    person_id: str | None  # None means the GT face was annotated but left unidentified
    age: int | None
    crop_path: Path
    gt_key: str  # "<folder>/<file>.jpg" — the GT item this face came from


def match_ground_truth(
    ground_truth_path: Path, faces_dir: Path, overlap_threshold: float
) -> list[MatchedFace]:
    """Match every GT-annotated face (identified or not) to its best-overlapping
    face_crop detection. Faces with no face_crop output, or no detection at or
    above overlap_threshold, are dropped."""
    gt = json.loads(ground_truth_path.read_text())
    matches: list[MatchedFace] = []

    n_no_crop_dir = 0
    n_no_overlap_match = 0

    for gt_key, item in gt["items"].items():
        # gt_key = "<folder>/<file>.jpg"; face_crop dir = faces_dir/<folder>/<stem>/
        folder, filename = gt_key.split("/", 1)
        stem = Path(filename).stem
        crop_dir = faces_dir / folder / stem

        if not crop_dir.exists():
            n_no_crop_dir += len(item["faces"])
            logger.debug(f"Gallery: no face_crop output dir for {gt_key}")
            continue

        sidecar_path = crop_dir / "faces.json"
        try:
            sidecar_data = json.loads(sidecar_path.read_text())
        except Exception as e:
            logger.warning(f"Gallery: could not read {sidecar_path}: {e} — skipping")
            n_no_crop_dir += len(item["faces"])
            continue

        detected_faces = sidecar_data.get("faces", [])

        for gt_face in item["faces"]:
            gt_box = _xywh_to_xyxy(gt_face["bbox_xywh"])

            # Find the detected face with highest overlap to this GT box. Uses
            # box_xyxy_source (source-image coordinates), which is the same
            # space as ground_truth.json's bbox_xywh. Skip faces that lack it
            # (old sidecars predating the coordinate-spaces update).
            best_overlap = 0.0
            best_crop: str | None = None
            for det in detected_faces:
                if "box_xyxy_source" not in det:
                    continue
                det_box = tuple(det["box_xyxy_source"])
                overlap = box_overlap(np.array(gt_box), np.array(det_box))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_crop = det.get("crop_image")

            if best_crop is None or best_overlap < overlap_threshold:
                n_no_overlap_match += 1
                logger.debug(
                    f"Gallery: no overlap match for face in {gt_key} "
                    f"(best={best_overlap:.2f} < threshold={overlap_threshold})"
                )
                continue

            matches.append(
                MatchedFace(
                    person_id=gt_face.get("person_id"),
                    age=gt_face.get("age"),
                    crop_path=Path(best_crop),
                    gt_key=gt_key,
                )
            )

    logger.info(
        f"Matched {len(matches)} GT face(s) to face_crop output "
        f"({n_no_crop_dir} faces without face_crop output, "
        f"{n_no_overlap_match} faces without an overlap match)"
    )
    return matches


def build_gallery(
    ground_truth_path: Path,
    faces_dir: Path,
    overlap_threshold: float,
    method: EmbeddingMethod,
) -> dict[str, list[np.ndarray]]:
    """Return {person_id: [embedding, ...]} built from ground-truth-matched crops."""
    matches = [
        m for m in match_ground_truth(ground_truth_path, faces_dir, overlap_threshold) if m.person_id
    ]
    gallery: dict[str, list[np.ndarray]] = {}

    logger.info(
        f"Gallery: embedding {len(matches)} labeled ground-truth face(s) "
        f"with {method.name} "
        "(first call loads model weights, which can take a while)"
    )

    n_embed_fail = 0
    for n_embed_attempted, m in enumerate(matches, start=1):
        start = time.perf_counter()
        result = method.embed(m.crop_path)
        elapsed = time.perf_counter() - start

        if result is None:
            n_embed_fail += 1
            logger.info(
                f"Gallery: [{n_embed_attempted}/{len(matches)}] embedding FAILED for "
                f"{m.crop_path} ({elapsed:.1f}s)"
            )
            continue

        assert m.person_id is not None  # filtered above
        gallery.setdefault(m.person_id, []).append(result.embedding)
        logger.info(
            f"Gallery: [{n_embed_attempted}/{len(matches)}] embedded {m.crop_path} "
            f"({elapsed:.1f}s)"
        )

    n_matched = sum(len(v) for v in gallery.values())
    logger.info(
        f"Gallery built: {n_matched} reference crops for {len(gallery)} people "
        f"({n_embed_fail} embedding failures)"
    )
    return gallery
