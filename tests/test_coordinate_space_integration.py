"""Cross-step test proving frame_crop's sidecar and face_crop's offset reader
actually agree with each other.

Both src/pipeline/frame_crop/test_step.py and src/pipeline/face_crop/test_step.py test
their own half of the coordinate-space handoff in isolation, each fabricating
the other step's side of the contract by hand. This test instead chains the
real frame_crop writer into the real face_crop reader/writer, so a future
drift between producer and consumer (e.g. a field renamed on one side only)
would actually be caught here rather than only in each step's own tests.
"""

import hashlib
import json

import numpy as np
import pytest

from pipeline.face_crop.detector import FaceDetection
from pipeline.face_crop.step import _load_frame_crop_sidecar
from pipeline.face_crop.step import _write_sidecar as face_crop_write_sidecar
from pipeline.frame_crop.methods.base import Detection
from pipeline.frame_crop.processing import CropDecision
from pipeline.frame_crop.step import _write_sidecar as frame_crop_write_sidecar


@pytest.fixture(autouse=True)
def _project_root_is_tmp_path(monkeypatch, tmp_path):
    """Both steps report paths relative to PROJECT_ROOT; point that at
    tmp_path so test fixtures don't need to live inside the real repo."""
    monkeypatch.setattr("pipeline.shared.paths.PROJECT_ROOT", tmp_path)


def test_face_crop_offset_matches_frame_crop_sidecar_end_to_end(tmp_path):
    raw_path = tmp_path / "data" / "raw" / "batch" / "photo.jpg"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"fake")

    frame_path = tmp_path / "data" / "steps" / "frame_crop" / "batch" / "photo.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"fake-frame-bytes")  # frame_crop records its sha256

    crop_rect_xywh = (100, 150, 600, 750)
    detection = Detection(
        quad=None, info="sam", score=0.9, box_xyxy=None, prompt="rectangular photograph"
    )
    decision = CropDecision(
        result=np.zeros((750, 600, 3), dtype=np.uint8),
        crop_found=True,
        detection=detection,
        crop_rect_xywh=crop_rect_xywh,
    )
    # Real frame_crop writer — not a hand-rolled FrameCropSidecar fixture.
    frame_crop_write_sidecar(frame_path, raw_path, decision)

    # Real face_crop reader: offset and the input-frame hash come from the same
    # sidecar the real writer just produced.
    fc_sidecar = _load_frame_crop_sidecar(frame_path)
    offset = fc_sidecar.crop_rect_xywh[:2]
    assert offset == (100, 150)
    # Producer and consumer agree on the frame hash: what frame_crop recorded is
    # exactly what face_crop re-hashes off disk and carries forward.
    source_sha256 = hashlib.sha256(frame_path.read_bytes()).hexdigest()
    assert fc_sidecar.sha256 == source_sha256

    faces = [FaceDetection(box_xyxy=(10.0, 20.0, 50.0, 80.0), score=0.95)]
    out_dir = tmp_path / "data" / "steps" / "face_crop" / "batch" / "photo"
    out_dir.mkdir(parents=True)
    crop_paths = [out_dir / "face_00.jpg"]
    crop_paths[0].write_bytes(b"fake-crop-bytes")  # _write_sidecar hashes the crop

    sidecar_path = face_crop_write_sidecar(
        out_dir, frame_path, source_sha256, (600, 750), "human face", faces, crop_paths, offset
    )

    data = json.loads(sidecar_path.read_text())
    assert data["source_sha256"] == source_sha256
    face = data["faces"][0]
    assert face["box_xyxy"] == [10.0, 20.0, 50.0, 80.0]
    # Shifted by frame_crop's persisted crop_rect_xywh offset (100, 150) — the
    # same coordinate space ground_truth.json's bbox_xywh is in.
    assert face["box_xyxy_source"] == [110.0, 170.0, 150.0, 230.0]
