import hashlib

import numpy as np
import pytest

from pipeline.frame_crop.methods.base import Detection
from pipeline.frame_crop.processing import CropDecision
from pipeline.frame_crop.sidecar import FrameCropSidecar
from pipeline.frame_crop.step import _write_sidecar


@pytest.fixture(autouse=True)
def _project_root_is_tmp_path(monkeypatch, tmp_path):
    """_write_sidecar reports paths relative to PROJECT_ROOT; point that at
    tmp_path so test fixtures don't need to live inside the real repo."""
    monkeypatch.setattr("pipeline.shared.paths.PROJECT_ROOT", tmp_path)


def _make_decision(crop_found: bool, crop_rect_xywh: tuple[int, int, int, int]) -> CropDecision:
    detection = Detection(
        quad=None, info="sam", score=0.9, box_xyxy=(10.0, 20.0, 50.0, 80.0),
        prompt="rectangular photograph",
    )
    _x, _y, w, h = crop_rect_xywh
    result = np.zeros((h, w, 3), dtype=np.uint8)
    return CropDecision(
        result=result, crop_found=crop_found, detection=detection, crop_rect_xywh=crop_rect_xywh
    )


def test_write_sidecar_persists_resolved_crop_rect(tmp_path):
    input_path = tmp_path / "data" / "raw" / "batch" / "photo.jpg"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"fake")
    out_path = tmp_path / "data" / "steps" / "frame_crop" / "batch" / "photo.jpg"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"fake-frame-bytes")  # _write_sidecar hashes the written crop

    decision = _make_decision(crop_found=True, crop_rect_xywh=(100, 150, 600, 750))
    _write_sidecar(out_path, input_path, decision)

    sidecar = FrameCropSidecar.model_validate_json(out_path.with_suffix(".json").read_text())
    assert sidecar.crop_found is True
    assert sidecar.crop_rect_xywh == (100, 150, 600, 750)
    assert sidecar.source_image == str(input_path.relative_to(tmp_path))
    assert sidecar.crop_image == str(out_path.relative_to(tmp_path))
    assert sidecar.detection.info == "sam"
    assert sidecar.detection.score == 0.9
    # SHA-256 of the frame bytes is recorded so face_crop can verify its input.
    assert sidecar.sha256 == hashlib.sha256(b"fake-frame-bytes").hexdigest()


def test_write_sidecar_no_crop_persists_full_image_rect_as_identity(tmp_path):
    """When no quad is detected, crop_rect_xywh must round-trip as the full
    image bounds — downstream consumers rely on this to treat 'no crop' as a
    zero offset rather than a special case."""
    input_path = tmp_path / "data" / "raw" / "batch" / "photo.jpg"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"fake")
    out_path = tmp_path / "data" / "steps" / "frame_crop" / "batch" / "photo.jpg"
    out_path.parent.mkdir(parents=True)
    out_path.write_bytes(b"fake-frame-bytes")  # _write_sidecar hashes the written crop

    decision = _make_decision(crop_found=False, crop_rect_xywh=(0, 0, 1000, 800))
    _write_sidecar(out_path, input_path, decision)

    sidecar = FrameCropSidecar.model_validate_json(out_path.with_suffix(".json").read_text())
    assert sidecar.crop_found is False
    assert sidecar.crop_rect_xywh == (0, 0, 1000, 800)
