import hashlib
import json

import pytest

from pipeline.face_crop.detector import FaceDetection
from pipeline.face_crop.step import _load_frame_crop_sidecar, _write_sidecar
from pipeline.frame_crop.sidecar import DetectionSidecar, FrameCropSidecar


@pytest.fixture(autouse=True)
def _project_root_is_tmp_path(monkeypatch, tmp_path):
    """_write_sidecar reports paths relative to PROJECT_ROOT; point that at
    tmp_path so test fixtures don't need to live inside the real repo."""
    monkeypatch.setattr("pipeline.shared.paths.PROJECT_ROOT", tmp_path)


def _write_frame_crop_sidecar(frame_path, crop_rect_xywh) -> None:
    sidecar = FrameCropSidecar(
        source_image="data/raw/batch/photo.jpg",
        crop_image="data/steps/frame_crop/batch/photo.jpg",
        crop_found=True,
        crop_rect_xywh=crop_rect_xywh,
        detection=DetectionSidecar(
            info="sam", score=0.9, box_xyxy=None, prompt="rectangular photograph", quad=None
        ),
        sha256="0" * 64,
    )
    frame_path.with_suffix(".json").write_text(sidecar.model_dump_json())


def test_load_frame_crop_sidecar_reads_offset_and_hash(tmp_path):
    frame_path = tmp_path / "batch" / "photo.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"fake")
    _write_frame_crop_sidecar(frame_path, (100, 150, 600, 750))

    sidecar = _load_frame_crop_sidecar(frame_path)
    assert sidecar.crop_rect_xywh[:2] == (100, 150)
    assert sidecar.sha256 == "0" * 64


def test_load_frame_crop_sidecar_missing_sidecar_raises(tmp_path):
    frame_path = tmp_path / "batch" / "photo.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"fake")

    with pytest.raises(FileNotFoundError, match="frame_crop"):
        _load_frame_crop_sidecar(frame_path)


def test_write_sidecar_adds_box_xyxy_source_shifted_by_offset(tmp_path):
    frame_path = tmp_path / "batch" / "photo.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"fake")

    faces = [FaceDetection(box_xyxy=(10.0, 20.0, 50.0, 80.0), score=0.9)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    crop_paths = [out_dir / "face_00.jpg"]
    crop_paths[0].write_bytes(b"fake-crop-bytes")  # _write_sidecar hashes the crop

    sidecar_path = _write_sidecar(
        out_dir, frame_path, "f" * 64, (1000, 1000), "human face", faces, crop_paths,
        frame_offset=(100, 150),
    )

    data = json.loads(sidecar_path.read_text())
    # The verified frame hash is carried forward at the frame level.
    assert data["source_sha256"] == "f" * 64
    face = data["faces"][0]
    assert face["box_xyxy"] == [10.0, 20.0, 50.0, 80.0]
    assert face["box_xyxy_source"] == [110.0, 170.0, 150.0, 230.0]
    # sha256 of b"fake-crop-bytes" is recorded for the crop
    assert face["sha256"] == hashlib.sha256(b"fake-crop-bytes").hexdigest()


def test_write_sidecar_zero_offset_leaves_box_unchanged(tmp_path):
    frame_path = tmp_path / "batch" / "photo.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"fake")

    faces = [FaceDetection(box_xyxy=(10.0, 20.0, 50.0, 80.0), score=0.9)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    crop_paths = [out_dir / "face_00.jpg"]
    crop_paths[0].write_bytes(b"fake-crop-bytes")  # _write_sidecar hashes the crop

    sidecar_path = _write_sidecar(
        out_dir, frame_path, "f" * 64, (1000, 1000), "human face", faces, crop_paths,
        frame_offset=(0, 0),
    )

    data = json.loads(sidecar_path.read_text())
    assert data["faces"][0]["box_xyxy_source"] == data["faces"][0]["box_xyxy"]
