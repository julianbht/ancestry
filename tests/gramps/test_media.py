"""Baking images upright, and the identities derived from a media key.

The orientation behaviour is what keeps a face region lined up with the face:
the annotations are in EXIF-corrected display space, and Gramps renders the raw
pixel buffer without applying the tag. If baking regressed, every region on a
sideways phone photo would silently point at the wrong part of the image.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from pipeline.gramps.annotations import load_face_annotations
from pipeline.gramps.media import bake_upright, media_handle, mime_type

_ORIENTATION = 274


def _make_jpeg(path: Path, size=(100, 200), orientation: int | None = None) -> Path:
    img = Image.new("RGB", size, "red")
    if orientation is None:
        img.save(path)
        return path
    exif = img.getexif()
    exif[_ORIENTATION] = orientation
    img.save(path, exif=exif)
    return path


def test_bake_rotates_a_sideways_photo_and_drops_the_tag(tmp_path: Path) -> None:
    """Orientation 6 means "rotate 90° CW to display", so a 100x200 stored image
    displays as 200x100 — and that's the space the annotations are in."""
    source = _make_jpeg(tmp_path / "in.jpg", size=(100, 200), orientation=6)
    dest = tmp_path / "out" / "in.jpg"

    bake_upright(source, dest, jpeg_quality=95)

    with Image.open(dest) as baked:
        assert baked.size == (200, 100), "orientation was not applied to the pixels"
        assert baked.getexif().get(_ORIENTATION) in (None, 1), "tag survived the bake"


def test_bake_leaves_an_already_upright_photo_the_same_shape(tmp_path: Path) -> None:
    source = _make_jpeg(tmp_path / "in.jpg", size=(100, 200))
    dest = tmp_path / "out" / "in.jpg"

    bake_upright(source, dest, jpeg_quality=95)

    with Image.open(dest) as baked:
        assert baked.size == (100, 200)


def test_bake_creates_missing_parent_directories(tmp_path: Path) -> None:
    source = _make_jpeg(tmp_path / "in.jpg")

    bake_upright(source, tmp_path / "a" / "b" / "c.jpg", jpeg_quality=95)

    assert (tmp_path / "a" / "b" / "c.jpg").exists()


def test_bake_handles_png_without_choking_on_jpeg_options(tmp_path: Path) -> None:
    source = tmp_path / "in.png"
    Image.new("RGB", (10, 10), "blue").save(source)

    bake_upright(source, tmp_path / "out.png", jpeg_quality=95)

    with Image.open(tmp_path / "out.png") as baked:
        assert baked.size == (10, 10)


def test_media_handle_is_stable_and_shaped_like_a_gramps_handle() -> None:
    handle = media_handle("raw/share/photo.jpg")

    assert handle == media_handle("raw/share/photo.jpg")
    assert handle != media_handle("raw/share/other.jpg")
    assert handle.startswith("_") and len(handle) == 29


def test_mime_type_follows_the_suffix() -> None:
    assert mime_type(Path("a.png")) == "image/png"
    assert mime_type(Path("a.PNG")) == "image/png"
    assert mime_type(Path("a.jpg")) == "image/jpeg"
    assert mime_type(Path("a.jpeg")) == "image/jpeg"


def test_annotations_are_flattened_with_unlabelled_faces_counted(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.json"
    path.write_text(
        """
        {"version": 1, "items": {
          "s/b.jpg": {"image_rel": "raw/s/b.jpg", "image_size": [10, 20],
            "faces": [{"bbox_xywh": [1, 2, 3, 4], "person_id": "I0002", "is_portrait": false}]},
          "s/a.jpg": {"image_rel": "raw/s/a.jpg", "image_size": [10, 20],
            "faces": [
              {"bbox_xywh": [1, 2, 3, 4], "person_id": "I0001", "is_portrait": true},
              {"bbox_xywh": [5, 6, 7, 8], "person_id": null, "is_portrait": false}
            ]}
        }}
        """,
        encoding="utf-8",
    )

    result = load_face_annotations(path)

    assert result.unlabelled_count == 1
    # Sorted by photo key, so the merge is reproducible run to run.
    assert [f.image_rel for f in result.faces] == ["raw/s/a.jpg", "raw/s/b.jpg"]
    assert result.faces[0].person_id == "I0001"
    assert result.faces[0].is_portrait is True
    assert result.faces[0].image_size == (10, 20)
    assert result.faces[0].bbox_xywh == (1, 2, 3, 4)
