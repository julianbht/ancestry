"""Builds the throwaway family tree the gramps tests run against.

A miniature of the real thing: an export richer than production (notes, sources,
citations carrying their own media, addresses, umlauts, people with
pre-existing objrefs), hand-cropped portraits, raw photos with real pixels, and
ground-truth labels over them.

Kept out of conftest.py so the test modules can import `Tree` from a module
whose job is obvious, rather than from pytest's fixture-plumbing file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

TEMPLATE = Path(__file__).parent / "tree.gramps.template"

# The photos every test's ground truth is drawn on. Small enough to be free to
# encode, big enough that a region has somewhere to land.
IMAGE_SIZE = (100, 200)

_EXIF_ORIENTATION = 274


@dataclass(frozen=True)
class Tree:
    """Where everything the fixture built lives."""

    root: Path  # the data root (stands in for DATA_DIR)
    export: Path  # data.gramps
    augmented: Path  # where the step writes
    portraits_dir: Path
    ground_truth: Path
    media_dir: Path  # baked output
    document: Path  # the file the citation's media points at
    state_file: Path

    def portrait(self, person_id: str) -> Path:
        return next(self.portraits_dir.glob(f"{person_id}-*"))


def write_png(path: Path, size: tuple[int, int] = (20, 20)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "blue").save(path)
    return path


def write_jpeg(path: Path, orientation: int | None = None) -> Path:
    """A JPEG, optionally carrying an EXIF orientation tag like a phone photo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", IMAGE_SIZE, "red")
    if orientation is None:
        img.save(path)
        return path
    exif = img.getexif()
    exif[_EXIF_ORIENTATION] = orientation
    img.save(path, exif=exif)
    return path


def _face(bbox, person_id, is_portrait) -> dict:
    return {
        "bbox_xywh": list(bbox),
        "person_id": person_id,
        "face_note": None,
        "age": 0,
        "is_portrait": is_portrait,
    }


def _ground_truth() -> dict:
    """Labels covering every branch the plan can take.

    I0000 has a curated portrait *and* an annotated portrait (priority);
    I0002 has one portrait face and one plain face (the include_faces filter);
    I9999 is labelled but absent from the tree (dropped);
    one face has no person_id at all (counted, never attached).
    """
    return {
        "version": 1,
        "items": {
            "share/photo1.jpg": {
                "image_rel": "raw/share/photo1.jpg",
                "image_size": list(IMAGE_SIZE),
                "faces": [
                    _face((10, 20, 30, 40), "I0002", True),
                    _face((50, 60, 20, 20), "I0002", False),
                    _face((0, 0, 10, 10), None, False),
                ],
            },
            "share/photo2.jpg": {
                "image_rel": "raw/share/photo2.jpg",
                "image_size": list(IMAGE_SIZE),
                "faces": [
                    _face((10, 10, 20, 20), "I0003", True),
                    _face((30, 30, 20, 20), "I9999", True),
                    _face((60, 60, 20, 20), "I0000", True),
                ],
            },
        },
    }


def build_tree(tmp_path: Path) -> Tree:
    """Lay the whole data root out under tmp_path."""
    root = tmp_path / "data"
    portraits_dir = root / "gramps" / "portraits"
    database_dir = root / "gramps" / "database"
    database_dir.mkdir(parents=True)

    # Hand-cropped portraits. I0000's is already referenced by the export (see
    # the template); I0001's is new, so the merge has to insert a ref for it.
    portrait_i0000 = write_png(portraits_dir / "I0000-muller-anna.png")
    write_png(portraits_dir / "I0001-muller-hans.png")
    write_png(portraits_dir / "notes-scan.png")  # no person id: must be ignored

    document = write_jpeg(root / "gramps" / "documents" / "cited.jpg")

    write_jpeg(root / "raw" / "share" / "photo1.jpg", orientation=6)  # sideways
    write_jpeg(root / "raw" / "share" / "photo2.jpg")  # already upright

    ground_truth = root / "curated" / "face_annotation" / "ground_truth.json"
    ground_truth.parent.mkdir(parents=True)
    ground_truth.write_text(json.dumps(_ground_truth(), indent=1), encoding="utf-8")

    export = database_dir / "data.gramps"
    export.write_text(
        TEMPLATE.read_text(encoding="utf-8")
        .replace("{{PORTRAIT_I0000_SRC}}", portrait_i0000.resolve().as_posix())
        .replace("{{DOCUMENT_SRC}}", document.resolve().as_posix()),
        encoding="utf-8",
    )

    return Tree(
        root=root,
        export=export,
        augmented=database_dir / "data.augmented.gramps",
        portraits_dir=portraits_dir,
        ground_truth=ground_truth,
        media_dir=root / "gramps" / "media",
        document=document,
        state_file=root / "state" / "gramps.json",
    )
