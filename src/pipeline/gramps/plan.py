"""Decide what to merge into the Gramps export — pure logic, no I/O.

Takes the two label sources (hand-cropped portraits, ground-truth faces) plus
the set of people that actually exist in the export, and produces a plan: which
media objects to create, and which media references to hang on which person, in
which order.

Order is the whole point of the ordering rules below. Gramps uses a person's
*first* media reference as their profile thumbnail, so whatever sorts first here
becomes the face you see when browsing that person. Hand-cropped portraits
therefore always sort ahead of anything derived from the annotations.

Keeping this separate from the XML editing and the image baking means the merge
decisions can be checked without an export, an image, or a filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from pipeline.gramps.annotations import FaceAnnotations
from pipeline.gramps.media import media_handle, media_path
from pipeline.shared.paths import DATA_DIR


class RefKind(IntEnum):
    """Where a media reference came from — and, by value, how it sorts.

    Lower sorts first, and first is what Gramps shows as the profile thumbnail.
    """

    CURATED_PORTRAIT = 0  # hand-cropped, data/gramps/portraits/ — always wins
    GROUND_TRUTH_PORTRAIT = 1  # a face annotated is_portrait
    GROUND_TRUTH_FACE = 2  # any other labelled face


@dataclass(frozen=True)
class Region:
    """A face box in Gramps' integer-percent (0..100) image coordinates."""

    corner1_x: int
    corner1_y: int
    corner2_x: int
    corner2_y: int


@dataclass(frozen=True)
class MediaItem:
    """One image to publish into the export, exactly once."""

    key: str  # stable identity: the state key and the handle seed
    path: Path  # the file the export points at
    description: str
    # The original to bake an upright copy of into `path`, or None to point at
    # `path` as it already is on disk. Only the raw photos need baking; see
    # pipeline.gramps.media.
    bake_from: Path | None

    @property
    def handle(self) -> str:
        return media_handle(self.key)


@dataclass(frozen=True)
class MediaRef:
    """One person's reference to one media item."""

    kind: RefKind
    media_key: str
    region: Region | None  # None = the whole image (a hand-cropped portrait)

    @property
    def sort_key(self) -> tuple[int, str]:
        return (int(self.kind), self.media_key)


@dataclass
class PlanStats:
    curated_portraits: int = 0
    ground_truth_faces: int = 0
    unlabelled_faces: int = 0  # detected but never named — nothing to attach
    skipped_not_portrait: int = 0  # dropped by include_faces="portrait"
    # Labelled or given a portrait, but not in the tree — a ref would dangle.
    unknown_person_ids: set[str] = field(default_factory=set[str])


@dataclass
class Plan:
    media: dict[str, MediaItem]  # key -> item
    refs_by_person: dict[str, list[MediaRef]]  # person id -> refs, best first
    stats: PlanStats

    @property
    def ref_count(self) -> int:
        return sum(len(refs) for refs in self.refs_by_person.values())


def region_percent(
    bbox_xywh: tuple[float, float, float, float], image_size: tuple[int, int]
) -> Region:
    """Convert a pixel bbox to Gramps' integer-percent corners, clamped to 0..100."""
    x, y, w, h = bbox_xywh
    width, height = image_size

    def pct(value: float, total: int) -> int:
        return max(0, min(100, round(value / total * 100)))

    return Region(pct(x, width), pct(y, height), pct(x + w, width), pct(y + h, height))


def _curated_media_key(path: Path) -> str:
    """Portraits get their own key namespace, so a portrait named after a photo
    can never collide with that photo's own media key."""
    return f"portraits/{path.name}"


def build_plan(
    annotations: FaceAnnotations,
    curated_portraits: dict[str, Path],
    known_person_ids: set[str],
    *,
    include_faces: str,
    include_curated_portraits: bool,
) -> Plan:
    """Build the merge plan.

    Args:
        annotations: labelled faces from ground_truth.json.
        curated_portraits: person id -> hand-cropped portrait file.
        known_person_ids: person ids present in the export. Labels for anyone
            else are counted and dropped — a media reference to a person the
            export doesn't have would be a dangling handle.
        include_faces: "portrait" attaches only is_portrait faces; "all"
            attaches every labelled face, so each photo's "people in this image"
            is fully populated.
        include_curated_portraits: whether to attach data/gramps/portraits/.
    """
    media: dict[str, MediaItem] = {}
    refs_by_person: dict[str, list[MediaRef]] = {}
    stats = PlanStats()

    def add_ref(person_id: str, ref: MediaRef) -> None:
        refs_by_person.setdefault(person_id, []).append(ref)

    if include_curated_portraits:
        for person_id, path in sorted(curated_portraits.items()):
            if person_id not in known_person_ids:
                stats.unknown_person_ids.add(person_id)
                continue
            key = _curated_media_key(path)
            # Pointed at in place, not baked: these are hand-made crops with no
            # EXIF orientation to resolve, so a copy would buy nothing and cost
            # the "replace the file, see the new portrait" property.
            media[key] = MediaItem(
                key=key, path=path, description=path.stem, bake_from=None
            )
            # No region: the file is already cropped to the face, so the whole
            # image *is* the portrait.
            add_ref(person_id, MediaRef(RefKind.CURATED_PORTRAIT, key, region=None))
            stats.curated_portraits += 1

    stats.unlabelled_faces = annotations.unlabelled_count
    for face in annotations.faces:
        if include_faces == "portrait" and not face.is_portrait:
            stats.skipped_not_portrait += 1
            continue
        if face.person_id not in known_person_ids:
            stats.unknown_person_ids.add(face.person_id)
            continue
        media.setdefault(
            face.image_rel,
            MediaItem(
                key=face.image_rel,
                path=media_path(face.image_rel),
                description=face.image_rel,
                bake_from=DATA_DIR / face.image_rel,
            ),
        )
        kind = (
            RefKind.GROUND_TRUTH_PORTRAIT
            if face.is_portrait
            else RefKind.GROUND_TRUTH_FACE
        )
        add_ref(
            face.person_id,
            MediaRef(kind, face.image_rel, region_percent(face.bbox_xywh, face.image_size)),
        )
        stats.ground_truth_faces += 1

    for refs in refs_by_person.values():
        refs.sort(key=lambda ref: ref.sort_key)

    return Plan(media=media, refs_by_person=refs_by_person, stats=stats)
