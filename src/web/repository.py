"""Data-access layer: build the in-memory index the web app serves from.

Reads three sources and joins them on `person_id` / photo identity:
  * data/curated/face_annotation/ground_truth.json  — human face labels + ages + the
    `is_portrait` flag used to pick a homepage portrait.
  * data/steps/face_recognition/**/recognition.json — the model's recognised
    faces (computed memberships + a cheap computed age).
  * the Gramps family tree (via web.family_tree) — person names/genders.

Nothing here knows about HTTP or templates; it returns plain domain objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.shared.paths import CURATED_DIR, DATA_DIR, STEPS_DIR

from web.family_tree import FamilyTree, load_family_tree
from web.models import Appearance, LabelSource, Person, PersonProfile

GROUND_TRUTH = CURATED_DIR / "face_annotation" / "ground_truth.json"
FRAME_CROP_DIR = STEPS_DIR / "frame_crop"
FACE_RECOGNITION_DIR = STEPS_DIR / "face_recognition"


@dataclass(frozen=True)
class PortraitSource:
    """A human-flagged portrait face: where to crop it from."""

    image_path: Path           # absolute path to the source photo
    bbox_xywh: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        return self.bbox_xywh[2] * self.bbox_xywh[3]


def _photo_id(share_and_name: str) -> str:
    """'share/20260421_185257.jpg' -> 'share/20260421_185257' (no extension)."""
    return str(Path(share_and_name).with_suffix("")).replace("\\", "/")


class Repository:
    """Loads everything once and answers the queries the routes need."""

    def __init__(self, tree: FamilyTree) -> None:
        self._tree = tree
        self._profiles: dict[str, PersonProfile] = {}
        self._portraits: dict[str, PortraitSource] = {}
        self._build()

    # --- construction -------------------------------------------------------

    def _build(self) -> None:
        # person_id -> {photo_id -> Appearance}; dict keeps one entry per photo
        # and lets human labels take precedence over computed ones.
        memberships: dict[str, dict[str, Appearance]] = {}

        self._load_ground_truth(memberships)
        self._load_recognitions(memberships)

        for pid, by_photo in memberships.items():
            person = self._tree.persons.get(pid) or self._placeholder_person(pid)
            appearances = sorted(
                by_photo.values(), key=lambda a: (a.sort_age, a.photo_id)
            )
            self._profiles[pid] = PersonProfile(person=person, appearances=appearances)

    def _load_ground_truth(self, memberships: dict[str, dict[str, Appearance]]) -> None:
        data = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
        for key, item in data["items"].items():
            pid_photo = _photo_id(key)
            source_image = DATA_DIR / item["image_rel"]
            for face in item["faces"]:
                pid = face.get("person_id")
                if not pid:
                    continue
                age = face.get("age") or None  # treat 0 as "unknown"
                memberships.setdefault(pid, {})[pid_photo] = Appearance(
                    photo_id=pid_photo,
                    membership_source=LabelSource.HUMAN,
                    age=age,
                    age_source=LabelSource.HUMAN if age is not None else None,
                )
                if face.get("is_portrait") and self._better_portrait(pid, face["bbox_xywh"]):
                    self._portraits[pid] = PortraitSource(
                        image_path=source_image,
                        bbox_xywh=tuple(face["bbox_xywh"]),
                    )

    def _better_portrait(self, pid: str, bbox: list[float]) -> bool:
        """Prefer the largest is_portrait crop (best resolution)."""
        existing = self._portraits.get(pid)
        if existing is None:
            return True
        return bbox[2] * bbox[3] > existing.area

    def _load_recognitions(self, memberships: dict[str, dict[str, Appearance]]) -> None:
        for sidecar in FACE_RECOGNITION_DIR.glob("*/*/recognition.json"):
            pid_photo = self._photo_id_from_sidecar(sidecar)
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            for rec in data["recognitions"]:
                if rec.get("status") != "recognized":
                    continue
                pid = rec.get("person_id")
                if not pid:
                    continue
                by_photo = memberships.setdefault(pid, {})
                if pid_photo in by_photo:
                    continue  # a human label already covers this photo
                age = rec.get("age")
                by_photo[pid_photo] = Appearance(
                    photo_id=pid_photo,
                    membership_source=LabelSource.COMPUTED,
                    age=age,
                    age_source=LabelSource.COMPUTED if age is not None else None,
                )

    @staticmethod
    def _photo_id_from_sidecar(sidecar: Path) -> str:
        # .../face_recognition/<share>/<basename>/recognition.json
        return f"{sidecar.parent.parent.name}/{sidecar.parent.name}"

    @staticmethod
    def _placeholder_person(pid: str) -> Person:
        """A recognised id with no entry in the family tree still needs a name."""
        return Person(id=pid, surname="", given=pid, gender="unknown", birth_year=None)

    # --- queries ------------------------------------------------------------

    def profiles(self) -> list[PersonProfile]:
        return list(self._profiles.values())

    def tree_persons(self) -> list[Person]:
        """All individuals in the family tree (the selectable 'who are you' set)."""
        return list(self._tree.persons.values())

    def profile(self, person_id: str) -> PersonProfile | None:
        return self._profiles.get(person_id)

    def portrait_source(self, person_id: str) -> PortraitSource | None:
        return self._portraits.get(person_id)

    def frame_image_path(self, photo_id: str) -> Path:
        """Absolute path to the cropped-print image shown for a photo."""
        return FRAME_CROP_DIR / f"{photo_id}.jpg"
