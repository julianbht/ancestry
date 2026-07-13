"""Build the per-photo "how did we get here" walkthrough for one person.

Given a photo and the person we are looking at, this reads the sidecar JSON each
pipeline step left behind and assembles an ordered list of steps — original →
(rotated) → frame-cropped → faces detected → identified — that the person page
shows in a modal. Its main job beyond narrating the pipeline is to point at
*which* face in the photo is the current person, which is otherwise hard to tell.

Data sources (all already resolved into the coordinate spaces we need):
  * data/steps/frame_crop/<id>.json          — was a print cropped out, and where
  * data/steps/face_crop/<id>/faces.json     — detected face boxes (frame-local)
  * data/steps/face_recognition/<id>/recognition.json — names/ages/genders
  * data/curated/face_annotation/ground_truth.json — human labels (target fallback)

Nothing here knows about HTTP; it returns plain view-model objects the template
renders. Face boxes are expressed as percentages of the displayed image so the
overlay is resolution-independent and needs no client-side scaling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.shared.paths import CURATED_DIR, STEPS_DIR

from web.family_tree import FamilyTree
from web.i18n import Translator

GROUND_TRUTH = CURATED_DIR / "face_annotation" / "ground_truth.json"
FRAME_CROP_DIR = STEPS_DIR / "frame_crop"
FACE_CROP_DIR = STEPS_DIR / "face_crop"
FACE_RECOGNITION_DIR = STEPS_DIR / "face_recognition"

# Minimum overlap before we trust a human ground-truth box and a detected
# face-crop box to be the same face (used only when recognition didn't match).
_MATCH_IOU = 0.3


@dataclass(frozen=True)
class FaceBox:
    """One detected face, positioned as percentages of the image it overlays."""

    left: float
    top: float
    width: float
    height: float
    label: str | None  # recognised first name, or None when unknown
    is_target: bool  # the person whose page we are on


@dataclass(frozen=True)
class WorkflowStep:
    title: str
    caption: str
    image_url: str | None  # served image to show, or None
    boxes: tuple[FaceBox, ...]  # overlaid on image_url (frame-local %)


@dataclass(frozen=True)
class Workflow:
    person_name: str
    steps: tuple[WorkflowStep, ...]


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# Recognition/tree gender codes -> the i18n key for the word.
_GENDER_KEY = {
    "M": "gender.male",
    "F": "gender.female",
    "male": "gender.male",
    "female": "gender.female",
}


def _gender_label(t: Translator, code: str | None) -> str | None:
    key = _GENDER_KEY.get(code or "")
    return t(key) if key else None


class WorkflowService:
    """Assembles a per-photo, per-person pipeline walkthrough on demand."""

    def __init__(self, tree: FamilyTree) -> None:
        self._tree = tree
        # Loaded once: keyed by "<share>/<name>.jpg" exactly as ground truth stores it.
        self._ground_truth: dict[str, dict] = json.loads(
            GROUND_TRUTH.read_text(encoding="utf-8")
        )["items"]

    def build(self, photo_id: str, person_id: str, t: Translator) -> Workflow:
        """photo_id is '<share>/<name>' (no extension); person_id e.g. 'I0007'."""
        person = self._tree.persons.get(person_id)
        person_name = person.full_name if person else person_id

        frame = self._read_json(FRAME_CROP_DIR / f"{photo_id}.json")
        faces_doc = self._read_json(FACE_CROP_DIR / photo_id / "faces.json")
        recog = self._read_json(FACE_RECOGNITION_DIR / photo_id / "recognition.json")

        rotated = bool(frame) and str(frame.get("source_image", "")).replace(
            "\\", "/"
        ).startswith("data/steps/rotate")

        # face_index -> recognised entry, for naming boxes and the final summary.
        by_index: dict[int, dict] = {
            r["face_index"]: r
            for r in (recog.get("recognitions", []) if recog else [])
            if r.get("status") == "recognized"
        }
        target_index = self._target_face_index(photo_id, person_id, faces_doc, by_index)

        steps: list[WorkflowStep] = [
            WorkflowStep(
                title=self._step_title(t, 1, "workflow.name.original"),
                caption=t("workflow.caption.original"),
                image_url=f"/media/raw/{photo_id}",
                boxes=(),
            )
        ]
        if rotated:
            steps.append(
                WorkflowStep(
                    title=self._step_title(t, 2, "workflow.name.orientation"),
                    caption=t("workflow.caption.orientation"),
                    image_url=f"/media/rotated/{photo_id}",
                    boxes=(),
                )
            )

        crop_found = bool(frame) and frame.get("crop_found")
        steps.append(
            WorkflowStep(
                title=self._step_title(t, len(steps) + 1, "workflow.name.frame"),
                caption=t(
                    "workflow.caption.frame_found"
                    if crop_found
                    else "workflow.caption.frame_none"
                ),
                image_url=f"/media/photo/{photo_id}",
                boxes=(),
            )
        )

        boxes = self._face_boxes(faces_doc, by_index, target_index)
        steps.append(
            WorkflowStep(
                title=self._step_title(t, len(steps) + 1, "workflow.name.faces"),
                caption=self._faces_caption(t, len(boxes)),
                image_url=f"/media/photo/{photo_id}",
                boxes=boxes,
            )
        )

        steps.append(
            self._identify_step(
                t, len(steps) + 1, photo_id, person_id, target_index, by_index
            )
        )

        return Workflow(person_name=person_name, steps=tuple(steps))

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _step_title(t: Translator, number: int, name_key: str) -> str:
        return t("workflow.step_fmt", n=number, title=t(name_key))

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _first_name(self, pid: str | None) -> str | None:
        if not pid:
            return None
        person = self._tree.persons.get(pid)
        return person.first_name if person else pid

    def _target_face_index(
        self,
        photo_id: str,
        person_id: str,
        faces_doc: dict,
        by_index: dict[int, dict],
    ) -> int | None:
        """Which detected face is this person? Prefer recognition; fall back to
        matching the human ground-truth box by overlap."""
        for idx, rec in by_index.items():
            if rec.get("person_id") == person_id:
                return idx

        gt_box = self._ground_truth_box(photo_id, person_id)
        if gt_box is None:
            return None
        best_idx, best_iou = None, _MATCH_IOU
        for face in faces_doc.get("faces", []):
            src = face.get("box_xyxy_source")
            if not src:
                continue
            score = _iou(gt_box, tuple(src))
            if score >= best_iou:
                best_idx, best_iou = face["index"], score
        return best_idx

    def _ground_truth_face(self, photo_id: str, person_id: str) -> dict | None:
        """The human-labelled face for this person on this photo, if any."""
        item = self._ground_truth.get(f"{photo_id}.jpg")
        if not item:
            return None
        for face in item.get("faces", []):
            if face.get("person_id") == person_id:
                return face
        return None

    def _ground_truth_box(
        self, photo_id: str, person_id: str
    ) -> tuple[float, float, float, float] | None:
        face = self._ground_truth_face(photo_id, person_id)
        if face is None:
            return None
        x, y, w, h = face["bbox_xywh"]
        return (x, y, x + w, y + h)

    def _face_boxes(
        self,
        faces_doc: dict,
        by_index: dict[int, dict],
        target_index: int | None,
    ) -> tuple[FaceBox, ...]:
        size = faces_doc.get("image_size")
        if not size:
            return ()
        w, h = size
        out: list[FaceBox] = []
        for face in faces_doc.get("faces", []):
            idx = face["index"]
            x1, y1, x2, y2 = face["box_xyxy"]
            rec = by_index.get(idx)
            out.append(
                FaceBox(
                    left=x1 / w * 100,
                    top=y1 / h * 100,
                    width=(x2 - x1) / w * 100,
                    height=(y2 - y1) / h * 100,
                    label=self._first_name(rec.get("person_id") if rec else None),
                    is_target=idx == target_index,
                )
            )
        return tuple(out)

    @staticmethod
    def _faces_caption(t: Translator, n: int) -> str:
        if n == 0:
            return t("workflow.faces.none")
        if n == 1:
            return t("workflow.faces.one")
        return t("workflow.faces.other", n=n)

    def _identify_step(
        self,
        t: Translator,
        number: int,
        photo_id: str,
        person_id: str,
        target_index: int | None,
        by_index: dict[int, dict],
    ) -> WorkflowStep:
        person = self._tree.persons.get(person_id)
        name = person.full_name if person else person_id
        image_url = (
            f"/media/facecrop/{photo_id}/{target_index:02d}"
            if target_index is not None
            else None
        )

        gt_face = self._ground_truth_face(photo_id, person_id)
        rec = by_index.get(target_index) if target_index is not None else None
        rec = rec if (rec and rec.get("person_id") == person_id) else None

        if gt_face is not None:
            # A human label exists — it wins over the model, exactly as the detail
            # page does (membership and age both come from ground truth there).
            bits = [name]
            age = gt_face.get("age") or None  # 0 means "unknown"
            if age:
                bits.append(t("age.years", age=age))
            gender = _gender_label(t, person.gender if person else None)
            if gender:
                bits.append(gender)
            caption = " · ".join(bits) + " " + t("workflow.identified.confirmed_suffix")
        elif rec is not None:
            # No human label — fall back to the model's identification.
            bits = [name]
            if rec.get("age") is not None:
                bits.append(t("workflow.age.approx", age=rec["age"]))
            gender = _gender_label(t, rec.get("gender"))
            if gender:
                bits.append(gender)
            caption = " · ".join(bits) + " " + t("workflow.identified.ai_suffix")
        else:
            caption = t("workflow.identified.unmatched", name=name)
            image_url = None

        return WorkflowStep(
            title=self._step_title(t, number, "workflow.name.identified"),
            caption=caption,
            image_url=image_url,
            boxes=(),
        )
