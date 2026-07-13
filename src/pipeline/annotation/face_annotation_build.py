"""Build and normalize Label Studio data for face annotation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from uuid import uuid4

import cv2
from pydantic import TypeAdapter

from pipeline.annotation.face_annotation_models import (
    FaceAnnotationGT,
    FaceExistingTask,
    FaceGroundTruthFile,
    FaceGroundTruthItem,
    FacePrediction,
    FacePredictionResult,
    FaceTask,
    FaceTaskData,
    LsChoicesValue,
    LsFaceAnnotation,
    LsFaceExportTask,
    LsFaceResult,
    LsNumberValue,
    LsRectangleValue,
    LsTaxonomyValue,
    LsTextAreaValue,
)
from pipeline.face_crop.sidecar import FaceCropSidecar
from pipeline.gramps.models import Person, person_id_from_choice
from pipeline.shared.paths import DATA_DIR, STEPS_DIR

_RAW_DIR = DATA_DIR / "raw"
_ROTATE_DIR = STEPS_DIR / "rotate"
_FACE_CROP_DIR = STEPS_DIR / "face_crop"


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _scan_raw_keys() -> list[Path]:
    """Return all image paths relative to data/raw/, sorted for reproducibility."""
    if not _RAW_DIR.exists():
        return []
    keys = sorted(
        p.relative_to(_RAW_DIR)
        for p in _RAW_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    return keys


def _resolve_image(rel: Path) -> tuple[Path, str]:
    rotated = _ROTATE_DIR / rel
    if rotated.exists():
        return rotated, "rotate"
    raw = _RAW_DIR / rel
    if raw.exists():
        return raw, "raw"
    raise FileNotFoundError(f"Image not found in rotate/ or raw/: {rel}")


def _ls_image_path(path: Path) -> str:
    return f"/data/local-files/?d={path.relative_to(DATA_DIR).as_posix()}"


def _load_existing_keys(tasks_file: Path) -> set[str]:
    if not tasks_file.exists():
        raise FileNotFoundError(f"Existing tasks file not found: {tasks_file}")
    data = json.loads(tasks_file.read_text(encoding="utf-8"))
    tasks = TypeAdapter(list[FaceExistingTask]).validate_python(data)
    return {t.data.key for t in tasks if t.data.key}


def _load_annotated_keys(ground_truth_file: Path) -> set[str]:
    """Keys already present in the ground-truth file (i.e. already annotated)."""
    if not ground_truth_file.exists():
        return set()
    gt = FaceGroundTruthFile.model_validate_json(
        ground_truth_file.read_text(encoding="utf-8")
    )
    return set(gt.items.keys())


def _sidecar_path(rel: Path) -> Path:
    """face_crop sidecar for a raw key: data/steps/face_crop/<key sans ext>/faces.json."""
    return _FACE_CROP_DIR / rel.with_suffix("") / "faces.json"


def _build_prediction(rel: Path, width: int, height: int) -> FacePrediction | None:
    """Turn a frame's face_crop sidecar into pre-drawn rectangles, or None if absent.

    box_xyxy_source is already in the displayed (raw/rotate) image's pixel space,
    so we only convert pixels -> percent of (width, height).
    """
    sidecar_file = _sidecar_path(rel)
    if not sidecar_file.exists():
        return None
    sidecar = FaceCropSidecar.model_validate_json(
        sidecar_file.read_text(encoding="utf-8")
    )
    results: list[FacePredictionResult] = []
    for face in sidecar.faces:
        x1, y1, x2, y2 = face.box_xyxy_source
        results.append(
            FacePredictionResult(
                id=uuid4().hex[:10],
                original_width=width,
                original_height=height,
                value={
                    "x": x1 / width * 100,
                    "y": y1 / height * 100,
                    "width": (x2 - x1) / width * 100,
                    "height": (y2 - y1) / height * 100,
                    "rotation": 0,
                },
            )
        )
    if not results:
        return None
    return FacePrediction(result=results)


def _annotation_sort_key(ann: LsFaceAnnotation) -> str:
    return ann.updated_at or ann.created_at or ""


def _select_annotation(task: LsFaceExportTask) -> LsFaceAnnotation | None:
    candidates = [a for a in task.annotations if not a.was_cancelled]
    if not candidates:
        return None
    return max(candidates, key=_annotation_sort_key)


def _parse_faces(
    results: list[LsFaceResult], image_width: int, image_height: int
) -> list[FaceAnnotationGT]:
    # In this LS export format, all results for a region share the same id.
    # Group by id, then find the rectangle within each group to get the bbox.
    by_id: dict[str, list[LsFaceResult]] = {}
    for r in results:
        by_id.setdefault(r.id, []).append(r)

    faces: list[FaceAnnotationGT] = []
    for region_results in by_id.values():
        rect_result = next((r for r in region_results if r.type == "rectangle"), None)
        if rect_result is None:
            continue
        try:
            rect = LsRectangleValue.model_validate(rect_result.value)
        except Exception:
            continue

        x = rect.x * image_width / 100
        y = rect.y * image_height / 100
        w = rect.width * image_width / 100
        h = rect.height * image_height / 100

        person_id: str | None = None
        face_note: str | None = None
        age: int | None = None
        is_portrait = False

        for r in region_results:
            if r.from_name == "person_id" and r.type == "taxonomy":
                tv = LsTaxonomyValue.model_validate(r.value)
                if tv.taxonomy:
                    leaf_path = max(tv.taxonomy, key=len)
                    person_id = person_id_from_choice(leaf_path[-1])
            elif r.from_name == "face_note" and r.type == "textarea":
                ta = LsTextAreaValue.model_validate(r.value)
                if ta.text:
                    face_note = ta.text[0]
            elif r.from_name == "age" and r.type == "number":
                nv = LsNumberValue.model_validate(r.value)
                if nv.number is not None:
                    age = int(round(nv.number))
            elif r.from_name == "is_portrait" and r.type == "choices":
                cv = LsChoicesValue.model_validate(r.value)
                is_portrait = "portrait" in cv.choices

        faces.append(
            FaceAnnotationGT(bbox_xywh=(x, y, w, h), person_id=person_id, face_note=face_note, age=age, is_portrait=is_portrait)
        )

    return faces


def build_face_tasks(
    tasks_output: Path,
    existing_tasks: Path | None,
    annotated_ground_truth: Path | None = None,
    sample: int | None = None,
    seed: int | None = None,
    include: tuple[str, ...] = (),
) -> tuple[int, int]:
    """Scan data/raw/ and build Label Studio tasks. Returns (written, skipped).

    Each task is pre-annotated with face_crop's detected boxes (as Label Studio
    predictions) so the annotator only labels faces rather than drawing them,
    while still viewing the full source image for context.

    Already-annotated images (those in `annotated_ground_truth`) and any in
    `existing_tasks` are skipped. With `sample`, a random `seed`-reproducible
    subset of the remaining pool is selected. `include` force-adds specific
    keys even if already annotated (e.g. to redo a mistake — the corrected
    annotation overwrites the old one when normalized).
    """
    keys = _scan_raw_keys()
    by_key = {rel.as_posix(): rel for rel in keys}

    skip = set(_load_existing_keys(existing_tasks)) if existing_tasks else set()
    if annotated_ground_truth:
        skip |= _load_annotated_keys(annotated_ground_truth)

    include_keys = list(include)
    missing = [k for k in include_keys if k not in by_key]
    if missing:
        raise ValueError(f"--include key(s) not found under data/raw/: {missing}")

    pool = [rel for rel in keys if rel.as_posix() not in skip and rel.as_posix() not in include_keys]
    skipped = len(keys) - len(pool) - len([k for k in include_keys if k in skip])

    if sample is not None and sample < len(pool):
        pool = random.Random(seed).sample(pool, sample)

    selected = sorted(pool + [by_key[k] for k in include_keys], key=lambda r: r.as_posix())

    tasks: list[FaceTask] = []
    for rel in selected:
        image_path, source = _resolve_image(rel)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h_px, w_px = img.shape[:2]
        prediction = _build_prediction(rel, w_px, h_px)
        tasks.append(
            FaceTask(
                data=FaceTaskData(
                    image=_ls_image_path(image_path),
                    key=rel.as_posix(),
                    image_rel=image_path.relative_to(DATA_DIR).as_posix(),
                    source=source,
                ),
                predictions=[prediction] if prediction else [],
            )
        )

    tasks_output.parent.mkdir(parents=True, exist_ok=True)
    tasks_output.write_text(
        json.dumps([t.model_dump(mode="json") for t in tasks], indent=2), encoding="utf-8"
    )
    with_preds = sum(1 for t in tasks if t.predictions)
    print(
        f"Wrote {len(tasks)} task(s) to {tasks_output} "
        f"({with_preds} pre-annotated from face_crop, {len(tasks) - with_preds} without); "
        f"{skipped} skipped (already annotated)"
        + (f"; force-included {len(include_keys)}" if include_keys else "")
    )
    return len(tasks), skipped


def normalize_face_export(export_path: Path, ground_truth_output: Path) -> tuple[int, int]:
    """Parse a Label Studio export and merge it into the face ground-truth JSON.

    Returns (written, skipped). Items are keyed by raw-relative path, so this
    merges into any existing ground truth: new keys are added and re-annotated
    keys overwrite their previous entry. Nothing in the existing file is lost.
    """
    if not export_path.exists():
        raise FileNotFoundError(f"Export file not found: {export_path}")

    tasks = TypeAdapter(list[LsFaceExportTask]).validate_python(
        json.loads(export_path.read_text(encoding="utf-8"))
    )

    items: dict[str, FaceGroundTruthItem] = {}
    skipped = 0

    for task in tasks:
        key = task.data.key
        if not key:
            skipped += 1
            continue

        annotation = _select_annotation(task)
        if annotation is None:
            skipped += 1
            continue

        image_rel = task.data.image_rel
        source = task.data.source or "raw"
        if not image_rel:
            image_rel = f"steps/rotate/{key}" if source == "rotate" else f"raw/{key}"

        image_path = DATA_DIR / Path(image_rel)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h_px, w_px = img.shape[:2]

        faces = _parse_faces(annotation.result, w_px, h_px)

        note: str | None = None
        for r in annotation.result:
            if r.type == "textarea" and r.from_name == "note":
                tv = LsTextAreaValue.model_validate(r.value)
                if tv.text:
                    note = tv.text[0]
                break

        items[key] = FaceGroundTruthItem(
            image_rel=image_rel,
            image_size=(w_px, h_px),
            faces=faces,
            note=note,
            task_id=task.id,
            annotation_id=annotation.id,
        )

    existing_items: dict[str, FaceGroundTruthItem] = {}
    if ground_truth_output.exists():
        existing_items = dict(
            FaceGroundTruthFile.model_validate_json(
                ground_truth_output.read_text(encoding="utf-8")
            ).items
        )

    added = [k for k in items if k not in existing_items]
    updated = [k for k in items if k in existing_items]
    merged = {**existing_items, **items}
    merged = {k: merged[k] for k in sorted(merged)}

    ground_truth_output.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_output.write_text(
        FaceGroundTruthFile(items=merged).model_dump_json(indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(merged)} item(s) to {ground_truth_output}: "
        f"{len(added)} added, {len(updated)} updated, {skipped} export task(s) skipped "
        f"(no usable annotation)"
    )
    return len(items), skipped


def generate_ls_template(persons: list[Person], output_path: Path) -> None:
    """Generate a Label Studio XML template with persons grouped by surname in a Taxonomy."""
    from itertools import groupby

    lines = ['    <Choice value="unknown"/>']
    by_surname = groupby(persons, key=lambda p: p.surname or "?")
    for surname, group in by_surname:
        members = list(group)
        surname_escaped = surname.replace('"', "&quot;")
        if len(members) == 1:
            label = members[0].choice_label.replace('"', "&quot;")
            lines.append(f'    <Choice value="{label}"/>')
        else:
            lines.append(f'    <Choice value="{surname_escaped}">')
            for person in members:
                label = person.choice_label.replace('"', "&quot;")
                lines.append(f'      <Choice value="{label}"/>')
            lines.append("    </Choice>")

    xml = """\
<View>
  <Header value="Draw a rectangle around each face, then fill in the attributes."/>
  <Image name="image" value="$image" zoom="true" rotateControl="true"/>
  <Rectangle name="face" toName="image" strokeWidth="2"/>
  <Header value="Person"/>
  <Taxonomy name="person_id" toName="image" perRegion="true" placeholder="Select person...">
{choices}
  </Taxonomy>
  <TextArea name="face_note" toName="image" perRegion="true" placeholder="Free note for this face..." rows="1" required="false"/>
  <Header value="Age in photo"/>
  <Number name="age" toName="image" perRegion="true" required="false" placeholder="e.g. 25"/>
  <Header value="Good portrait?"/>
  <Choices name="is_portrait" toName="image" perRegion="true" choice="single" showInline="true">
    <Choice value="portrait"/>
  </Choices>
  <TextArea name="note" toName="image" placeholder="Optional note..." rows="2" required="false"/>
</View>
""".format(choices="\n".join(lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml, encoding="utf-8")
    print(f"Wrote Label Studio template to {output_path} ({len(persons)} persons + unknown)")
