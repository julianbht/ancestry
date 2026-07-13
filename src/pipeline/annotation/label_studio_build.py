"""Build and normalize Label Studio data for frame-crop annotations."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from pydantic import TypeAdapter

from pipeline.annotation.label_studio_models import (
    BuildOutcome,
    BuildSettings,
    GroundTruthFile,
    GroundTruthItem,
    LabelStudioAnnotation,
    LabelStudioExistingTask,
    LabelStudioExportTask,
    LabelStudioResult,
    LabelStudioTask,
    LabelStudioTaskData,
    NormalizeOutcome,
    NormalizeSettings,
)
from pipeline.shared.paths import DATA_DIR, STEPS_DIR

RAW_DIR = DATA_DIR / "raw"
ROTATED_DIR = STEPS_DIR / "rotate"


def read_cases(paths: list[Path]) -> list[Path]:
    cases: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Cases file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            normalized = Path(entry).as_posix()
            if normalized in seen:
                continue
            seen.add(normalized)
            cases.append(Path(normalized))
    return cases


def resolve_image(rel: Path) -> tuple[Path, str]:
    rotated = ROTATED_DIR / rel
    if rotated.exists():
        return rotated, "rotated"
    raw = RAW_DIR / rel
    if raw.exists():
        return raw, "raw"
    raise FileNotFoundError(f"Image not found in raw/ or rotated/: {rel}")


def ls_image_path(path: Path) -> str:
    rel = path.relative_to(DATA_DIR)
    return f"/data/local-files/?d={rel.as_posix()}"


def load_existing_keys(tasks_file: Path) -> set[str]:
    if not tasks_file.exists():
        raise FileNotFoundError(f"Existing tasks file not found: {tasks_file}")
    data = json.loads(tasks_file.read_text(encoding="utf-8"))
    tasks = TypeAdapter(list[LabelStudioExistingTask]).validate_python(data)
    return {task.data.key for task in tasks if task.data.key}


def annotation_sort_key(annotation: LabelStudioAnnotation) -> str:
    return annotation.updated_at or annotation.created_at or ""


def select_annotation(task: LabelStudioExportTask) -> LabelStudioAnnotation | None:
    annotations = task.annotations
    if not annotations:
        return None
    return max(annotations, key=annotation_sort_key)


def find_polygon_result(
    annotation: LabelStudioAnnotation, label_name: str
) -> LabelStudioResult | None:
    for result in annotation.result:
        if result.type != "polygonlabels":
            continue
        if result.value is None:
            continue
        if label_name in result.value.polygonlabels:
            return result
    return None


def build_tasks(settings: BuildSettings) -> BuildOutcome:
    cases = read_cases(settings.cases_files)
    tasks: list[LabelStudioTask] = []
    missing: list[str] = []
    existing_keys = (
        load_existing_keys(settings.existing_tasks)
        if settings.existing_tasks is not None
        else set()
    )
    skipped_existing = 0

    for rel in cases:
        try:
            image_path, source = resolve_image(rel)
        except FileNotFoundError:
            missing.append(rel.as_posix())
            continue

        key = rel.as_posix()
        if key in existing_keys:
            skipped_existing += 1
            continue

        tasks.append(
            LabelStudioTask(
                data=LabelStudioTaskData(
                    image=ls_image_path(image_path),
                    key=key,
                    image_rel=image_path.relative_to(DATA_DIR).as_posix(),
                    source=source,
                )
            )
        )

    if missing:
        missing_list = "\n  ".join(missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} image(s):\n  {missing_list}"
        )

    settings.tasks_output.parent.mkdir(parents=True, exist_ok=True)
    payload = [task.model_dump(mode="json") for task in tasks]
    settings.tasks_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if skipped_existing:
        print(
            f"Skipped {skipped_existing} task(s) already in {settings.existing_tasks}"
        )
    print(f"Wrote {len(tasks)} task(s) to {settings.tasks_output}")

    return BuildOutcome(tasks_written=len(tasks), skipped_existing=skipped_existing)


def normalize_export(settings: NormalizeSettings, label_name: str) -> NormalizeOutcome:
    if not settings.export_path.exists():
        raise FileNotFoundError(f"Export file not found: {settings.export_path}")
    export = json.loads(settings.export_path.read_text(encoding="utf-8"))
    tasks = TypeAdapter(list[LabelStudioExportTask]).validate_python(export)

    items: dict[str, GroundTruthItem] = {}
    skipped = 0

    for task in tasks:
        key = task.data.key
        if not key:
            skipped += 1
            continue

        source = task.data.source or "raw"
        image_rel = task.data.image_rel
        if not image_rel:
            if source == "rotated":
                image_rel = (Path("rotated") / key).as_posix()
            else:
                image_rel = (Path("raw") / key).as_posix()

        image_path = DATA_DIR / Path(image_rel)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        annotation = select_annotation(task)
        if annotation is None:
            skipped += 1
            continue

        result = find_polygon_result(annotation, label_name)
        if result is None or result.value is None:
            skipped += 1
            continue

        points = result.value.points
        if len(points) != 4:
            raise ValueError(f"{key}: expected 4 polygon points, got {len(points)}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]

        quad = [(p[0] * w / 100, p[1] * h / 100) for p in points]

        items[key] = GroundTruthItem(
            image_rel=image_rel,
            image_size=(w, h),
            quad=quad,
            task_id=task.id,
            annotation_id=annotation.id,
        )

    settings.ground_truth_output.parent.mkdir(parents=True, exist_ok=True)
    output = GroundTruthFile(items=items)
    settings.ground_truth_output.write_text(
        output.model_dump_json(indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(items)} label(s) to {settings.ground_truth_output} ({skipped} skipped)"
    )

    return NormalizeOutcome(labels_written=len(items), skipped=skipped)
