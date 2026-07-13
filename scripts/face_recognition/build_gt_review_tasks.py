"""Build Label Studio review tasks for ground-truth photos you already annotated.

Unlike `ancestry-face-annotation build` (which proposes *not-yet-annotated*
photos with only face_crop's boxes), this re-proposes photos that are *already*
in ground_truth.json, pre-filled with their existing annotation — box,
person_id, age, face_note and portrait flag — as a Label Studio prediction.

Workflow:
    uv run python scripts/face_recognition/build_gt_review_tasks.py
    # import the written tasks file into Label Studio, then for each task:
    #   - looks right  -> Skip  (normalize ignores Skipped tasks, so the
    #                            existing ground-truth entry is kept untouched)
    #   - needs a fix  -> edit the pre-filled regions, then Submit
    uv run ancestry-face-annotation normalize --export <export.json>

normalize merges by key, so only the photos you actually re-submit overwrite
their old entry; everything you Skip is preserved exactly.

Optionally restrict to a subset of keys (e.g. only the embedding-failure photos)
with --keys-file (one key per line, with or without the .jpg suffix).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from uuid import uuid4

from pipeline.annotation.face_annotation_models import FaceGroundTruthFile
from pipeline.gramps import load_family_tree
from pipeline.gramps.models import Person
from pipeline.shared.paths import CURATED_DIR, LABEL_STUDIO_DIR

GT_PATH = CURATED_DIR / "face_annotation" / "ground_truth.json"
DEFAULT_OUTPUT = LABEL_STUDIO_DIR / "face_annotation" / "review_tasks.json"


def _taxonomy_resolver(persons: list[Person]):
    """Return person_id -> Label Studio taxonomy path (root..leaf), matching the
    nesting that generate_ls_template builds: a surname is a parent node only
    when more than one person shares it. None -> the top-level 'unknown' choice."""
    by_id = {p.id: p for p in persons}
    surname_counts = Counter((p.surname or "?") for p in persons)

    def resolve(person_id: str | None) -> list[str] | None:
        if person_id is None:
            return ["unknown"]
        person = by_id.get(person_id)
        if person is None:
            return None
        surname = person.surname or "?"
        if surname_counts[surname] > 1:
            return [surname, person.choice_label]
        return [person.choice_label]

    return resolve


def _result(region_id: str, w: int, h: int, from_name: str, type_: str, value: dict) -> dict:
    return {
        "id": region_id,
        "type": type_,
        "from_name": from_name,
        "to_name": "image",
        "original_width": w,
        "original_height": h,
        "image_rotation": 0,
        "value": value,
    }


def build_review_tasks(keys: list[str] | None, output: Path) -> int:
    gt = FaceGroundTruthFile.model_validate_json(GT_PATH.read_text(encoding="utf-8"))
    resolve_taxonomy = _taxonomy_resolver(load_family_tree("csv"))

    selected = sorted(gt.items) if keys is None else [k for k in sorted(gt.items) if k in keys]
    if keys is not None:
        missing = sorted(set(keys) - set(gt.items))
        if missing:
            raise SystemExit(
                f"{len(missing)} requested key(s) are not in the ground truth: {missing}"
            )

    tasks: list[dict] = []
    unresolved_pids: set[str] = set()

    for key in selected:
        item = gt.items[key]
        w, h = item.image_size
        results: list[dict] = []

        for face in item.faces:
            rid = uuid4().hex[:10]
            x, y, bw, bh = face.bbox_xywh
            results.append(
                _result(rid, w, h, "face", "rectangle", {
                    "x": x / w * 100, "y": y / h * 100,
                    "width": bw / w * 100, "height": bh / h * 100, "rotation": 0,
                })
            )
            path = resolve_taxonomy(face.person_id)
            if path is None:
                unresolved_pids.add(face.person_id or "")
            else:
                results.append(_result(rid, w, h, "person_id", "taxonomy", {"taxonomy": [path]}))
            if face.age is not None:
                results.append(_result(rid, w, h, "age", "number", {"number": face.age}))
            if face.face_note:
                results.append(_result(rid, w, h, "face_note", "textarea", {"text": [face.face_note]}))
            if face.is_portrait:
                results.append(_result(rid, w, h, "is_portrait", "choices", {"choices": ["portrait"]}))

        if item.note:
            results.append(_result(uuid4().hex[:10], w, h, "note", "textarea", {"text": [item.note]}))

        source = "rotate" if item.image_rel.startswith("steps/rotate/") else "raw"
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={item.image_rel}",
                "key": key,
                "image_rel": item.image_rel,
                "source": source,
            },
            "predictions": [{"model_version": "existing_gt", "result": results}],
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    n_faces = sum(len(gt.items[k].faces) for k in selected)
    print(f"Wrote {len(tasks)} review task(s) ({n_faces} pre-filled face(s)) to {output}")
    if unresolved_pids:
        print(
            f"WARNING: {len(unresolved_pids)} person_id(s) not found in the family tree "
            f"(left unselected in their region): {sorted(unresolved_pids)}"
        )
    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=None,
        help="Optional file with one key per line (with or without .jpg) to limit the review set.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    keys: list[str] | None = None
    if args.keys_file is not None:
        raw = [ln.strip() for ln in args.keys_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        keys = [k if k.endswith(".jpg") else f"{k}.jpg" for k in raw]

    build_review_tasks(keys, args.output)


if __name__ == "__main__":
    main()
