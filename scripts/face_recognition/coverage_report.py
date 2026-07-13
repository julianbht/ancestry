"""
Report how well the labeled ground truth (data/curated/face_annotation/ground_truth.json)
covers the family tree (data/gramps/database/family-tree-data.csv).

For each person in the tree, counts how many labeled face crops exist. Helps
answer two questions before building face_recognition's reference gallery:
  - Who do we have many reference faces for (good gallery candidates)?
  - Who is in the tree but has zero or very few labeled faces (gaps — either
    they're not actually in the photo collection, or they are but haven't
    been labeled yet)?

The family tree spans many more people than will ever appear in the photos
(going back generations), so a large "zero faces" list is expected. This
report is meant to help curate a smaller "people likely in the photos" list
by hand, not to be acted on directly.

Writes a CSV to data/debug/face_recognition/coverage_report.csv and prints a
summary to stdout.

Usage:
    uv run python scripts/face_recognition/coverage_report.py
"""

import collections
import csv
import json

from pipeline.gramps import load_family_tree
from pipeline.shared.paths import CURATED_DIR, DEBUG_DIR

GROUND_TRUTH_FILE = CURATED_DIR / "face_annotation" / "ground_truth.json"
REPORT_FILE = DEBUG_DIR / "face_recognition" / "coverage_report.csv"


def _face_counts_by_person() -> tuple[collections.Counter, int]:
    """Labeled face count per Gramps person_id, plus the count of unlabeled
    (person_id=None, i.e. 'unknown') faces."""
    gt = json.loads(GROUND_TRUTH_FILE.read_text())
    counts: collections.Counter = collections.Counter()
    unknown = 0
    for item in gt["items"].values():
        for face in item["faces"]:
            person_id = face["person_id"]
            if person_id is None:
                unknown += 1
            else:
                counts[person_id] += 1
    return counts, unknown


def main() -> None:
    counts, unknown_faces = _face_counts_by_person()
    people = load_family_tree()
    tree_ids = {p.id for p in people}

    stray_ids = set(counts) - tree_ids
    if stray_ids:
        print(f"WARNING: {len(stray_ids)} person_id(s) in ground truth not found in tree: {sorted(stray_ids)}")

    rows = [
        {
            "person_id": p.id,
            "name": p.full_name,
            "birth_year": p.birth_year or "",
            "face_count": counts.get(p.id, 0),
        }
        for p in people
    ]
    rows.sort(key=lambda r: -r["face_count"])

    with_faces = [r for r in rows if r["face_count"] > 0]
    singleton = [r for r in with_faces if r["face_count"] == 1]
    zero = [r for r in rows if r["face_count"] == 0]

    print(f"Tree: {len(people)} people total")
    print(f"  {len(with_faces)} have >=1 labeled face")
    print(f"  {len(singleton)} have exactly 1 labeled face (too few to average for a reference gallery)")
    print(f"  {len(zero)} have 0 labeled faces (not in photos, or not yet labeled)")
    print(f"  {unknown_faces} labeled face(s) marked unknown (no person_id)")

    print("\nTop 10 by face count:")
    for r in rows[:10]:
        print(f"  {r['face_count']:>3}  {r['person_id']}  {r['name']}")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person_id", "name", "birth_year", "face_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull report written to {REPORT_FILE.relative_to(REPORT_FILE.parents[3])}")


if __name__ == "__main__":
    main()
