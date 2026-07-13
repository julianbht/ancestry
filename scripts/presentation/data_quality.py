"""Compute data-quality metrics for the ancestry photo dataset.

The dataset is self-collected (smartphone scans of printed photos), so the
relevant quality questions are not "is a field null" but "how much of the raw
input is actually usable by the pipeline, and where does signal get lost?".
This script reads the artifacts each step already wrote (state files, config
CSVs, and face_recognition sidecars) and reports the standard data-quality
dimensions adapted to an image dataset:

    Completeness  — did each stage extract the signal it was supposed to?
                    (print frame found, ≥1 face found, usable face embedding)
    Validity      — does the input conform to the expected form? (orientation)
    Relevance     — how much input is non-content noise? (back-of-photo scans)
    Uniqueness    — are there exact-duplicate scans?
    Coverage      — gallery size vs. recognition outcome (recognized/unknown)

Nothing here re-runs a model: it aggregates what is already on disk, so it is
cheap and re-runnable. Counts and percentages are printed; pass --out to also
write a Markdown report (e.g. for the presentation).

Usage:
    uv run python scripts/data_quality.py
    uv run python scripts/data_quality.py --out presentation/data_quality.md
    uv run python scripts/data_quality.py --no-duplicates   # skip the hash pass
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STATE_DIR = PROJECT_ROOT / "data" / "state"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FACE_RECOGNITION_DIR = PROJECT_ROOT / "data" / "steps" / "face_recognition"
GROUND_TRUTH_FILE = CURATED_DIR / "face_annotation" / "ground_truth.json"


@dataclass
class Metric:
    """One reported number, optionally as a fraction of a denominator."""

    label: str
    value: int
    total: int | None = None
    note: str = ""

    @property
    def pct(self) -> str:
        if self.total in (None, 0):
            return ""
        return f"{self.value / self.total:.1%}"

    def line(self) -> str:
        head = f"{self.value:>6,}"
        if self.total is not None:
            head += f" / {self.total:<6,} ({self.pct:>6})"
        else:
            head += " " * 18
        tail = f"  {self.label}"
        if self.note:
            tail += f"  — {self.note}"
        return head + tail


@dataclass
class Dimension:
    name: str
    blurb: str
    metrics: list[Metric] = field(default_factory=list)


def _load_state(name: str) -> dict:
    path = STATE_DIR / f"{name}.json"
    return json.loads(path.read_text())["processed"]


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def completeness() -> Dimension:
    dim = Dimension(
        "Completeness",
        "Did each stage extract the signal it was supposed to?",
    )

    # --- frame_crop: physical print detected within the smartphone photo ---
    frame = _load_state("frame_crop")
    n_frames = len(frame)
    frame_found = sum(1 for v in frame.values() if v.get("crop_found"))
    dim.metrics.append(
        Metric("photos with print frame detected", frame_found, n_frames)
    )
    dim.metrics.append(
        Metric(
            "photos with NO frame — full image used",
            n_frames - frame_found,
            n_frames,
            note="cropped too tight / unusual background",
        )
    )

    # --- face_crop: at least one face detected per photo ---
    face = _load_state("face_crop")
    n_photos = len(face)
    faces_per = [
        sum(
            1
            for o in v.get("output", [])
            if os.path.basename(o).startswith("face_") and o.endswith(".jpg")
        )
        for v in face.values()
    ]
    total_faces = sum(faces_per)
    zero_face = sum(1 for n in faces_per if n == 0)
    dim.metrics.append(Metric("photos with ≥1 face detected", n_photos - zero_face, n_photos))
    dim.metrics.append(
        Metric(
            "photos with 0 faces",
            zero_face,
            n_photos,
            note="landscapes / objects / undetectable faces",
        )
    )
    dim.metrics.append(
        Metric(
            "faces extracted total",
            total_faces,
            note=f"mean {total_faces / n_photos:.2f}/photo, max {max(faces_per)}",
        )
    )

    # --- face_recognition: did each detected face yield a usable embedding? ---
    statuses = _recognition_statuses()
    n_rec_faces = sum(statuses.values())
    dim.metrics.append(
        Metric(
            "faces with usable embedding",
            n_rec_faces - statuses["no_embedding"],
            n_rec_faces,
            note="InsightFace re-detected the crop",
        )
    )
    dim.metrics.append(
        Metric(
            "faces with NO embedding",
            statuses["no_embedding"],
            n_rec_faces,
            note="too blurry/small for the recognizer's detector",
        )
    )
    return dim


def validity() -> Dimension:
    dim = Dimension(
        "Validity",
        "Does the raw input conform to the expected upright form?",
    )
    n_raw = len(_load_state("frame_crop"))  # one entry per ingested photo
    rotated = _count_csv_rows(CURATED_DIR / "rotate" / "rotations.csv")
    dim.metrics.append(
        Metric(
            "photos with wrong orientation (corrected)",
            rotated,
            n_raw,
            note="smartphone EXIF rotation quirk",
        )
    )
    return dim


def relevance() -> Dimension:
    dim = Dimension(
        "Relevance",
        "How much ingested input is non-content noise?",
    )
    n_raw = len(_load_state("frame_crop"))
    backs = _count_csv_rows(CURATED_DIR / "photo_backs.csv")
    dim.metrics.append(
        Metric(
            "back-of-photo scans (handwritten notes)",
            backs,
            n_raw,
            note="excluded from face_crop & downstream",
        )
    )
    return dim


def uniqueness(check_duplicates: bool) -> Dimension:
    dim = Dimension(
        "Uniqueness",
        "Are there exact-duplicate scans of the same photo?",
    )
    if not check_duplicates:
        dim.metrics.append(Metric("duplicate check", 0, note="skipped (--no-duplicates)"))
        return dim

    hashes: dict[str, int] = Counter()
    raw = list(RAW_DIR.rglob("*.jpg"))
    for p in raw:
        hashes[hashlib.md5(p.read_bytes()).hexdigest()] += 1
    extra = sum(c - 1 for c in hashes.values() if c > 1)
    dim.metrics.append(Metric("raw scans", len(raw)))
    dim.metrics.append(
        Metric("exact-duplicate files", extra, len(raw), note="byte-identical md5")
    )
    return dim


def coverage() -> Dimension:
    dim = Dimension(
        "Coverage",
        "Label/gallery coverage (the annotated reference set) vs. recognition outcome.",
    )

    # The recognition gallery is built from the Label Studio ground truth
    # (data/curated/face_annotation/ground_truth.json), NOT the curated portraits. Each
    # face with a person_id becomes a reference example; faces left without one
    # are annotated-but-unidentified.
    gt = json.loads(GROUND_TRUTH_FILE.read_text())
    faces = [f for item in gt["items"].values() for f in item["faces"]]
    labeled = [f for f in faces if f.get("person_id")]
    people = {f["person_id"] for f in labeled}
    n_faces = len(faces)
    dim.metrics.append(Metric("annotated frames (ground truth)", len(gt["items"])))
    dim.metrics.append(Metric("annotated faces", n_faces))
    dim.metrics.append(
        Metric("labeled reference faces (gallery)", len(labeled), n_faces, note="have a person_id")
    )
    dim.metrics.append(Metric("distinct known people in gallery", len(people)))
    dim.metrics.append(
        Metric(
            "faces annotated but left unidentified",
            n_faces - len(labeled),
            n_faces,
            note="genuinely unknown to the annotator",
        )
    )

    statuses = _recognition_statuses()
    n = sum(statuses.values())
    dim.metrics.append(Metric("faces recognized as a known person", statuses["recognized"], n))
    dim.metrics.append(
        Metric("faces left unknown", statuses["unknown"], n, note="below match threshold")
    )
    return dim


def _recognition_statuses() -> Counter:
    """Tally per-face recognition status across all sidecars (cached)."""
    if _recognition_statuses._cache is None:  # type: ignore[attr-defined]
        st: Counter = Counter()
        for sidecar in FACE_RECOGNITION_DIR.rglob("recognition.json"):
            data = json.loads(sidecar.read_text())
            for r in data.get("recognitions", []):
                st[r.get("status", "unknown")] += 1
        _recognition_statuses._cache = st  # type: ignore[attr-defined]
    return _recognition_statuses._cache  # type: ignore[attr-defined]


_recognition_statuses._cache = None  # type: ignore[attr-defined]


def render(dims: list[Dimension]) -> str:
    out: list[str] = []
    out.append("=" * 78)
    out.append("DATA QUALITY REPORT — ancestry photo dataset")
    out.append("=" * 78)
    for d in dims:
        out.append("")
        out.append(f"## {d.name}")
        out.append(f"   {d.blurb}")
        out.append("")
        for m in d.metrics:
            out.append("   " + m.line())
    out.append("")
    return "\n".join(out)


def render_markdown(dims: list[Dimension]) -> str:
    out: list[str] = ["# Data Quality Report — ancestry photo dataset", ""]
    for d in dims:
        out.append(f"## {d.name}")
        out.append("")
        out.append(f"*{d.blurb}*")
        out.append("")
        out.append("| Metric | Count | Of total | % |")
        out.append("| --- | ---: | ---: | ---: |")
        for m in d.metrics:
            label = m.label + (f" ({m.note})" if m.note else "")
            total = f"{m.total:,}" if m.total is not None else ""
            out.append(f"| {label} | {m.value:,} | {total} | {m.pct} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="also write a Markdown report to this path")
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="skip the raw-image hashing pass (the only slow part)",
    )
    args = parser.parse_args()

    # The report uses a few non-ASCII glyphs (≥, —); force UTF-8 so it prints
    # on a Windows console (cp1252 by default).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    dims = [
        completeness(),
        validity(),
        relevance(),
        uniqueness(not args.no_duplicates),
        coverage(),
    ]

    print(render(dims))

    if args.out:
        out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(dims), encoding="utf-8")
        print(f"Markdown report written to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
