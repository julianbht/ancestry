"""Build a PDF contact sheet: one row of face photos per person, A4 pages.

For each person in the Gramps family tree that we have any photo of, the sheet
shows a single row of face images under their name and lineage. The Gramps
portrait (data/gramps/portraits/) comes first if one exists; the rest of the row
is filled from the face annotations in curated/face_annotation/ground_truth.json,
preferring faces flagged is_portrait, then the largest remaining faces.

A curated JSON file (see MANUAL_PHOTOS below) can pin specific photos to the
front of a person's row when the automatic pick is wrong.

Rotation needs no special handling here: ground_truth's image_rel already points
at data/steps/rotate/ for every photo that has a rotations.csv entry, and its
bbox_xywh coordinates are in EXIF-transposed space (asserted below), so opening
the referenced image and applying exif_transpose() lands in the right frame.

Usage:
    uv run python scripts/presentation/person_sheets.py
    uv run python scripts/presentation/person_sheets.py --photo-size 4 --persons-per-page 4
"""

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps

from pipeline.shared.paths import CURATED_DIR, DATA_DIR, GRAMPS_DIR, rel

GROUND_TRUTH = CURATED_DIR / "face_annotation" / "ground_truth.json"
MANUAL_PHOTOS = CURATED_DIR / "person_sheets" / "photos.json"
PORTRAITS_DIR = GRAMPS_DIR / "portraits"
FAMILY_TREE_CSV = GRAMPS_DIR / "database" / "family-tree-data.csv"
DEFAULT_OUT = GRAMPS_DIR / "graphs" / "person_sheets.pdf"

A4_INCHES = (8.27, 11.69)
CM_PER_INCH = 2.54


@dataclass
class Person:
    """A person from the Gramps CSV, with the dates and relations the sheet prints."""

    person_id: str
    name: str
    dates: dict[str, str] = field(default_factory=dict)  # label -> date, as recorded
    parents: list[str] = field(default_factory=list)
    marriages: list[tuple[str, str]] = field(default_factory=list)  # (spouse id, date)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photo-size", type=float, default=3.0,
                   help="edge length of each square photo cell, in cm (default: 3.0)")
    p.add_argument("--persons-per-page", type=int, default=6,
                   help="rows per A4 page (default: 6)")
    p.add_argument("--photos-per-person", type=int, default=5,
                   help="photos per row (default: 5)")
    p.add_argument("--face-margin", type=float, default=0.25,
                   help="fraction of the face box added around it when cropping (default: 0.25)")
    p.add_argument("--manual", type=Path, default=MANUAL_PHOTOS,
                   help=f"curated manual photo picks (default: {rel(MANUAL_PHOTOS)})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output PDF path (default: {rel(DEFAULT_OUT)})")
    return p.parse_args()


# Date columns of the CSV's Person row, as (label printed on the sheet, index).
PERSON_DATE_COLUMNS = [("Born", 9), ("Baptised", 12), ("Died", 15), ("Buried", 18)]


def load_people() -> dict[str, Person]:
    """Read persons, their dates and their relations from the Gramps CSV export.

    The export concatenates several sections, each introduced by a header row:
    'Person' (one row per person), 'Marriage' (family -> husband, wife, date) and
    'Family' (family -> child, one row per child). Parents and marriages are
    derived by joining the latter two on the family id.
    """
    sections: dict[str, list[list[str]]] = {}
    with FAMILY_TREE_CSV.open(newline="", encoding="utf-8-sig") as f:
        current: list[list[str]] | None = None
        for row in csv.reader(f):
            if not row or not row[0]:
                continue
            if not row[0].startswith("["):
                current = sections.setdefault(row[0], [])
            elif current is not None:
                current.append(row)

    def pid(cell: str) -> str:
        return cell.strip("[]")

    people = {
        pid(r[0]): Person(
            pid(r[0]),
            f"{r[2].strip()} {r[1].strip()}".strip() or pid(r[0]),
            dates={label: r[i].strip() for label, i in PERSON_DATE_COLUMNS if r[i].strip()},
        )
        for r in sections.get("Person", [])
    }

    couples = {pid(r[0]): [pid(c) for c in (r[1], r[2]) if c] for r in sections.get("Marriage", [])}
    for row in sections.get("Marriage", []):
        partners, date = couples[pid(row[0])], row[3].strip()
        for person_id in partners:
            if person_id in people:
                # Appended, not assigned: a person may appear in several marriages.
                people[person_id].marriages += [
                    (spouse, date) for spouse in partners if spouse != person_id
                ]

    for row in sections.get("Family", []):
        child = people.get(pid(row[1]))
        if child is not None:
            child.parents = couples.get(pid(row[0]), [])

    return people


def lineage_line(person: Person, people: dict[str, Person]) -> str:
    """One-line summary: dates, parents, marriages — whichever parts are known.

    Every date is labelled with what it means, since the tree records them
    sparsely and a bare year would be ambiguous.
    """

    def name_of(person_id: str) -> str:
        known = people.get(person_id)
        return known.name if known else person_id

    parts = [f"{label}: {date}" for label, date in person.dates.items()]
    if person.parents:
        parts.append("Child of " + " & ".join(name_of(p) for p in person.parents))
    for spouse, date in person.marriages:
        parts.append(f"Married: {name_of(spouse)}" + (f" ({date})" if date else ""))
    return "  ·  ".join(parts)


def load_portraits() -> dict[str, Path]:
    """Map person id -> Gramps portrait path, keyed off the 'I0004-...' prefix."""
    return {
        path.stem.split("-")[0]: path
        for path in sorted(PORTRAITS_DIR.glob("*"))
        if path.is_file()
    }


def load_faces() -> dict[str, list[dict]]:
    """Map person id -> annotated faces, best candidate first.

    Ordering: is_portrait faces first, then by box area descending — a larger box
    means the face was photographed closer, so the crop carries more detail.
    """
    data = json.loads(GROUND_TRUTH.read_text())
    by_person: dict[str, list[dict]] = {}
    for item in data["items"].values():
        for index, face in enumerate(item["faces"]):
            person_id = face["person_id"]
            if person_id:
                by_person.setdefault(person_id, []).append({**face, "item": item, "index": index})
    for faces in by_person.values():
        faces.sort(key=lambda f: (not f["is_portrait"], -f["bbox_xywh"][2] * f["bbox_xywh"][3]))
    return by_person


def load_manual(path: Path) -> dict[str, list]:
    """Read the curated manual photo picks; an absent file just means none.

    Keys starting with '_' are comments (the file documents its own format).
    """
    if not path.exists():
        return {}
    return {
        person_id: entries
        for person_id, entries in json.loads(path.read_text()).items()
        if not person_id.startswith("_")
    }


def open_source_image(image_rel: str, expected_size: list[int]) -> Image.Image:
    """Open a ground-truth source image in the frame its bbox coordinates use."""
    image = ImageOps.exif_transpose(Image.open(DATA_DIR / image_rel))
    assert image is not None  # exif_transpose only returns None for a None input
    if list(image.size) != expected_size:
        raise ValueError(
            f"{image_rel}: image is {image.size}, but ground truth was annotated "
            f"on {tuple(expected_size)} — bbox coordinates would not line up"
        )
    return image


def crop_face(face: dict, margin: float) -> Image.Image:
    """Crop a square around the annotated face box, with margin, clamped to the image."""
    image = open_source_image(face["item"]["image_rel"], face["item"]["image_size"])
    x, y, w, h = face["bbox_xywh"]
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    box = (
        max(0, round(cx - half)),
        max(0, round(cy - half)),
        min(image.width, round(cx + half)),
        min(image.height, round(cy + half)),
    )
    return image.crop(box)


def manual_photos(person_id: str, entries: list, faces: list[dict],
                  margin: float) -> tuple[list[Image.Image], set[int]]:
    """Resolve one person's curated picks, and report which faces they used up.

    An entry is either a path string (the whole image is used) or
    {"image": <ground-truth image_rel>, "face": <index into its faces>}.
    Returns the images plus the ids of the face dicts already spent, so the
    automatic fill below does not repeat them.
    """
    photos: list[Image.Image] = []
    used: set[int] = set()
    for entry in entries:
        if isinstance(entry, str):
            path = DATA_DIR / entry
            if not path.exists():
                raise FileNotFoundError(f"{person_id}: manual photo not found: {rel(path)}")
            photos.append(Image.open(path).convert("RGB"))
            continue

        match = next(
            (f for f in faces
             if f["item"]["image_rel"] == entry["image"] and f["index"] == entry["face"]),
            None,
        )
        if match is None:
            raise ValueError(
                f"{person_id}: manual entry {entry} matches no face annotated for "
                f"this person in {rel(GROUND_TRUTH)}"
            )
        photos.append(crop_face(match, margin).convert("RGB"))
        used.add(id(match))
    return photos, used


def collect_photos(person_id: str, portrait: Path | None, faces: list[dict], manual: list,
                   limit: int, margin: float) -> list[Image.Image]:
    """Fill a person's row, best-known source first.

    Curated picks, then the Gramps portrait, then the annotated faces.
    """
    photos, used = manual_photos(person_id, manual, faces, margin)
    if portrait is not None and len(photos) < limit:
        photos.append(Image.open(portrait).convert("RGB"))
    for face in faces:
        if len(photos) >= limit:
            break
        if id(face) not in used:
            photos.append(crop_face(face, margin).convert("RGB"))
    return photos[:limit]


def draw_row(fig: plt.Figure, person: Person, lineage: str, photos: list[Image.Image],
             row_index: int, args: argparse.Namespace) -> None:
    """Place one person's heading and photo row onto the figure, in figure fractions."""
    cell_w = args.photo_size / CM_PER_INCH / A4_INCHES[0]
    cell_h = args.photo_size / CM_PER_INCH / A4_INCHES[1]
    row_h = (1.0 - 2 * 0.04) / args.persons_per_page
    row_top = 1.0 - 0.04 - row_index * row_h

    fig.text(0.06, row_top - 0.016, f"{person.name}  [{person.person_id}]",
             fontsize=9, fontweight="bold", va="bottom")
    if lineage:
        fig.text(0.06, row_top - 0.030, lineage, fontsize=6, color="#666666", va="bottom")

    photos_top = row_top - 0.038
    for i, photo in enumerate(photos):
        ax = fig.add_axes((0.06 + i * (cell_w + 0.01), photos_top - cell_h, cell_w, cell_h))
        ax.imshow(photo)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbbbbb")
            spine.set_linewidth(0.8)


def main() -> None:
    args = parse_args()

    people = load_people()
    portraits = load_portraits()
    faces_by_person = load_faces()
    manual = load_manual(args.manual)

    unknown = set(manual) - set(people)
    if unknown:
        raise ValueError(f"{rel(args.manual)}: unknown person id(s): {', '.join(sorted(unknown))}")

    person_ids = sorted(set(portraits) | set(faces_by_person) | set(manual),
                        key=lambda pid: people[pid].name if pid in people else pid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with PdfPages(args.out) as pdf:
        fig = None
        for person_id in person_ids:
            photos = collect_photos(person_id, portraits.get(person_id),
                                    faces_by_person.get(person_id, []),
                                    manual.get(person_id, []),
                                    args.photos_per_person, args.face_margin)
            if not photos:
                continue

            if rows_written % args.persons_per_page == 0:
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
                fig = plt.figure(figsize=A4_INCHES)

            assert fig is not None
            person = people.get(person_id, Person(person_id, "(unknown name)"))
            draw_row(fig, person, lineage_line(person, people), photos,
                     rows_written % args.persons_per_page, args)
            rows_written += 1

        if fig is not None:
            pdf.savefig(fig)
            plt.close(fig)

    pages = -(-rows_written // args.persons_per_page)
    print(f"Done — {rows_written} persons on {pages} page(s): {rel(args.out)}")


if __name__ == "__main__":
    main()
