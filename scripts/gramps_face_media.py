"""Inject annotated faces from ground_truth.json into a Gramps XML export.

Reads the face annotations in data/curated/face_annotation/ground_truth.json and
augments the Gramps XML export (data/gramps/database/data.gramps) so that each
labeled face becomes a Gramps *media reference with a region* on its person:

  - one media <object> per source photo (the raw/rotated image the bbox is drawn
    on), pointing at the real file on disk;
  - for each labeled face, an <objref> on that person carrying a <region> — the
    face bbox converted to Gramps' integer-percent (0..100) coordinates.

Gramps uses the first media-ref-with-region on a person as their profile
thumbnail, so portraits are inserted first: browsing a person shows a cropped
face, and each photo's "people in this image" is populated from the boxes.

This never touches the live Gramps SQLite database. It edits an *exported*
.gramps XML and writes a new augmented .gramps. Because the augmented file is a
strict superset of the export (same handles for the existing 72 people), the
reliable way to load it is to IMPORT IT INTO A FRESH, EMPTY family tree — see
--help epilog. Importing into the existing tree is NOT recommended: Gramps
remaps colliding handles on import, so it duplicates people rather than adding
the new objrefs.

Coordinates: each face's bbox_xywh is in the pixel space of its item's
`image_rel` (`image_size` gives that image's dimensions), so the media file and
the region denominators are always the same image — no pipeline geometry to
replay here.

Usage:
    uv run python scripts/gramps_face_media.py [--faces portrait|all]
                                               [--replace-media]
                                               [--input PATH] [--output PATH]
                                               [--dry-run]

Respects ANCESTRY_DATA_DIR (via pipeline.shared.paths) for the data root.
"""

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from pipeline.shared.paths import CURATED_DIR, DATA_DIR, GRAMPS_DIR, rel

GRAMPS_NS = "http://gramps-project.org/xml/1.7.2/"
_NS = f"{{{GRAMPS_NS}}}"

# Person child tags that, per the Gramps DTD, come *after* <objref>. We insert
# our objrefs immediately before the first of these, so the result stays in a
# schema-valid order (Gramps' importer is lenient about this, but it keeps the
# output clean and diff-friendly).
_AFTER_OBJREF = {
    "address",
    "attribute",
    "url",
    "childof",
    "parentin",
    "personref",
    "noteref",
    "citationref",
    "tagref",
}


def _tag(elem: ET.Element) -> str:
    """Local tag name without the {namespace} prefix."""
    return elem.tag.split("}", 1)[-1]


def _handle_for_image(image_rel: str) -> str:
    """Deterministic, stable media handle derived from the image path.

    Same path -> same handle on every run, so re-augmenting a fresh export never
    produces a different tree for the same inputs.
    """
    digest = hashlib.sha1(image_rel.encode("utf-8")).hexdigest()
    return "_" + digest[:28]  # existing Gramps handles are "_" + 28 hex chars


def _region_percent(
    bbox: list[float], image_size: list[int]
) -> tuple[int, int, int, int]:
    """Convert a pixel bbox_xywh to Gramps integer-percent corner1/corner2."""
    x, y, w, h = bbox
    iw, ih = image_size

    def pct(v: float, total: int) -> int:
        return max(0, min(100, round(v / total * 100)))

    return (
        pct(x, iw),
        pct(y, ih),
        pct(x + w, iw),
        pct(y + h, ih),
    )


def _find_objects_container(root: ET.Element) -> ET.Element:
    """The <objects> element, created (in DTD order) if the export has none."""
    objects = root.find(f"{_NS}objects")
    if objects is not None:
        return objects
    # <objects> sits after <events> and before <repositories>/<sources> in the
    # DTD; appending is fine for Gramps' importer, which is order-tolerant.
    objects = ET.SubElement(root, f"{_NS}objects")
    return objects


def _next_media_id(objects: ET.Element) -> int:
    """One past the highest existing O#### id, so new media ids don't collide."""
    highest = -1
    for obj in objects.findall(f"{_NS}object"):
        oid = obj.get("id", "")
        if oid.startswith("O") and oid[1:].isdigit():
            highest = max(highest, int(oid[1:]))
    return highest + 1


def _insert_objrefs(person: ET.Element, objrefs: list[ET.Element]) -> None:
    """Insert objref elements into a person at the schema-correct position."""
    children = list(person)
    # After the last existing objref if there is one, else before the first
    # "after-objref" child, else at the end.
    insert_at = len(children)
    last_objref_idx = None
    for i, child in enumerate(children):
        name = _tag(child)
        if name == "objref":
            last_objref_idx = i
        elif name in _AFTER_OBJREF and insert_at == len(children):
            insert_at = i
    if last_objref_idx is not None:
        insert_at = last_objref_idx + 1
    for offset, ref in enumerate(objrefs):
        person.insert(insert_at + offset, ref)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Importing the result into Gramps\n"
            "--------------------------------\n"
            "The augmented .gramps is a full superset of your export, so load it\n"
            "into a NEW tree (guarantees exactly what's in the file):\n\n"
            "  1. Gramps -> Family Trees -> Manage Family Trees -> New,\n"
            "     name it e.g. 'ancestry+faces', then Load.\n"
            "  2. With that empty tree open: Family Trees -> Import...,\n"
            "     select the augmented .gramps.\n"
            "  3. Verify a person's gallery shows the cropped face, then keep\n"
            "     using this tree (or export it back over data/gramps/).\n\n"
            "Do NOT import into your existing tree: Gramps remaps colliding\n"
            "handles on import, duplicating all people instead of adding the\n"
            "face regions.\n"
        ),
    )
    parser.add_argument(
        "--faces",
        choices=["portrait", "all"],
        default="portrait",
        help="Which labeled faces to attach: only is_portrait faces (default), "
        "or every face that has a person_id.",
    )
    parser.add_argument(
        "--replace-media",
        action="store_true",
        help="Remove each touched person's existing media references before "
        "adding face regions, then drop any media <object> left with no "
        "references. Default is additive (keeps existing portraits).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=GRAMPS_DIR / "database" / "data.gramps",
        help="Source Gramps XML export (default: data/gramps/database/data.gramps).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GRAMPS_DIR / "database" / "data.augmented.gramps",
        help="Where to write the augmented XML "
        "(default: data/gramps/database/data.augmented.gramps).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change; write nothing.",
    )
    args = parser.parse_args()

    gt_path = CURATED_DIR / "face_annotation" / "ground_truth.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    items = gt["items"]

    ET.register_namespace("", GRAMPS_NS)  # emit default ns, no ns0: prefixes
    tree = ET.parse(args.input)
    root = tree.getroot()

    people_container = root.find(f"{_NS}people")
    if people_container is None:
        raise SystemExit(f"No <people> element in {rel(args.input)}")
    id_to_person = {p.get("id"): p for p in people_container.findall(f"{_NS}person")}

    objects = _find_objects_container(root)
    # Index existing media by resolved file path, so a photo that somehow already
    # has an object reuses it instead of getting a duplicate.
    existing_media_by_src: dict[str, ET.Element] = {}
    for obj in objects.findall(f"{_NS}object"):
        file_el = obj.find(f"{_NS}file")
        if file_el is not None and file_el.get("src"):
            existing_media_by_src[file_el.get("src")] = obj

    change = str(int(time.time()))
    next_id = _next_media_id(objects)

    # Media objects to create, keyed by image_rel (one per photo).
    media_handles: dict[str, str] = {}
    new_media: list[tuple[str, str, str, str]] = []  # (handle, id, src, desc)
    # objrefs to add, grouped per person handle: (is_portrait, media_handle, region)
    per_person: dict[str, list[tuple[bool, str, tuple[int, int, int, int]]]] = {}

    touched_persons: set[str] = set()
    n_faces_attached = 0
    n_skipped_no_person = 0
    n_skipped_not_portrait = 0
    missing_person_ids: set[str] = set()
    missing_files: set[str] = set()

    for key in sorted(items):
        item = items[key]
        image_rel = item["image_rel"]
        image_size = item["image_size"]
        for face in item["faces"]:
            pid = face.get("person_id")
            is_portrait = bool(face.get("is_portrait"))
            if pid is None:
                n_skipped_no_person += 1
                continue
            if args.faces == "portrait" and not is_portrait:
                n_skipped_not_portrait += 1
                continue
            person = id_to_person.get(pid)
            if person is None:
                missing_person_ids.add(pid)
                continue

            # Resolve (or plan) the media object for this photo.
            src = (DATA_DIR / image_rel).resolve().as_posix()
            if not (DATA_DIR / image_rel).exists():
                missing_files.add(image_rel)
            if image_rel not in media_handles:
                if src in existing_media_by_src:
                    media_handles[image_rel] = existing_media_by_src[src].get("handle")
                else:
                    handle = _handle_for_image(image_rel)
                    media_handles[image_rel] = handle
                    new_media.append((handle, f"O{next_id:04d}", src, image_rel))
                    next_id += 1

            region = _region_percent(face["bbox_xywh"], image_size)
            per_person.setdefault(person.get("handle"), []).append(
                (is_portrait, media_handles[image_rel], region)
            )
            touched_persons.add(pid)
            n_faces_attached += 1

    # --- report -------------------------------------------------------------
    print(f"Source export : {rel(args.input)}")
    print(f"Ground truth  : {rel(gt_path)}")
    print(f"Face filter   : {args.faces}")
    print(f"Mode          : {'replace-media' if args.replace_media else 'additive'}")
    print()
    print(
        f"  {n_faces_attached} face refs to attach across "
        f"{len(touched_persons)} people"
    )
    print(f"  {len(new_media)} new media objects (one per photo)")
    print(f"  {n_skipped_no_person} faces skipped (no person_id)")
    if args.faces == "portrait":
        print(f"  {n_skipped_not_portrait} faces skipped (not is_portrait)")
    if missing_person_ids:
        print(
            f"  WARNING: {len(missing_person_ids)} person_id(s) not in tree: "
            f"{', '.join(sorted(missing_person_ids))}"
        )
    if missing_files:
        print(
            f"  WARNING: {len(missing_files)} referenced image file(s) missing "
            f"on disk (media will still be written)"
        )

    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    # --- mutate the tree ----------------------------------------------------
    # Optionally strip existing media refs on the people we touch.
    if args.replace_media:
        for pid in touched_persons:
            person = id_to_person[pid]
            for ref in person.findall(f"{_NS}objref"):
                person.remove(ref)

    # Append new media objects.
    for handle, oid, src, desc in new_media:
        obj = ET.SubElement(
            objects,
            f"{_NS}object",
            {"handle": handle, "change": change, "id": oid},
        )
        ET.SubElement(
            obj,
            f"{_NS}file",
            {"src": src, "mime": "image/jpeg", "description": desc},
        )

    # Add objrefs to each person, portraits first so a portrait becomes the
    # profile thumbnail.
    id_by_handle = {p.get("handle"): p.get("id") for p in id_to_person.values()}
    for person_handle, refs in per_person.items():
        refs.sort(key=lambda r: (not r[0]))  # portraits (True) first
        objrefs: list[ET.Element] = []
        for _is_portrait, media_handle, region in refs:
            ref = ET.Element(f"{_NS}objref", {"hlink": media_handle})
            c1x, c1y, c2x, c2y = region
            ET.SubElement(
                ref,
                f"{_NS}region",
                {
                    "corner1_x": str(c1x),
                    "corner1_y": str(c1y),
                    "corner2_x": str(c2x),
                    "corner2_y": str(c2y),
                },
            )
            objrefs.append(ref)
        _insert_objrefs(id_to_person[id_by_handle[person_handle]], objrefs)

    # Optionally prune media objects that now have zero references anywhere.
    n_pruned = 0
    if args.replace_media:
        referenced = {
            r.get("hlink") for r in root.iter(f"{_NS}objref") if r.get("hlink")
        }
        for obj in list(objects.findall(f"{_NS}object")):
            if obj.get("handle") not in referenced:
                objects.remove(obj)
                n_pruned += 1

    # --- write, preserving the original XML declaration + DOCTYPE ------------
    original = args.input.read_text(encoding="utf-8")
    header = original[: original.index("<database")]
    body = ET.tostring(root, encoding="unicode")
    args.output.write_text(header + body + "\n", encoding="utf-8")

    print(f"\nWrote {rel(args.output)}")
    print(f"  +{len(new_media)} media objects, +{n_faces_attached} face refs")
    if args.replace_media:
        print(f"  pruned {n_pruned} now-unreferenced media object(s)")
    print(
        "\nImport it into a NEW, empty Gramps tree (see --help). Do not import "
        "into your existing tree."
    )


if __name__ == "__main__":
    main()
