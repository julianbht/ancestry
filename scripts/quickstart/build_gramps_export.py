"""Build a Gramps XML export (.gramps) from the quickstart family CSV.

The quickstart Curie family lives as a Gramps-CSV-format file at
quickstart/data/gramps/database/family-tree-data.csv (people, places, marriages,
children). Gramps can import that CSV directly, but the gramps step
(pipeline.gramps) augments a .gramps *XML* export — and the committed quickstart
tree has no XML yet.

This converter turns the CSV into a minimal-but-valid Gramps XML 1.7.2 export so
the quickstart is self-contained: no Gramps install needed to produce the base
tree. It emits people (with gender, birth name, birth event), places, families
(marriage event + parents + children) — enough for Gramps to show the family and
for the gramps step to attach face regions by matching person id (I0000…).

Output is deterministic (fixed change-times, id-derived handles) so regenerating
it produces a byte-identical file, suitable for committing under quickstart/.

Usage:
    uv run python scripts/quickstart/build_gramps_export.py
    # then, to attach the portraits and annotated faces:
    ANCESTRY_DATA_DIR=quickstart/data uv run ancestry-gramps
"""

import argparse
import csv
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

GRAMPS_NS = "http://gramps-project.org/xml/1.7.2/"
_NS = f"{{{GRAMPS_NS}}}"

# Fixed so regeneration is byte-stable (this file is committed under quickstart/).
_CHANGE = "1700000000"

# The exact header Gramps writes: XML declaration + DOCTYPE. ElementTree drops
# the DOCTYPE, so we prepend it verbatim and let ET serialize the <database> root.
_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE database PUBLIC "-//Gramps//DTD Gramps XML 1.7.2//EN"\n'
    '"http://gramps-project.org/xml/1.7.2/grampsxml.dtd">\n'
)

_GENDER = {"male": "M", "female": "F"}


def _handle(kind: str, key: str) -> str:
    """Deterministic Gramps handle ('_' + 28 hex) from a kind-scoped key."""
    digest = hashlib.sha1(f"{kind}:{key}".encode("utf-8")).hexdigest()
    return "_" + digest[:28]


def _unbracket(cell: str) -> str:
    """'[I0000]' -> 'I0000'; '' -> ''."""
    cell = cell.strip()
    return cell[1:-1] if cell.startswith("[") and cell.endswith("]") else cell


def _parse_csv(path: Path) -> dict:
    """Parse the multi-section Gramps CSV into plain dicts keyed by gramps id."""
    places: dict[str, dict] = {}
    persons: dict[str, dict] = {}
    families: dict[str, dict] = {}

    section = None
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if not row or not any(c.strip() for c in row):
                continue
            head = row[0].strip()
            if head in ("Place", "Person", "Marriage", "Family"):
                section = head
                continue
            if section == "Place":
                places[_unbracket(row[0])] = {"name": row[2].strip(), "type": row[3].strip() or "Unknown"}
            elif section == "Person":
                persons[_unbracket(row[0])] = {
                    "surname": row[1].strip(),
                    "given": row[2].strip(),
                    "gender": row[8].strip(),
                    "birth_date": row[9].strip(),
                    "birth_place": _unbracket(row[10]),
                }
            elif section == "Marriage":
                families.setdefault(_unbracket(row[0]), {"children": []}).update(
                    {
                        "husband": _unbracket(row[1]),
                        "wife": _unbracket(row[2]),
                        "date": row[3].strip(),
                        "place": _unbracket(row[4]),
                    }
                )
            elif section == "Family":
                families.setdefault(_unbracket(row[0]), {"children": []})["children"].append(
                    _unbracket(row[1])
                )
    return {"places": places, "persons": persons, "families": families}


def _sub(parent: ET.Element, tag: str, attrib: dict | None = None) -> ET.Element:
    return ET.SubElement(parent, f"{_NS}{tag}", attrib or {})


def build_xml(data: dict) -> ET.Element:
    places, persons, families = data["places"], data["persons"], data["families"]

    root = ET.Element(f"{_NS}database")
    header = _sub(root, "header")
    _sub(header, "created", {"date": "2026-07-13", "version": "quickstart-csv"})
    _sub(header, "researcher")

    events = _sub(root, "events")
    people = _sub(root, "people")
    families_el = _sub(root, "families")
    places_el = _sub(root, "places")

    # Which family each person is a child of / a parent in.
    child_of: dict[str, str] = {}
    parent_in: dict[str, list[str]] = {pid: [] for pid in persons}
    for fid, fam in families.items():
        for cid in fam["children"]:
            child_of[cid] = fid
        for role in ("husband", "wife"):
            pid = fam.get(role)
            if pid:
                parent_in.setdefault(pid, []).append(fid)

    # --- events (birth per person, marriage per family) ---------------------
    eid = 0
    birth_event: dict[str, str] = {}
    marriage_event: dict[str, str] = {}
    for pid, p in persons.items():
        if p["birth_date"] or p["birth_place"]:
            handle = _handle("event", f"birth:{pid}")
            birth_event[pid] = handle
            ev = _sub(events, "event", {"handle": handle, "change": _CHANGE, "id": f"E{eid:04d}"})
            _sub(ev, "type").text = "Birth"
            if p["birth_date"]:
                _sub(ev, "dateval", {"val": p["birth_date"]})
            if p["birth_place"]:
                _sub(ev, "place", {"hlink": _handle("place", p["birth_place"])})
            eid += 1
    for fid, fam in families.items():
        if fam.get("date") or fam.get("place"):
            handle = _handle("event", f"marriage:{fid}")
            marriage_event[fid] = handle
            ev = _sub(events, "event", {"handle": handle, "change": _CHANGE, "id": f"E{eid:04d}"})
            _sub(ev, "type").text = "Marriage"
            if fam.get("date"):
                _sub(ev, "dateval", {"val": fam["date"]})
            if fam.get("place"):
                _sub(ev, "place", {"hlink": _handle("place", fam["place"])})
            eid += 1

    # --- people -------------------------------------------------------------
    for pid, p in persons.items():
        person = _sub(people, "person", {"handle": _handle("person", pid), "change": _CHANGE, "id": pid})
        _sub(person, "gender").text = _GENDER.get(p["gender"], "U")
        name = _sub(person, "name", {"type": "Birth Name"})
        if p["given"]:
            _sub(name, "first").text = p["given"]
        if p["surname"]:
            _sub(name, "surname").text = p["surname"]
        if pid in birth_event:
            _sub(person, "eventref", {"hlink": birth_event[pid], "role": "Primary"})
        if pid in child_of:
            _sub(person, "childof", {"hlink": _handle("family", child_of[pid])})
        for fid in parent_in.get(pid, []):
            _sub(person, "parentin", {"hlink": _handle("family", fid)})

    # --- families -----------------------------------------------------------
    for fid, fam in families.items():
        family = _sub(families_el, "family", {"handle": _handle("family", fid), "change": _CHANGE, "id": fid})
        _sub(family, "rel", {"type": "Married"})
        if fam.get("husband"):
            _sub(family, "father", {"hlink": _handle("person", fam["husband"])})
        if fam.get("wife"):
            _sub(family, "mother", {"hlink": _handle("person", fam["wife"])})
        if fid in marriage_event:
            _sub(family, "eventref", {"hlink": marriage_event[fid], "role": "Family"})
        for cid in fam["children"]:
            _sub(family, "childref", {"hlink": _handle("person", cid)})

    # --- places -------------------------------------------------------------
    for pid, place in places.items():
        obj = _sub(places_el, "placeobj", {"handle": _handle("place", pid), "change": _CHANGE, "id": pid, "type": place["type"]})
        _sub(obj, "pname", {"value": place["name"]})

    return root


def main() -> None:
    default_csv = Path("quickstart/data/gramps/database/family-tree-data.csv")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=default_csv, help=f"Gramps CSV (default: {default_csv})")
    parser.add_argument("--output", type=Path, default=None, help="Output .gramps (default: <input dir>/data.gramps)")
    args = parser.parse_args()

    output = args.output or args.input.with_name("data.gramps")
    data = _parse_csv(args.input)

    ET.register_namespace("", GRAMPS_NS)  # emit default ns, no ns0: prefixes
    root = build_xml(data)
    body = ET.tostring(root, encoding="unicode")
    output.write_text(_HEADER + body + "\n", encoding="utf-8")

    print(
        f"Wrote {output}\n"
        f"  {len(data['persons'])} people, {len(data['families'])} families, "
        f"{len(data['places'])} places"
    )


if __name__ == "__main__":
    main()
