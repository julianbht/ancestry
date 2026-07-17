"""Read, edit and write a Gramps XML export.

Everything that knows about the .gramps file format lives here: the namespace,
the DTD's element ordering, handles, ids. The rest of the step works in terms of
people and media and lets this module worry about where an <objref> is allowed
to sit.

This never touches the live Gramps SQLite database — it edits an *exported*
.gramps and writes a new one alongside it.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

from pipeline.gramps.media import media_handle, mime_type
from pipeline.gramps.plan import MediaItem, MediaRef, Region

GRAMPS_NS = "http://gramps-project.org/xml/1.7.2/"
_NS = f"{{{GRAMPS_NS}}}"

# Person child tags that, per the Gramps DTD, come *after* <objref>. New objrefs
# go immediately before the first of these, so the result stays schema-valid.
# (Gramps' importer is lenient about order, but it keeps the output clean and
# diff-friendly.)
_AFTER_OBJREF = frozenset(
    {
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
)


def _local_tag(element: ET.Element) -> str:
    """Local tag name, without the {namespace} prefix."""
    return element.tag.split("}", 1)[-1]


# A region flattened to its four corners, or None for a whole-image ref. Used to
# tell "this person already has this exact ref" from "this is a new one".
type RegionKey = tuple[int, int, int, int] | None
type RefKey = tuple[str, RegionKey]


def _region_key(region: Region | None) -> RegionKey:
    """A region flattened for comparison against one already in the export."""
    if region is None:
        return None
    return (region.corner1_x, region.corner1_y, region.corner2_x, region.corner2_y)


def _ref_key(objref: ET.Element) -> RefKey:
    """The same key, read back off an objref already in the export, so a planned
    ref can be matched to the element the export already has for it."""
    region_el = objref.find(f"{_NS}region")
    region: RegionKey = None
    if region_el is not None:
        region = (
            int(region_el.get("corner1_x", 0)),
            int(region_el.get("corner1_y", 0)),
            int(region_el.get("corner2_x", 0)),
            int(region_el.get("corner2_y", 0)),
        )
    return (objref.get("hlink", ""), region)


class GrampsExport:
    """An in-memory Gramps XML export, open for editing."""

    def __init__(self, root: ET.Element, header: str) -> None:
        self._root = root
        # The XML declaration + DOCTYPE, kept verbatim: ElementTree drops both
        # on serialisation and Gramps' importer wants them back.
        self._header = header
        self._people = root.find(f"{_NS}people")
        if self._people is None:
            raise ValueError("export has no <people> element")
        self._person_by_id = {
            person.get("id"): person for person in self._people.findall(f"{_NS}person")
        }
        self._objects = self._find_or_create_objects()
        self._change = str(int(time.time()))
        # media key -> handle, for media the export already had under a handle
        # of its own. Ours are derived from the key (media_handle); a reused one
        # keeps whatever handle Gramps gave it, and objrefs must follow.
        self._handle_overrides: dict[str, str] = {}

    @classmethod
    def load(cls, path: Path) -> GrampsExport:
        ET.register_namespace("", GRAMPS_NS)  # emit a default ns, no ns0: prefixes
        text = path.read_text(encoding="utf-8")
        root = ET.fromstring(text)
        return cls(root, header=text[: text.index("<database")])

    def _find_or_create_objects(self) -> ET.Element:
        objects = self._root.find(f"{_NS}objects")
        if objects is not None:
            return objects
        # <objects> belongs after <events> and before <repositories>/<sources>,
        # but Gramps' importer is order-tolerant, so appending is fine.
        return ET.SubElement(self._root, f"{_NS}objects")

    # --- queries ------------------------------------------------------------

    @property
    def person_ids(self) -> set[str]:
        return {pid for pid in self._person_by_id if pid}

    def _media_src_index(self) -> dict[str, ET.Element]:
        """Resolved file src -> media object, so an image the export already
        carries is reused instead of duplicated."""
        index: dict[str, ET.Element] = {}
        for obj in self._objects.findall(f"{_NS}object"):
            file_el = obj.find(f"{_NS}file")
            src = file_el.get("src") if file_el is not None else None
            if src:
                index[src] = obj
        return index

    def _next_media_id(self) -> int:
        """One past the highest existing O#### id, so new ids can't collide."""
        highest = -1
        for obj in self._objects.findall(f"{_NS}object"):
            oid = obj.get("id", "")
            if oid.startswith("O") and oid[1:].isdigit():
                highest = max(highest, int(oid[1:]))
        return highest + 1

    # --- edits --------------------------------------------------------------

    def add_media(self, items: list[MediaItem]) -> list[MediaItem]:
        """Publish media objects for `items`, skipping any the export already
        has. Returns the items that were actually added.

        Reuse is keyed on the resolved file path, so a portrait the export
        already points at keeps its original handle, id and description — we
        attach to it rather than adding a second object for the same file.
        """
        by_src = self._media_src_index()
        next_id = self._next_media_id()
        added: list[MediaItem] = []

        for item in items:
            src = item.path.resolve().as_posix()
            existing = by_src.get(src)
            if existing is not None:
                self._handle_overrides[item.key] = existing.get("handle", "")
                continue
            obj = ET.SubElement(
                self._objects,
                f"{_NS}object",
                {"handle": item.handle, "change": self._change, "id": f"O{next_id:04d}"},
            )
            ET.SubElement(
                obj,
                f"{_NS}file",
                {
                    "src": src,
                    "mime": mime_type(item.path),
                    "description": item.description,
                },
            )
            by_src[src] = obj
            next_id += 1
            added.append(item)
        return added

    def clear_person_media(self, person_ids: set[str]) -> None:
        """Drop every existing media reference on these people."""
        for person_id in person_ids:
            person = self._person_by_id[person_id]
            for ref in person.findall(f"{_NS}objref"):
                person.remove(ref)

    def add_person_media_refs(self, person_id: str, refs: list[MediaRef]) -> int:
        """Lay out a person's media references in the planned order.

        The person's objref block is rebuilt rather than inserted into, because
        priority has to hold over refs the export *already* had — a person whose
        portrait we merged last time carries that ref already, and appending the
        rest in front of it would hand the thumbnail to an annotated photo.

        A ref the person already has is reused as-is (never duplicated), just
        moved into its planned position. Refs the plan says nothing about keep
        their relative order and follow the planned ones. Returns how many refs
        were newly created.
        """
        person = self._person_by_id[person_id]
        unclaimed = list(person.findall(f"{_NS}objref"))
        ordered: list[ET.Element] = []
        added = 0

        for ref in refs:
            key = (self._handle_for(ref.media_key), _region_key(ref.region))
            match = next((el for el in unclaimed if _ref_key(el) == key), None)
            if match is None:
                ordered.append(self._objref_element(ref))
                added += 1
            else:
                unclaimed.remove(match)
                ordered.append(match)

        # Anything the plan didn't mention — media attached in Gramps by hand,
        # or a duplicate the export already carried — survives, after ours.
        ordered.extend(unclaimed)

        for el in person.findall(f"{_NS}objref"):
            person.remove(el)
        self._insert_objrefs(person, ordered)
        return added

    def _handle_for(self, media_key: str) -> str:
        return self._handle_overrides.get(media_key) or media_handle(media_key)

    def _objref_element(self, ref: MediaRef) -> ET.Element:
        element = ET.Element(f"{_NS}objref", {"hlink": self._handle_for(ref.media_key)})
        if ref.region is not None:
            ET.SubElement(
                element,
                f"{_NS}region",
                {
                    "corner1_x": str(ref.region.corner1_x),
                    "corner1_y": str(ref.region.corner1_y),
                    "corner2_x": str(ref.region.corner2_x),
                    "corner2_y": str(ref.region.corner2_y),
                },
            )
        return element

    @staticmethod
    def _insert_objrefs(person: ET.Element, objrefs: list[ET.Element]) -> None:
        """Place a person's whole objref block, in the order given.

        Callers pass the complete block (see add_person_media_refs), so this only
        has to decide where it goes: before the first existing objref if any
        survived, else before the first child the DTD says must follow an objref,
        else at the end. That keeps the output schema-valid and diff-friendly —
        Gramps' importer is order-tolerant, but the file stays clean.
        """
        children = list(person)
        insert_at = len(children)
        for index, child in enumerate(children):
            name = _local_tag(child)
            if name == "objref":
                insert_at = index
                break
            if name in _AFTER_OBJREF:
                insert_at = index
                break
        for offset, ref in enumerate(objrefs):
            person.insert(insert_at + offset, ref)

    def prune_unreferenced_media(self) -> int:
        """Remove media objects nothing points at any more. Returns the count."""
        referenced = {
            ref.get("hlink") for ref in self._root.iter(f"{_NS}objref") if ref.get("hlink")
        }
        pruned = 0
        for obj in list(self._objects.findall(f"{_NS}object")):
            if obj.get("handle") not in referenced:
                self._objects.remove(obj)
                pruned += 1
        return pruned

    # --- output -------------------------------------------------------------

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = ET.tostring(self._root, encoding="unicode")
        path.write_text(self._header + body + "\n", encoding="utf-8")
