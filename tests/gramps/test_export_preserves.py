"""The merge must never lose anything that was already in the export.

The load-bearing test here is `test_merge_changes_nothing_but_its_own_additions`:
rather than spot-checking fields we happened to think of, it subtracts our known
additions from the output and asserts what remains is identical to the input.
Anything the merge disturbs — an attribute, a note's text, an element's order —
fails it, including things nobody thought to assert.

The other tests pin the specific ways this step could destroy data: the header
Gramps needs to import at all, `replace_media`'s blast radius, and media that
belongs to a citation rather than a person.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pipeline.gramps.annotations import load_face_annotations
from pipeline.gramps.export import GRAMPS_NS, GrampsExport
from pipeline.gramps.plan import Plan, build_plan
from pipeline.gramps.portraits import load_curated_portraits

from tree_fixture import Tree

_NS = f"{{{GRAMPS_NS}}}"


def _plan(tree: Tree, export: GrampsExport, include_faces: str = "all") -> Plan:
    return build_plan(
        annotations=load_face_annotations(tree.ground_truth),
        curated_portraits=load_curated_portraits(tree.portraits_dir),
        known_person_ids=export.person_ids,
        include_faces=include_faces,
        include_curated_portraits=True,
    )


def _merged(tree: Tree, *, replace_media: bool = False) -> tuple[ET.Element, Path]:
    """Run a merge over the fixture export and return (output root, path)."""
    export = GrampsExport.load(tree.export)
    plan = _plan(tree, export)
    if replace_media:
        export.clear_person_media(set(plan.refs_by_person))
    export.add_media(list(plan.media.values()))
    for person_id, refs in plan.refs_by_person.items():
        export.add_person_media_refs(person_id, refs)
    if replace_media:
        export.prune_unreferenced_media()
    export.write(tree.augmented)
    return ET.parse(tree.augmented).getroot(), tree.augmented


def _strip_our_additions(root: ET.Element, plan: Plan, original: ET.Element) -> ET.Element:
    """Remove everything the merge is supposed to have added.

    Additions are identified against the *original*: any media object or objref
    whose handle the input didn't have is ours. What's left must be the input.
    """
    original_media = {obj.get("handle") for obj in original.iter(f"{_NS}object")}

    for parent in root.iter():
        for objref in list(parent.findall(f"{_NS}objref")):
            hlink = objref.get("hlink")
            if hlink not in original_media:
                parent.remove(objref)  # a ref to media we added
                continue
            # A ref to pre-existing media: ours only if the input lacked it.
            owner_id = parent.get("id")
            if owner_id and not _input_had_ref(original, owner_id, objref):
                parent.remove(objref)

    objects = root.find(f"{_NS}objects")
    if objects is not None:
        for obj in list(objects.findall(f"{_NS}object")):
            if obj.get("handle") not in original_media:
                objects.remove(obj)
    return root


def _input_had_ref(original: ET.Element, owner_id: str, objref: ET.Element) -> bool:
    for parent in original.iter():
        if parent.get("id") != owner_id:
            continue
        for existing in parent.findall(f"{_NS}objref"):
            if existing.get("hlink") == objref.get("hlink"):
                return True
    return False


def _canonical(root: ET.Element) -> str:
    return ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)


def test_merge_changes_nothing_but_its_own_additions(tree: Tree) -> None:
    original = ET.parse(tree.export).getroot()
    export = GrampsExport.load(tree.export)
    plan = _plan(tree, export)

    merged_root, _ = _merged(tree)
    remainder = _strip_our_additions(merged_root, plan, ET.parse(tree.export).getroot())

    assert _canonical(remainder) == _canonical(original)


def test_merge_preserves_the_xml_declaration_and_doctype(tree: Tree) -> None:
    _merged(tree)

    original_header = tree.export.read_text(encoding="utf-8").split("<database")[0]
    written_header = tree.augmented.read_text(encoding="utf-8").split("<database")[0]

    assert written_header == original_header
    assert "<!DOCTYPE database PUBLIC" in written_header


def test_merge_keeps_every_person_family_note_source_and_citation(tree: Tree) -> None:
    original = ET.parse(tree.export).getroot()
    merged, _ = _merged(tree)

    for tag in ("person", "family", "note", "source", "citation", "event", "placeobj"):
        before = {e.get("id") for e in original.iter(f"{_NS}{tag}")}
        after = {e.get("id") for e in merged.iter(f"{_NS}{tag}")}
        assert after == before, f"<{tag}> set changed"


def test_merge_leaves_an_unlabelled_person_untouched(tree: Tree) -> None:
    """I0004 has no portrait and no faces — nothing should happen to them."""
    original = ET.parse(tree.export).getroot()
    merged, _ = _merged(tree)

    before = next(p for p in original.iter(f"{_NS}person") if p.get("id") == "I0004")
    after = next(p for p in merged.iter(f"{_NS}person") if p.get("id") == "I0004")

    assert _canonical(after) == _canonical(before)


def test_merge_preserves_umlauts(tree: Tree) -> None:
    """The tree is German: names and notes are full of ä/ö/ü/ß, and a mangled
    re-encode would corrupt them silently."""
    merged, _ = _merged(tree)

    person = next(p for p in merged.iter(f"{_NS}person") if p.get("id") == "I0003")
    surname = person.find(f"{_NS}name/{_NS}surname")
    assert surname is not None and surname.text == "Weiß"
    given = person.find(f"{_NS}name/{_NS}first")
    assert given is not None and given.text == "Jörg"

    note = next(iter(merged.iter(f"{_NS}note")))
    text = note.find(f"{_NS}text")
    assert text is not None and text.text is not None
    assert "Grüße aus Bremen" in text.text

    street = next(iter(merged.iter(f"{_NS}street")))
    assert street.text == "Hauptstraße 1"


def test_replace_media_only_touches_the_people_being_merged(tree: Tree) -> None:
    """I0003's existing objref goes (they're merged); I0004 has none to lose."""
    merged, _ = _merged(tree, replace_media=True)

    original = ET.parse(tree.export).getroot()
    i0003_before = next(p for p in original.iter(f"{_NS}person") if p.get("id") == "I0003")
    assert i0003_before.find(f"{_NS}objref") is not None  # it had one to begin with

    i0003 = next(p for p in merged.iter(f"{_NS}person") if p.get("id") == "I0003")
    hlinks = [r.get("hlink") for r in i0003.findall(f"{_NS}objref")]
    assert "_o0002" not in hlinks  # the pre-existing ref was replaced
    assert hlinks, "I0003 should still carry the refs we added"


def test_replace_media_does_not_prune_media_a_citation_still_references(tree: Tree) -> None:
    """_o0002 loses its person ref under replace_media, but citation C0000 still
    points at it — pruning it would silently drop the cited document."""
    merged, _ = _merged(tree, replace_media=True)

    handles = {obj.get("handle") for obj in merged.iter(f"{_NS}object")}
    assert "_o0002" in handles

    citation = next(c for c in merged.iter(f"{_NS}citation"))
    assert [r.get("hlink") for r in citation.findall(f"{_NS}objref")] == ["_o0002"]


def test_new_media_ids_do_not_collide_with_existing_ones(tree: Tree) -> None:
    """The fixture's highest id is O0007, so ours must start at O0008."""
    merged, _ = _merged(tree)

    ids = [obj.get("id") for obj in merged.iter(f"{_NS}object")]
    assert len(ids) == len(set(ids)), "duplicate media id"
    assert "O0000" in ids and "O0007" in ids  # the originals survive
    assert min(i for i in ids if i not in ("O0000", "O0007")) == "O0008"


def test_output_is_valid_xml_gramps_can_parse(tree: Tree) -> None:
    _, path = _merged(tree)
    # Would raise on malformed output; the DOCTYPE must not confuse the parser.
    root = ET.parse(path).getroot()
    assert root.tag == f"{_NS}database"


@pytest.mark.parametrize("replace_media", [False, True])
def test_no_objref_points_at_a_missing_media_object(tree: Tree, replace_media: bool) -> None:
    """A dangling hlink is the failure mode that makes Gramps refuse an import."""
    merged, _ = _merged(tree, replace_media=replace_media)

    handles = {obj.get("handle") for obj in merged.iter(f"{_NS}object")}
    dangling = [
        r.get("hlink") for r in merged.iter(f"{_NS}objref") if r.get("hlink") not in handles
    ]
    assert dangling == []
