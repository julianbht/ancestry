"""The step end to end: run() over the fixture tree.

These cover the things only the whole step can show — that a portrait actually
wins the thumbnail slot in the written file, that re-running doesn't duplicate
what it already did, and that a missing export fails with an explanation rather
than a traceback.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pipeline.gramps.config import GrampsConfig
from pipeline.gramps.export import GRAMPS_NS
from pipeline.gramps.step import run

from tests.gramps.tree_fixture import Tree

_NS = f"{{{GRAMPS_NS}}}"


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch):
    """Pin the step's config so tests don't depend on config/gramps/*.yaml."""

    def _configure(**overrides) -> GrampsConfig:
        config = GrampsConfig.model_validate(
            {
                "include_curated_portraits": True,
                "include_faces": "all",
                "replace_media": False,
                "max_files_to_bake": None,
                "ignore_state": False,
                "dry_run": False,
                "jpeg_quality": 95,
            }
            | overrides
        )
        monkeypatch.setattr("pipeline.gramps.step.load_config", lambda *_: config)
        return config

    return _configure


def _person(root: ET.Element, person_id: str) -> ET.Element:
    return next(p for p in root.iter(f"{_NS}person") if p.get("id") == person_id)


def _media_src(root: ET.Element, handle: str | None) -> str:
    obj = next(o for o in root.iter(f"{_NS}object") if o.get("handle") == handle)
    file_el = obj.find(f"{_NS}file")
    assert file_el is not None
    return file_el.get("src", "")


def test_run_writes_an_augmented_export(tree: Tree, configure) -> None:
    configure()

    run()

    assert tree.augmented.exists()
    root = ET.parse(tree.augmented).getroot()
    assert len(list(root.iter(f"{_NS}person"))) == 5


def test_the_hand_cropped_portrait_is_the_first_ref_so_gramps_shows_it(
    tree: Tree, configure
) -> None:
    """I0001 has a portrait and no prior media: ours must land first."""
    configure()

    run()

    root = ET.parse(tree.augmented).getroot()
    first = _person(root, "I0001").find(f"{_NS}objref")
    assert first is not None
    assert _media_src(root, first.get("hlink")) == tree.portrait("I0001").resolve().as_posix()


def test_a_portrait_beats_an_existing_ref_that_was_already_on_the_person(
    tree: Tree, configure
) -> None:
    """I0003 already carries a ref to a cited document. The portrait-less I0003
    only has annotated faces, so the annotated portrait must still come first —
    an appended ref would never be seen as the thumbnail."""
    configure()

    run()

    root = ET.parse(tree.augmented).getroot()
    refs = _person(root, "I0003").findall(f"{_NS}objref")
    assert refs[0].get("hlink") != "_o0002", "the pre-existing ref kept the thumbnail slot"
    assert "_o0002" in [r.get("hlink") for r in refs], "the existing ref was lost"


def test_objrefs_are_inserted_before_the_children_that_must_follow_them(
    tree: Tree, configure
) -> None:
    """I0001 has <address>, <citationref> and <noteref>; the DTD puts objref
    ahead of all three, and Gramps' importer is happier when we honour it."""
    configure()

    run()

    root = ET.parse(tree.augmented).getroot()
    tags = [child.tag.split("}", 1)[-1] for child in _person(root, "I0001")]
    assert tags.index("objref") < tags.index("address")
    assert tags.index("objref") < tags.index("citationref")
    assert tags.index("objref") < tags.index("noteref")


def test_a_portrait_the_export_already_points_at_is_reused_not_duplicated(
    tree: Tree, configure
) -> None:
    """I0000's export already has a media object for their portrait file and an
    objref to it. Re-adding either would give them two identical portraits."""
    configure()

    run()

    root = ET.parse(tree.augmented).getroot()
    portrait_src = tree.portrait("I0000").resolve().as_posix()
    matching = [
        o
        for o in root.iter(f"{_NS}object")
        if (o.find(f"{_NS}file") is not None)
        and o.find(f"{_NS}file").get("src") == portrait_src  # type: ignore[union-attr]
    ]
    assert len(matching) == 1
    assert matching[0].get("handle") == "_o0001", "the original media object was replaced"

    refs = [r.get("hlink") for r in _person(root, "I0000").findall(f"{_NS}objref")]
    assert refs.count("_o0001") == 1


def test_an_already_present_portrait_keeps_the_thumbnail_slot(
    tree: Tree, configure
) -> None:
    """I0000's portrait ref is already in the export, and they also have an
    annotated portrait face. Skipping the duplicate must not let the annotated
    photo be inserted in front of it — that hands Gramps the wrong thumbnail,
    and it's the shape of every person merged on a previous run."""
    configure()

    run()

    root = ET.parse(tree.augmented).getroot()
    refs = _person(root, "I0000").findall(f"{_NS}objref")
    assert len(refs) > 1, "I0000 should have the portrait plus an annotated face"
    assert refs[0].get("hlink") == "_o0001", (
        "the pre-existing hand-cropped portrait must stay first; "
        f"got {_media_src(root, refs[0].get('hlink'))}"
    )


def test_every_person_with_a_hand_cropped_portrait_shows_it_first(
    tree: Tree, configure
) -> None:
    """The whole point of the step, asserted over everyone at once rather than
    one hand-picked person — whether or not the ref was already there."""
    configure()

    run()

    root = ET.parse(tree.augmented).getroot()
    for person_id in ("I0000", "I0001"):
        first = _person(root, person_id).find(f"{_NS}objref")
        assert first is not None, f"{person_id} has no media at all"
        assert (
            _media_src(root, first.get("hlink"))
            == tree.portrait(person_id).resolve().as_posix()
        ), f"{person_id} does not show their hand-cropped portrait first"


def test_running_twice_changes_nothing(tree: Tree, configure) -> None:
    """The second run re-reads the same export, so it must plan the same tree
    and add nothing new — otherwise every run grows the file."""
    configure()

    run()
    first = tree.augmented.read_text(encoding="utf-8")
    run()
    second = tree.augmented.read_text(encoding="utf-8")

    first_root = ET.fromstring(first)
    second_root = ET.fromstring(second)
    assert len(list(second_root.iter(f"{_NS}objref"))) == len(
        list(first_root.iter(f"{_NS}objref"))
    )
    assert len(list(second_root.iter(f"{_NS}object"))) == len(
        list(first_root.iter(f"{_NS}object"))
    )


def test_merging_our_own_output_is_a_no_op(tree: Tree, configure) -> None:
    """The real workflow re-exports from Gramps after importing ours, so the
    step's own additions come back as input. They must not be doubled."""
    configure()
    run()

    # Feed the augmented file back in as the source export.
    tree.augmented.replace(tree.export)
    before = ET.parse(tree.export).getroot()
    run()
    after = ET.parse(tree.augmented).getroot()

    assert len(list(after.iter(f"{_NS}objref"))) == len(list(before.iter(f"{_NS}objref")))
    assert len(list(after.iter(f"{_NS}object"))) == len(list(before.iter(f"{_NS}object")))


def test_photos_are_baked_upright_and_portraits_are_left_alone(
    tree: Tree, configure
) -> None:
    configure()

    run()

    # photo1 is stored sideways with orientation=6; baked it must be 200x100.
    from PIL import Image

    with Image.open(tree.media_dir / "raw" / "share" / "photo1.jpg") as baked:
        assert baked.size == (200, 100)
    # Portraits are referenced where they lie, so nothing is copied for them.
    assert not (tree.media_dir / "portraits").exists()


def test_a_second_run_skips_baking_thanks_to_state(tree: Tree, configure) -> None:
    configure()
    run()
    baked = tree.media_dir / "raw" / "share" / "photo1.jpg"
    baked.write_bytes(b"sentinel")  # would be overwritten by a re-bake

    run()

    assert baked.read_bytes() == b"sentinel"


def test_ignore_state_re_bakes(tree: Tree, configure) -> None:
    configure()
    run()
    baked = tree.media_dir / "raw" / "share" / "photo1.jpg"
    baked.write_bytes(b"sentinel")

    configure(ignore_state=True)
    run()

    assert baked.read_bytes() != b"sentinel"


def test_dry_run_writes_nothing(tree: Tree, configure) -> None:
    configure(dry_run=True)

    run()

    assert not tree.augmented.exists()
    assert not tree.media_dir.exists()


def test_a_missing_export_is_reported_not_raised(tree: Tree, configure, caplog) -> None:
    configure()
    tree.export.unlink()

    run()  # must not raise

    assert not tree.augmented.exists()


def test_faces_land_as_regions_in_percent_of_the_image(tree: Tree, configure) -> None:
    """I0002's portrait face is (10,20,30,40) on a 100x200 photo."""
    configure()

    run()

    root = ET.parse(tree.augmented).getroot()
    regions = [
        r.find(f"{_NS}region")
        for r in _person(root, "I0002").findall(f"{_NS}objref")
    ]
    corners = {
        (
            r.get("corner1_x"),
            r.get("corner1_y"),
            r.get("corner2_x"),
            r.get("corner2_y"),
        )
        for r in regions
        if r is not None
    }
    assert ("10", "10", "40", "30") in corners
