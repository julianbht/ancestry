"""The merge decisions: what gets attached, to whom, in what order.

build_plan does no I/O, so these run on hand-built inputs rather than the
fixture tree. The ordering tests are the important ones: Gramps shows a person's
first media reference as their profile thumbnail, so "hand-cropped wins" is a
statement about sort order and nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.gramps.annotations import AnnotatedFace, FaceAnnotations
from pipeline.gramps.plan import RefKind, build_plan, region_percent

KNOWN = {"I0001", "I0002"}


def face(person_id: str, *, is_portrait: bool, image: str = "raw/a.jpg") -> AnnotatedFace:
    return AnnotatedFace(
        person_id=person_id,
        image_rel=image,
        image_size=(100, 200),
        bbox_xywh=(10, 20, 30, 40),
        is_portrait=is_portrait,
    )


def annotations(*faces: AnnotatedFace, unlabelled: int = 0) -> FaceAnnotations:
    return FaceAnnotations(faces=list(faces), unlabelled_count=unlabelled)


def plan_for(
    faces: FaceAnnotations,
    curated: dict[str, Path] | None = None,
    *,
    include_faces: str = "all",
    include_curated_portraits: bool = True,
):
    return build_plan(
        annotations=faces,
        curated_portraits=curated or {},
        known_person_ids=KNOWN,
        include_faces=include_faces,
        include_curated_portraits=include_curated_portraits,
    )


# --- ordering: the whole point of the step ---------------------------------


def test_hand_cropped_portrait_outranks_everything_from_the_annotations() -> None:
    plan = plan_for(
        annotations(
            face("I0001", is_portrait=True, image="raw/a.jpg"),
            face("I0001", is_portrait=False, image="raw/b.jpg"),
        ),
        curated={"I0001": Path("portraits/I0001-x.png")},
    )

    kinds = [ref.kind for ref in plan.refs_by_person["I0001"]]
    assert kinds == [
        RefKind.CURATED_PORTRAIT,
        RefKind.GROUND_TRUTH_PORTRAIT,
        RefKind.GROUND_TRUTH_FACE,
    ]


def test_annotated_portrait_outranks_a_plain_face_when_there_is_no_hand_crop() -> None:
    plan = plan_for(
        annotations(
            face("I0001", is_portrait=False, image="raw/a.jpg"),
            face("I0001", is_portrait=True, image="raw/b.jpg"),
        )
    )

    assert plan.refs_by_person["I0001"][0].kind is RefKind.GROUND_TRUTH_PORTRAIT


def test_ordering_is_stable_for_equal_priority_faces() -> None:
    """Two photos, same priority: sorted by key, so the export is reproducible."""
    plan = plan_for(
        annotations(
            face("I0001", is_portrait=True, image="raw/z.jpg"),
            face("I0001", is_portrait=True, image="raw/a.jpg"),
        )
    )

    assert [ref.media_key for ref in plan.refs_by_person["I0001"]] == [
        "raw/a.jpg",
        "raw/z.jpg",
    ]


# --- filtering --------------------------------------------------------------


def test_include_faces_portrait_drops_plain_faces_and_counts_them() -> None:
    plan = plan_for(
        annotations(
            face("I0001", is_portrait=True),
            face("I0001", is_portrait=False, image="raw/b.jpg"),
        ),
        include_faces="portrait",
    )

    assert [ref.kind for ref in plan.refs_by_person["I0001"]] == [
        RefKind.GROUND_TRUTH_PORTRAIT
    ]
    assert plan.stats.skipped_not_portrait == 1


def test_include_faces_all_keeps_plain_faces() -> None:
    plan = plan_for(
        annotations(
            face("I0001", is_portrait=True),
            face("I0001", is_portrait=False, image="raw/b.jpg"),
        ),
        include_faces="all",
    )

    assert len(plan.refs_by_person["I0001"]) == 2
    assert plan.stats.skipped_not_portrait == 0


def test_curated_portraits_can_be_turned_off() -> None:
    plan = plan_for(
        annotations(face("I0001", is_portrait=True)),
        curated={"I0001": Path("portraits/I0001-x.png")},
        include_curated_portraits=False,
    )

    assert plan.stats.curated_portraits == 0
    assert plan.refs_by_person["I0001"][0].kind is RefKind.GROUND_TRUTH_PORTRAIT


# --- dropping what would dangle --------------------------------------------


def test_a_face_labelled_for_someone_not_in_the_tree_is_dropped_and_reported() -> None:
    plan = plan_for(annotations(face("I9999", is_portrait=True)))

    assert plan.refs_by_person == {}
    assert plan.stats.unknown_person_ids == {"I9999"}


def test_a_portrait_for_someone_not_in_the_tree_is_dropped_and_reported() -> None:
    plan = plan_for(
        annotations(), curated={"I9999": Path("portraits/I9999-ghost.png")}
    )

    assert plan.refs_by_person == {}
    assert plan.stats.curated_portraits == 0
    assert plan.stats.unknown_person_ids == {"I9999"}


def test_unlabelled_faces_are_counted_not_attached() -> None:
    plan = plan_for(annotations(face("I0001", is_portrait=True), unlabelled=7))

    assert plan.stats.unlabelled_faces == 7
    assert plan.ref_count == 1


# --- media ------------------------------------------------------------------


def test_one_media_object_per_photo_however_many_faces_it_holds() -> None:
    plan = plan_for(
        annotations(
            face("I0001", is_portrait=True, image="raw/group.jpg"),
            face("I0002", is_portrait=True, image="raw/group.jpg"),
        )
    )

    assert list(plan.media) == ["raw/group.jpg"]
    assert plan.ref_count == 2


def test_a_portrait_named_like_a_photo_gets_its_own_media_key() -> None:
    """Portraits are namespaced, so they can't collide with a photo's key."""
    plan = plan_for(
        annotations(face("I0001", is_portrait=True, image="raw/a.jpg")),
        curated={"I0002": Path("raw/a.jpg")},
    )

    assert set(plan.media) == {"raw/a.jpg", "portraits/a.jpg"}


def test_hand_cropped_portraits_are_pointed_at_in_place_never_baked() -> None:
    portrait = Path("portraits/I0001-x.png")
    plan = plan_for(annotations(), curated={"I0001": portrait})

    item = plan.media["portraits/I0001-x.png"]
    assert item.bake_from is None
    assert item.path == portrait


def test_photos_are_baked_from_the_data_root_into_the_media_dir(monkeypatch) -> None:
    monkeypatch.setattr("pipeline.gramps.plan.DATA_DIR", Path("/data"))
    monkeypatch.setattr("pipeline.gramps.media.MEDIA_DIR", Path("/media"))

    plan = plan_for(annotations(face("I0001", is_portrait=True, image="raw/a.jpg")))

    item = plan.media["raw/a.jpg"]
    assert item.bake_from == Path("/data/raw/a.jpg")
    assert item.path == Path("/media/raw/a.jpg")


def test_media_handles_are_deterministic() -> None:
    first = plan_for(annotations(face("I0001", is_portrait=True)))
    second = plan_for(annotations(face("I0001", is_portrait=True)))

    assert first.media["raw/a.jpg"].handle == second.media["raw/a.jpg"].handle
    assert first.media["raw/a.jpg"].handle.startswith("_")


# --- region maths -----------------------------------------------------------


@pytest.mark.parametrize(
    "bbox, size, expected",
    [
        # a quarter-width box at the origin of a 100x200 image
        ((0, 0, 25, 50), (100, 200), (0, 0, 25, 25)),
        # centred
        ((25, 50, 50, 100), (100, 200), (25, 25, 75, 75)),
        # the whole image
        ((0, 0, 100, 200), (100, 200), (0, 0, 100, 100)),
    ],
)
def test_region_percent_converts_pixels_to_percent(bbox, size, expected) -> None:
    region = region_percent(bbox, size)

    assert (
        region.corner1_x,
        region.corner1_y,
        region.corner2_x,
        region.corner2_y,
    ) == expected


def test_region_percent_clamps_a_box_that_runs_off_the_image() -> None:
    """Ground-truth boxes can sit slightly outside; Gramps only accepts 0..100."""
    region = region_percent((-10, -10, 200, 400), (100, 200))

    assert (region.corner1_x, region.corner1_y) == (0, 0)
    assert (region.corner2_x, region.corner2_y) == (100, 100)
