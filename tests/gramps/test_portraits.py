"""The hand-cropped portrait index.

This is the lookup that lets you fix someone's portrait by dropping a file in a
folder, so the filename convention is load-bearing: it's matched on the person
id, never the name, and a portrait must survive a rename in the family tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.gramps.portraits import load_curated_portraits, person_id_from_filename


def touch(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"not really an image")
    return path


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("I0004-hergoss-julian.png", "I0004"),
        ("I0004_hergoss.png", "I0004"),
        ("I0004.png", "I0004"),
        ("I12345-someone.jpg", "I12345"),
        # No id prefix: not a portrait we can place.
        ("hergoss-julian.png", None),
        ("Ilse-mueller.png", None),  # starts with I but isn't an id
        ("notes-scan.png", None),
    ],
)
def test_person_id_is_read_from_the_filename_prefix(filename, expected) -> None:
    assert person_id_from_filename(Path(filename)) == expected


def test_index_maps_person_id_to_file(tmp_path: Path) -> None:
    portrait = touch(tmp_path, "I0004-hergoss-julian.png")
    touch(tmp_path, "I0005-hergoss-henrik.jpg")

    index = load_curated_portraits(tmp_path)

    assert index["I0004"] == portrait
    assert set(index) == {"I0004", "I0005"}


def test_files_without_a_person_id_are_ignored(tmp_path: Path) -> None:
    """The folder can hold notes or work-in-progress without breaking lookup."""
    touch(tmp_path, "I0004-hergoss-julian.png")
    touch(tmp_path, "README.md")
    touch(tmp_path, "old-naming-scheme.png")

    assert set(load_curated_portraits(tmp_path)) == {"I0004"}


def test_non_image_files_are_ignored(tmp_path: Path) -> None:
    touch(tmp_path, "I0004-notes.txt")

    assert load_curated_portraits(tmp_path) == {}


def test_suffix_matching_is_case_insensitive(tmp_path: Path) -> None:
    portrait = touch(tmp_path, "I0004-hergoss.PNG")

    assert load_curated_portraits(tmp_path) == {"I0004": portrait}


def test_a_duplicate_id_resolves_the_same_way_every_run(tmp_path: Path) -> None:
    """Two files for one person is a mistake, but it must not make the Gramps
    export nondeterministic — first alphabetically wins, always."""
    first = touch(tmp_path, "I0004-a-version.png")
    touch(tmp_path, "I0004-b-version.png")

    assert load_curated_portraits(tmp_path) == {"I0004": first}
    assert load_curated_portraits(tmp_path) == {"I0004": first}


def test_a_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    """No portraits folder yet is a normal state, not a crash."""
    assert load_curated_portraits(tmp_path / "nope") == {}
