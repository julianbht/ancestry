"""Domain model for the web frontend.

Pure data structures with no I/O. Everything the templates render is one of
these objects; the repository (data access) and services (kinship, portraits)
produce them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class LabelSource(StrEnum):
    """Where a piece of information came from. The frontend always shows this so
    a viewer can tell a human annotation apart from a model's guess."""

    HUMAN = "human"      # from data/curated/face_annotation/ground_truth.json
    COMPUTED = "computed"  # from the face_recognition / model outputs


@dataclass(frozen=True)
class Person:
    """A labelled relative from the Gramps family tree."""

    id: str               # e.g. "I0004"
    surname: str
    given: str
    gender: str           # "male" | "female" | "unknown"
    birth_year: int | None

    @property
    def full_name(self) -> str:
        return f"{self.given} {self.surname}".strip()

    @property
    def first_name(self) -> str:
        """First given name only — for compact card display."""
        return self.given.split()[0] if self.given else self.id


@dataclass(frozen=True)
class Appearance:
    """One person showing up in one photo, with the best age estimate we have."""

    photo_id: str               # "<share>/<basename>", canonical across steps
    membership_source: LabelSource  # how we know they're in this photo
    age: int | None
    age_source: LabelSource | None

    @property
    def sort_age(self) -> int:
        """Age key for ascending sort; unknown ages sink to the end."""
        return self.age if self.age is not None else 10_000


@dataclass
class PersonProfile:
    """A person plus everything the person page needs."""

    person: Person
    appearances: list[Appearance] = field(default_factory=list)

    @property
    def photo_count(self) -> int:
        return len(self.appearances)


@dataclass(frozen=True)
class Relationship:
    """A computed kinship label of one person relative to the chosen viewer."""

    term: str                       # localised label, e.g. "Onkel" / "uncle"
    source: LabelSource = LabelSource.COMPUTED  # derived from the tree → computed
