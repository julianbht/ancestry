"""Parse the Gramps CSV export into people and the kinship edges between them.

This is the data-access layer for the family tree. The CSV (data/gramps/database/
family-tree-data.csv) is a Gramps "CSV export" with several stacked sections,
each introduced by its own header row and separated by blank lines:

    Place,...                       # ignored here
    Person,Surname,Given,...        # one row per individual
    Marriage,Husband,Wife,...       # one row per spousal union (a "family")
    Family,Child                    # links each child to its family

IDs are bracketed in the CSV ("[I0004]") but bare elsewhere ("I0004"); we strip
the brackets so they line up with ground_truth.json's person_id.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.shared.paths import GRAMPS_DIR

from web.models import Person

FAMILY_TREE_CSV = GRAMPS_DIR / "database" / "family-tree-data.csv"


def _strip_id(raw: str) -> str:
    """'[I0004]' -> 'I0004'; passes bare ids through unchanged."""
    return raw.strip().strip("[]")


def _birth_year(raw: str) -> int | None:
    raw = raw.strip()
    if len(raw) >= 4 and raw[:4].isdigit():
        return int(raw[:4])
    return None


@dataclass
class FamilyTree:
    """People plus the three kinship relations needed to name any connection.

    Edges are stored as the minimal primitives; the kinship resolver derives
    everything else (grandparents, cousins, in-laws, …) from these."""

    persons: dict[str, Person] = field(default_factory=dict)
    parents_of: dict[str, set[str]] = field(default_factory=dict)   # child -> parents
    children_of: dict[str, set[str]] = field(default_factory=dict)  # parent -> children
    spouses_of: dict[str, set[str]] = field(default_factory=dict)   # person -> spouses

    def add_parent_child(self, parent: str, child: str) -> None:
        self.parents_of.setdefault(child, set()).add(parent)
        self.children_of.setdefault(parent, set()).add(child)

    def add_spouses(self, a: str, b: str) -> None:
        self.spouses_of.setdefault(a, set()).add(b)
        self.spouses_of.setdefault(b, set()).add(a)


# Section dispatch keyed by the first cell of each header row.
_PERSON_HEADER = "Person"
_MARRIAGE_HEADER = "Marriage"
_FAMILY_HEADER = "Family"


def load_family_tree(csv_path: Path = FAMILY_TREE_CSV) -> FamilyTree:
    tree = FamilyTree()
    # Maps a Gramps "family" id (F####) to its two partners, so the Family/Child
    # section can resolve each child's parents.
    family_partners: dict[str, set[str]] = {}

    with csv_path.open(encoding="utf-8") as fh:
        section: str | None = None
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                section = None  # blank line ends a section
                continue

            first = row[0].strip()
            if first in (_PERSON_HEADER, _MARRIAGE_HEADER, _FAMILY_HEADER):
                section = first
                continue
            if first == "Place":  # the one section we don't use
                section = "Place"
                continue

            if section == _PERSON_HEADER:
                pid = _strip_id(row[0])
                tree.persons[pid] = Person(
                    id=pid,
                    surname=row[1].strip(),
                    given=row[2].strip(),
                    gender=row[8].strip().lower() or "unknown",
                    birth_year=_birth_year(row[9]),
                )
            elif section == _MARRIAGE_HEADER:
                fid = _strip_id(row[0])
                husband, wife = _strip_id(row[1]), _strip_id(row[2])
                partners = {p for p in (husband, wife) if p}
                family_partners[fid] = partners
                if len(partners) == 2:
                    a, b = partners
                    tree.add_spouses(a, b)
            elif section == _FAMILY_HEADER:
                fid = _strip_id(row[0])
                child = _strip_id(row[1])
                for parent in family_partners.get(fid, set()):
                    tree.add_parent_child(parent, child)

    return tree
