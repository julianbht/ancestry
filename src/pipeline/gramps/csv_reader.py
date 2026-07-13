from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path

from pipeline.gramps.models import Person


def _extract_year(date_str: str) -> int | None:
    if not date_str:
        return None
    m = re.search(r"\b(\d{4})\b", date_str)
    return int(m.group(1)) if m else None


class CSVGrampsReader:
    """Reads persons from a Gramps CSV export (multi-section format)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def read_persons(self) -> list[Person]:
        text = self._path.read_text(encoding="utf-8")
        persons: list[Person] = []
        in_person_section = False
        headers: list[str] = []

        for row in csv.reader(StringIO(text)):
            if not row or not any(cell.strip() for cell in row):
                in_person_section = False
                headers = []
                continue

            if row[0].strip() == "Person":
                in_person_section = True
                headers = [h.strip() for h in row]
                continue

            if not in_person_section or not headers:
                continue

            record = dict(zip(headers, [c.strip() for c in row]))
            raw_id = record.get("Person", "").strip("[]")
            if not re.match(r"^I\d+$", raw_id):
                continue

            persons.append(
                Person(
                    id=raw_id,
                    surname=record.get("Surname", ""),
                    given=record.get("Given", ""),
                    gender=record.get("Gender", ""),
                    birth_year=_extract_year(record.get("Birth date", "")),
                )
            )

        return sorted(persons, key=lambda p: (p.surname.casefold(), p.given.casefold()))
