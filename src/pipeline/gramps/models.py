from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Person:
    id: str          # Gramps handle, e.g. "I0004"
    surname: str
    given: str
    gender: str      # "male" | "female" | ""
    birth_year: int | None

    @property
    def full_name(self) -> str:
        parts = [self.given, self.surname]
        return " ".join(p for p in parts if p)

    @property
    def choice_label(self) -> str:
        """Label embedded in Label Studio XML — encodes ID for round-trip parsing."""
        name = self.full_name or "(unnamed)"
        if self.birth_year:
            return f"{self.id}: {name} (b. {self.birth_year})"
        return f"{self.id}: {name}"


def person_id_from_choice(value: str) -> str | None:
    """Extract Gramps person ID from a choice label value, or None for 'unknown'."""
    if value == "unknown":
        return None
    m = re.match(r"^(I\d+):", value)
    return m.group(1) if m else None
