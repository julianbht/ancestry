from typing import Protocol, runtime_checkable

from pipeline.gramps.models import Person


@runtime_checkable
class GrampsReader(Protocol):
    def read_persons(self) -> list[Person]: ...
