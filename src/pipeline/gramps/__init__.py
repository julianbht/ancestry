from pipeline.gramps.csv_reader import CSVGrampsReader
from pipeline.gramps.models import Person, person_id_from_choice
from pipeline.shared.paths import GRAMPS_DIR

GRAMPS_DB_DIR = GRAMPS_DIR / "database"

__all__ = ["Person", "person_id_from_choice", "load_family_tree"]


def load_family_tree(source: str = "csv") -> list[Person]:
    """Load persons from the family tree database, sorted by surname then given name."""
    if source == "csv":
        return CSVGrampsReader(GRAMPS_DB_DIR / "family-tree-data.csv").read_persons()
    raise ValueError(f"Unknown gramps source: {source!r}. Supported: 'csv'")
