"""i18n: catalog integrity, the Translator, and locale-aware kinship rendering."""

from __future__ import annotations

import pytest

from web.family_tree import FamilyTree
from web.i18n import DEFAULT_LOCALE, Locale, _catalog, negotiate, translator
from web.kinship import KinshipKind, RelationshipResolver
from web.models import Person


def test_catalogs_share_identical_keys() -> None:
    """The whole point of the fallback chain is that it never fires: every key in
    one locale must exist in the other, so no page silently shows a raw key."""
    de = set(_catalog(Locale.DE))
    en = set(_catalog(Locale.EN))
    assert de == en, {"de_only": de - en, "en_only": en - de}


@pytest.mark.parametrize(
    "raw,expected",
    [("de", Locale.DE), ("en", Locale.EN), (None, DEFAULT_LOCALE), ("xx", DEFAULT_LOCALE)],
)
def test_negotiate(raw: str | None, expected: Locale) -> None:
    assert negotiate(raw) == expected


def test_translator_lookup_interpolation_and_plural() -> None:
    en = translator(Locale.EN)
    assert en("nav.overview") == "Overview"
    assert en("age.years", age=7) == "7 years"
    assert en.plural("photos", 1) == "1 photo"
    assert en.plural("photos", 3) == "3 photos"
    assert en.locale == Locale.EN


def test_unknown_key_falls_back_to_the_key_itself() -> None:
    assert translator(Locale.EN)("no.such.key") == "no.such.key"


def _person(pid: str, gender: str) -> Person:
    return Person(id=pid, surname="X", given=pid, gender=gender, birth_year=None)


def _line_tree() -> FamilyTree:
    """great-grandfather GG -> grandfather G -> father F -> viewer V, plus V's
    sister, uncle (F's brother) and a female first cousin."""
    tree = FamilyTree()
    for pid, gender in [
        ("GG", "male"), ("G", "male"), ("F", "male"), ("V", "male"),
        ("SIS", "female"), ("UNC", "male"), ("COU", "female"),
    ]:
        tree.persons[pid] = _person(pid, gender)
    tree.add_parent_child("GG", "G")
    tree.add_parent_child("G", "F")
    tree.add_parent_child("G", "UNC")  # F's brother
    tree.add_parent_child("F", "V")
    tree.add_parent_child("F", "SIS")  # V's sister
    tree.add_parent_child("UNC", "COU")  # V's cousin
    return tree


@pytest.mark.parametrize(
    "target,kind,degree,de,en",
    [
        ("F", KinshipKind.PARENT, 0, "Vater", "father"),
        ("G", KinshipKind.GRANDPARENT, 0, "Großvater", "grandfather"),
        ("GG", KinshipKind.GRANDPARENT, 1, "Urgroßvater", "great-grandfather"),
        ("SIS", KinshipKind.SIBLING, 0, "Schwester", "sister"),
        ("UNC", KinshipKind.PIBLING, 0, "Onkel", "uncle"),
        ("COU", KinshipKind.COUSIN, 0, "Cousine", "cousin"),
    ],
)
def test_kinship_computed_once_rendered_per_locale(
    target: str, kind: KinshipKind, degree: int, de: str, en: str
) -> None:
    resolver = RelationshipResolver(_line_tree())

    kinship = resolver.resolve("V", target)
    assert kinship is not None
    assert kinship.kind == kind
    assert kinship.degree == degree

    assert resolver.label("V", target, translator(Locale.DE)) == de
    assert resolver.label("V", target, translator(Locale.EN)) == en


def test_self_and_no_relation() -> None:
    resolver = RelationshipResolver(_line_tree())
    assert resolver.label("V", "V", translator(Locale.EN)) == "You"
    assert resolver.label("V", "V", translator(Locale.DE)) == "Du"
    assert resolver.resolve("V", "MISSING") is None
