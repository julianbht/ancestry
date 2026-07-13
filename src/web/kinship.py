"""Work out a person's kinship relation to a chosen viewer ("Wer bist du?" /
"Who are you?").

Pure domain logic over a `FamilyTree`. The core idea: any blood relation is
fully described by two numbers — how many generations up from the viewer to the
nearest common ancestor (`up`), and how many generations down from that ancestor
to the target (`down`). Relations through marriage are handled by a thin in-law
layer on top of the blood-relation core.

Computation is kept strictly separate from wording: `RelationshipResolver`
returns a language-neutral `Kinship` descriptor (a kind, the target's gender, and
a "grand"/"Ur"/"great-" stacking degree), and `render()` turns that into a string
for one locale, pulling the actual words from the i18n catalogs. That split is
what lets the same tree logic name a relation in German or English — and it means
adding a language never touches this file.

Deliberately covers the cases that occur in a normal family tree; anything more
exotic than a distant cousin resolves to `None` (no label) rather than inventing
dubious terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from web.family_tree import FamilyTree
from web.i18n import Locale, Translator


class KinshipKind(StrEnum):
    """The relation categories the tree logic can name. Each maps to a base word
    (gendered) in the i18n catalogs under `kin.<kind>`; the four grand-* kinds
    additionally stack a `degree` of Ur-/great- prefixes for deeper lines."""

    SELF = "self"
    CHILD = "child"
    GRANDCHILD = "grandchild"
    PARENT = "parent"
    GRANDPARENT = "grandparent"
    SIBLING = "sibling"
    NIBLING = "nibling"                # niece / nephew
    GRAND_NIBLING = "grand_nibling"
    PIBLING = "pibling"                # uncle / aunt
    GRAND_PIBLING = "grand_pibling"
    COUSIN = "cousin"
    COUSIN_DISTANT = "cousin_distant"
    SPOUSE = "spouse"
    SIBLING_IN_LAW = "sibling_in_law"
    CHILD_IN_LAW = "child_in_law"
    STEP_PARENT = "step_parent"
    PARENT_IN_LAW = "parent_in_law"
    MARRIED_IN = "married_in"          # spouse of a distant blood relative


@dataclass(frozen=True)
class Kinship:
    """A language-neutral description of one person's relation to the viewer."""

    kind: KinshipKind
    gender: str = "unknown"  # target's gender, selects the gendered word
    degree: int = 0          # Ur-/great- prefixes for grand-* kinds (0 = base)


# Kinds whose word is a single gendered lookup, `kin.<name>.{male,female,neutral}`.
_SIMPLE_BASE: dict[KinshipKind, str] = {
    KinshipKind.CHILD: "kin.child",
    KinshipKind.PARENT: "kin.parent",
    KinshipKind.SIBLING: "kin.sibling",
    KinshipKind.NIBLING: "kin.nibling",
    KinshipKind.PIBLING: "kin.pibling",
    KinshipKind.COUSIN: "kin.cousin",
    KinshipKind.SPOUSE: "kin.spouse",
    KinshipKind.SIBLING_IN_LAW: "kin.sibling_in_law",
    KinshipKind.CHILD_IN_LAW: "kin.child_in_law",
    KinshipKind.STEP_PARENT: "kin.step_parent",
    KinshipKind.PARENT_IN_LAW: "kin.parent_in_law",
}

# Kinds whose word is a gendered base plus a stack of Ur-/great- prefixes.
_GRAND_BASE: dict[KinshipKind, str] = {
    KinshipKind.GRANDCHILD: "kin.grandchild",
    KinshipKind.GRANDPARENT: "kin.grandparent",
    KinshipKind.GRAND_NIBLING: "kin.grand_nibling",
    KinshipKind.GRAND_PIBLING: "kin.grand_pibling",
}


def _gendered(t: Translator, base_key: str, gender: str) -> str:
    variant = gender if gender in ("male", "female") else "neutral"
    return t(f"{base_key}.{variant}")


def _prefixed(locale: Locale, word: str, degree: int) -> str:
    """Stack ancestral-line prefixes onto a base word for deep generations.

    degree 0 -> word; English stacks 'great-' (great-grandmother), German stacks
    'Ur' onto the lower-cased base (Urgroßmutter, Urur…)."""
    if degree <= 0:
        return word
    if locale == Locale.EN:
        return "great-" * degree + word
    return "Ur" + "ur" * (degree - 1) + word.lower()


def render(kinship: Kinship, t: Translator) -> str:
    """Localised label for a `Kinship`, in the translator's locale."""
    kind = kinship.kind
    if kind == KinshipKind.SELF:
        return t("kin.self")
    if kind == KinshipKind.MARRIED_IN:
        return t("kin.married_in")
    if kind == KinshipKind.COUSIN_DISTANT:
        return f"{_gendered(t, 'kin.cousin', kinship.gender)} ({t('kin.distant')})"
    if kind in _GRAND_BASE:
        base = _gendered(t, _GRAND_BASE[kind], kinship.gender)
        return _prefixed(t.locale, base, kinship.degree)
    return _gendered(t, _SIMPLE_BASE[kind], kinship.gender)


class RelationshipResolver:
    """Computes the `Kinship` of `target` relative to `viewer` from the tree."""

    def __init__(self, tree: FamilyTree) -> None:
        self._tree = tree

    # --- public API ---------------------------------------------------------

    def resolve(self, viewer_id: str, target_id: str) -> Kinship | None:
        """Language-neutral relation, or None if none can be determined."""
        if viewer_id == target_id:
            return Kinship(KinshipKind.SELF)
        if viewer_id not in self._tree.persons or target_id not in self._tree.persons:
            return None
        return self._blood(viewer_id, target_id) or self._in_law(viewer_id, target_id)

    def label(self, viewer_id: str, target_id: str, t: Translator) -> str | None:
        """Localised kinship label, or None when there is no relation to name."""
        kinship = self.resolve(viewer_id, target_id)
        return render(kinship, t) if kinship is not None else None

    # --- blood relations ----------------------------------------------------

    def _ancestors(self, person: str) -> dict[str, int]:
        """Map every ancestor (including `person` at distance 0) to its minimum
        generation distance, via BFS over parent edges."""
        dist = {person: 0}
        frontier = [person]
        while frontier:
            nxt: list[str] = []
            for p in frontier:
                for parent in self._tree.parents_of.get(p, set()):
                    if parent not in dist:
                        dist[parent] = dist[p] + 1
                        nxt.append(parent)
            frontier = nxt
        return dist

    def _coordinates(self, viewer: str, target: str) -> tuple[int, int] | None:
        """(up, down): generations from viewer up to the nearest common ancestor,
        and from that ancestor down to target. None if no common ancestor."""
        va = self._ancestors(viewer)
        ta = self._ancestors(target)
        common = set(va) & set(ta)
        if not common:
            return None
        # Nearest common ancestor minimises total path length.
        lca = min(common, key=lambda a: va[a] + ta[a])
        return va[lca], ta[lca]

    def _blood(self, viewer: str, target: str) -> Kinship | None:
        coords = self._coordinates(viewer, target)
        if coords is None:
            return None
        up, down = coords
        gender = self._tree.persons[target].gender

        # Direct line: target is a descendant of viewer.
        if up == 0:
            if down == 1:
                return Kinship(KinshipKind.CHILD, gender)
            return Kinship(KinshipKind.GRANDCHILD, gender, degree=down - 2)
        # Direct line: target is an ancestor of viewer.
        if down == 0:
            if up == 1:
                return Kinship(KinshipKind.PARENT, gender)
            return Kinship(KinshipKind.GRANDPARENT, gender, degree=up - 2)

        # Collateral lines.
        if up == 1 and down == 1:
            return Kinship(KinshipKind.SIBLING, gender)
        # Viewer's sibling's descendants: Neffe/Nichte, then Groß-/Ur-.
        if up == 1:
            if down == 2:
                return Kinship(KinshipKind.NIBLING, gender)
            return Kinship(KinshipKind.GRAND_NIBLING, gender, degree=down - 3)
        # Viewer's ancestor's siblings: Onkel/Tante, then Groß-/Ur-.
        if down == 1:
            if up == 2:
                return Kinship(KinshipKind.PIBLING, gender)
            return Kinship(KinshipKind.GRAND_PIBLING, gender, degree=up - 3)

        # Cousins (up>=2, down>=2). (2,2) is a first cousin; deeper coordinates
        # are still cousins in everyday usage, just more distant.
        if up == 2 and down == 2:
            return Kinship(KinshipKind.COUSIN, gender)
        return Kinship(KinshipKind.COUSIN_DISTANT, gender)

    # --- in-law relations ---------------------------------------------------

    def _in_law(self, viewer: str, target: str) -> Kinship | None:
        """Relations created by a marriage edge: express `target` as the spouse of
        a blood relative, or as a blood relative of viewer's spouse."""
        tree = self._tree
        gender = tree.persons[target].gender

        # target is married to someone with a known blood relation to viewer.
        for spouse in tree.spouses_of.get(target, set()):
            if spouse == viewer:
                return Kinship(KinshipKind.SPOUSE, gender)
            coords = self._coordinates(viewer, spouse)
            if coords is None:
                continue
            up, down = coords
            if up == 1 and down == 1:  # spouse of viewer's sibling
                return Kinship(KinshipKind.SIBLING_IN_LAW, gender)
            if up == 0 and down == 1:  # spouse of viewer's child
                return Kinship(KinshipKind.CHILD_IN_LAW, gender)
            if up == 1 and down == 0:  # spouse of viewer's parent (step-parent)
                return Kinship(KinshipKind.STEP_PARENT, gender)
            # Spouse of any other blood relative (e.g. a grand-uncle's wife):
            # no single term, but they did marry into the family.
            return Kinship(KinshipKind.MARRIED_IN, gender)

        # target is a blood relative of viewer's spouse.
        for spouse in tree.spouses_of.get(viewer, set()):
            coords = self._coordinates(spouse, target)
            if coords is None:
                continue
            up, down = coords
            if down == 0 and up == 1:   # parent of viewer's spouse
                return Kinship(KinshipKind.PARENT_IN_LAW, gender)
            if up == 1 and down == 1:   # sibling of viewer's spouse
                return Kinship(KinshipKind.SIBLING_IN_LAW, gender)

        return None
