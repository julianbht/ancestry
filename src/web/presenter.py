"""Presentation layer: turn domain objects into the view models the templates
render. Keeps grouping/ordering/label-wiring out of both the routes and the
domain code.
"""

from __future__ import annotations

from dataclasses import dataclass

from web.family_tree import FamilyTree
from web.family_tree_graph import build_graph
from web.i18n import Translator
from web.kinship import RelationshipResolver
from web.models import Appearance, Person
from web.portraits import PortraitService
from web.repository import Repository


@dataclass(frozen=True)
class PersonCard:
    person: Person
    photo_count: int
    has_portrait: bool
    relationship: str | None  # localised kinship term relative to the viewer


@dataclass(frozen=True)
class SurnameGroup:
    surname: str
    cards: list[PersonCard]

    @property
    def total_photos(self) -> int:
        return sum(c.photo_count for c in self.cards)


@dataclass(frozen=True)
class ViewerOption:
    id: str
    name: str


class Presenter:
    def __init__(
        self,
        repo: Repository,
        resolver: RelationshipResolver,
        portraits: PortraitService,
        tree: FamilyTree,
    ) -> None:
        self._repo = repo
        self._resolver = resolver
        self._portraits = portraits
        self._tree = tree

    # --- homepage -----------------------------------------------------------

    def home_groups(self, viewer_id: str, t: Translator) -> list[SurnameGroup]:
        cards: list[PersonCard] = []
        for profile in self._repo.profiles():
            person = profile.person
            source = self._repo.portrait_source(person.id)
            has_portrait = self._portraits.resolve(person, source) is not None
            cards.append(
                PersonCard(
                    person=person,
                    photo_count=profile.photo_count,
                    has_portrait=has_portrait,
                    relationship=self._resolver.label(viewer_id, person.id, t),
                )
            )

        groups: dict[str, list[PersonCard]] = {}
        for card in cards:
            groups.setdefault(card.person.surname or "—", []).append(card)

        result = [
            SurnameGroup(
                surname=surname,
                # most-photographed people first within a family
                cards=sorted(members, key=lambda c: c.photo_count, reverse=True),
            )
            for surname, members in groups.items()
        ]
        # families with the most photos overall first
        result.sort(key=lambda g: g.total_photos, reverse=True)
        return result

    def family_tree_graph(self, viewer_id: str, locale: str) -> dict:
        """Node/edge graph for the Stammbaum view. People with a photo page are
        clickable and carry a portrait; the rest are plain connecting nodes."""
        profiles = {p.person.id: p for p in self._repo.profiles()}

        def has_portrait(pid: str) -> bool:
            profile = profiles.get(pid)
            if profile is None:
                return False
            source = self._repo.portrait_source(pid)
            return self._portraits.resolve(profile.person, source) is not None

        return build_graph(self._tree, set(profiles), has_portrait, viewer_id, locale)

    def viewer_options(self) -> list[ViewerOption]:
        people = sorted(
            self._repo.tree_persons(),
            key=lambda p: (p.surname, p.given),
        )
        return [ViewerOption(id=p.id, name=p.full_name) for p in people]

    # --- person page --------------------------------------------------------

    def person_appearances(self, person_id: str) -> list[Appearance]:
        profile = self._repo.profile(person_id)
        return profile.appearances if profile else []

    def relationship(self, viewer_id: str, person_id: str, t: Translator) -> str | None:
        return self._resolver.label(viewer_id, person_id, t)
