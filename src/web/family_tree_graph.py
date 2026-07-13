"""Turn the FamilyTree kinship edges into a node/edge graph for the Stammbaum
(family-tree) homepage view.

Pure transformation — no I/O. The presenter supplies which people are
clickable (have a photo profile) and which have a portrait image; this module
only assigns generation levels and emits the vis-network-shaped dict the
template hands to the client library.

The whole family-tree feature lives behind web.config.FAMILY_TREE_ENABLED and
is confined to this module + tree.html + the vendored vis-network bundle, so it
can be removed without touching the roster view.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable

from web.family_tree import FamilyTree


def _generation_levels(tree: FamilyTree) -> dict[str, int]:
    """Assign each person a generation row so that every generation sits at the
    same vertical height: a child is exactly one row below its parents, and
    spouses share a row.

    These are *relative* constraints (parent→child = +1, spouse = 0), propagated
    by BFS through the kinship graph rather than by longest-path-from-a-root.
    Longest-path would push otherwise-same-generation people apart whenever one
    branch has more recorded ancestors; relative propagation keeps the rows
    consistent. The family graph is a single connected, conflict-free component,
    so one pass fixes every level; the result is shifted so the oldest row is 0.
    """
    # neighbour -> required offset (neighbour level = this level + offset)
    offsets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for child, parents in tree.parents_of.items():
        for parent in parents:
            offsets[parent].append((child, +1))
            offsets[child].append((parent, -1))
    for a, spouses in tree.spouses_of.items():
        for b in spouses:
            offsets[a].append((b, 0))
            offsets[b].append((a, 0))

    level: dict[str, int] = {}
    for start in tree.persons:
        if start in level:
            continue
        level[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour, delta in offsets[node]:
                if neighbour not in level:
                    level[neighbour] = level[node] + delta
                    queue.append(neighbour)

    if level:
        floor = min(level.values())
        for pid in level:
            level[pid] -= floor
    return level


def _families(tree: FamilyTree) -> dict[frozenset[str], set[str]]:
    """Group children by their exact set of parents. Each distinct parent-set
    that has children is one 'family' — the unit a marriage/union node sits on,
    so a couple's children hang from a single point instead of from both parents
    (which is what crossed the lines in the flat version)."""
    families: dict[frozenset[str], set[str]] = {}
    for child, parents in tree.parents_of.items():
        key = frozenset(p for p in parents if p in tree.persons)
        if key:
            families.setdefault(key, set()).add(child)
    return families


def build_graph(
    tree: FamilyTree,
    clickable_ids: set[str],
    has_portrait: Callable[[str], bool],
    viewer_id: str,
    locale: str,
) -> dict:
    """Return {"nodes": [...], "edges": [...]} ready to JSON-encode for vis-network.

    `clickable_ids` are people with a photo page; only those get a link and a
    portrait. Everyone else in the tree is shown as a plain, dimmed node so the
    structure stays connected.

    Layout uses union (marriage) nodes: an invisible junction is placed half a
    generation below each couple, the parents drop into it, and the children
    drop out of it. Person rows are therefore on even levels and union junctions
    on the odd level between them, so vis-network routes every sibling group
    through one shared point.
    """
    level = _generation_levels(tree)
    families = _families(tree)
    family_keys = set(families)

    nodes: list[dict] = []
    for pid, person in tree.persons.items():
        clickable = pid in clickable_ids
        nodes.append(
            {
                "id": pid,
                "kind": "person",
                "label": person.full_name or pid,
                "level": level.get(pid, 0) * 2,
                "clickable": clickable,
                "image": f"/media/portrait/{pid}" if clickable and has_portrait(pid) else None,
                "gender": person.gender,
                "url": f"/person/{pid}?you={viewer_id}&lang={locale}" if clickable else None,
            }
        )

    edges: list[dict] = []
    for idx, (parents, children) in enumerate(families.items()):
        union_id = f"U{idx}"
        union_level = max(level.get(p, 0) for p in parents) * 2 + 1
        nodes.append(
            {
                "id": union_id,
                "kind": "union",
                "label": "",
                "level": union_level,
                "clickable": False,
                "image": None,
                "gender": None,
                "url": None,
            }
        )
        for parent in parents:
            edges.append({"from": parent, "to": union_id, "kind": "union"})
        for child in children:
            edges.append({"from": union_id, "to": child, "kind": "union"})

    # Spouse links: drawn dashed only for childless couples. Couples who share a
    # family are already joined visually through their union node, so a dashed
    # line there would just add clutter. Emit each pair once (smaller id first).
    seen: set[tuple[str, str]] = set()
    for person, spouses in tree.spouses_of.items():
        for spouse in spouses:
            pair = tuple(sorted((person, spouse)))
            if pair in seen or frozenset(pair) in family_keys:
                continue
            seen.add(pair)
            edges.append({"from": pair[0], "to": pair[1], "kind": "spouse"})

    return {"nodes": nodes, "edges": edges}
