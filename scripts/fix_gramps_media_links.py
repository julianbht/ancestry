"""One-off: repair the Gramps export's broken media links, in place.

The hand-cropped portraits in data/gramps/portraits/ were renamed to carry the
person id ("hergoss-julian.png" -> "I0004-hergoss-julian.png"), which left the
media objects in data/gramps/database/data.gramps pointing at filenames that no
longer exist. This fixes the export so it stays usable — same tree, same
handles, same ids — in two passes:

  1. Repoint what was renamed. Only the `src` attribute changes, and only when
     the rename is unambiguous: exactly one portrait matches the old name once
     its "I####-" prefix is stripped, every person referencing that media is the
     person whose id the new file carries, and the suffix is unchanged so the
     recorded `mime` and `checksum` stay correct (a rename doesn't alter bytes).

  2. Delete what is gone. A portrait whose file cannot be found and cannot be
     repaired is a dead pointer: it is removed along with every <objref> to it.
     Nothing else references a media object, so this leaves no dangling handles.

     Media attached to a <citation> or <source> is never touched, even when its
     file is missing. Those are scanned evidence — a baptism certificate, a book
     page — and the recorded filename is itself a record of which document
     backs the claim, worth keeping until the scan is restored.

Both passes are text surgery on exact spans rather than an XML re-serialisation,
so every byte outside the repaired srcs and the deleted elements is preserved:
element order, attribute order, whitespace, the DOCTYPE. The result is verified
against the original before anything is written — same tree, same text, same
attributes, differing only where intended.

Usage:
    uv run python scripts/fix_gramps_media_links.py            # dry run + diff
    uv run python scripts/fix_gramps_media_links.py --apply    # write (backs up first)
"""

import argparse
import difflib
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pipeline.gramps.portraits import PORTRAITS_DIR, person_id_from_filename
from pipeline.shared.paths import GRAMPS_DIR, rel

GRAMPS_NS = "http://gramps-project.org/xml/1.7.2/"
_NS = f"{{{GRAMPS_NS}}}"

EXPORT_FILE = GRAMPS_DIR / "database" / "data.gramps"
BACKUP_DIR = GRAMPS_DIR / "database" / "backup"


@dataclass(frozen=True)
class Repair:
    """A media object's src, and what it should become."""

    old_src: str
    new_src: str
    person_id: str


@dataclass(frozen=True)
class Deletion:
    """A media object whose file is gone, to be removed with its references."""

    handle: str
    src: str
    reason: str
    referenced_by: list[str] = field(default_factory=list)


def _portraits_by_stripped_stem() -> dict[str, tuple[str, Path]]:
    """Map a portrait's pre-rename stem -> (person id, current file).

    'I0004-hergoss-julian.png' is indexed under 'hergoss-julian', which is
    exactly the name the stale export still refers to.
    """
    index: dict[str, tuple[str, Path]] = {}
    for path in sorted(PORTRAITS_DIR.iterdir()):
        person_id = person_id_from_filename(path)
        if person_id is None:
            continue
        stripped = path.stem[len(person_id) :].lstrip("-_.")
        index.setdefault(stripped.lower(), (person_id, path))
    return index


@dataclass(frozen=True)
class Ref:
    """One objref pointing at a media object, and what owns it."""

    kind: str  # "person" | "citation" | "source" | ...
    owner_id: str


# Media hanging off these is scanned evidence, not a portrait: never touched.
_EVIDENCE_KINDS = frozenset({"citation", "source"})


def _referencing(root: ET.Element) -> dict[str, list[Ref]]:
    """media handle -> the refs pointing at it.

    Media can be referenced from people, citations and sources alike, so this
    walks the whole tree rather than just <people>.
    """
    refs: dict[str, list[Ref]] = {}
    for parent in root.iter():
        kind = parent.tag.split("}", 1)[-1]
        for objref in parent.findall(f"{_NS}objref"):
            ref = Ref(kind=kind, owner_id=parent.get("id") or f"<{kind}>")
            refs.setdefault(objref.get("hlink", ""), []).append(ref)
    return refs


def plan(root: ET.Element) -> tuple[list[Repair], list[Deletion], list[str]]:
    """Decide which srcs to rewrite, which media to delete, and what to leave.

    Returns (repairs, deletions, reports of dead media left alone).
    """
    portraits = _portraits_by_stripped_stem()
    referenced_by = _referencing(root)
    repairs: list[Repair] = []
    deletions: list[Deletion] = []
    kept: list[str] = []

    for obj in root.iter(f"{_NS}object"):
        file_el = obj.find(f"{_NS}file")
        if file_el is None:
            continue
        old_src = file_el.get("src", "")
        old_path = Path(old_src)
        if old_path.exists():
            continue  # link works — not our business

        handle = obj.get("handle", "")
        refs = referenced_by.get(handle, [])
        owner_ids = sorted({ref.owner_id for ref in refs})

        evidence = sorted({ref.owner_id for ref in refs if ref.kind in _EVIDENCE_KINDS})
        if evidence:
            kept.append(
                f"{old_path.name}: attached to {', '.join(evidence)} — scanned "
                "evidence, left for you to restore or remove in Gramps"
            )
            continue

        def _delete(reason: str) -> None:
            deletions.append(
                Deletion(handle=handle, src=old_src, reason=reason, referenced_by=owner_ids)
            )

        match = portraits.get(old_path.stem.lower())
        if match is None:
            _delete("no file with this name exists anywhere under data/")
            continue
        person_id, new_path = match

        if not refs:
            # Repointing an orphan would silently create a second media object
            # for a file another object already claims. It's dead either way.
            _delete("orphan — nothing references it")
        elif owner_ids != [person_id]:
            _delete(
                f"referenced by {', '.join(owner_ids)} but the matching file "
                f"on disk is for {person_id}"
            )
        elif new_path.suffix.lower() != old_path.suffix.lower():
            _delete(
                f"candidate {new_path.name} has a different suffix, so the "
                "recorded mime would be wrong"
            )
        else:
            repairs.append(
                Repair(old_src=old_src, new_src=new_path.resolve().as_posix(), person_id=person_id)
            )

    return repairs, deletions, kept


def _sub_once(text: str, needle: str, replacement: str) -> str:
    count = text.count(needle)
    if count != 1:
        raise ValueError(f"expected exactly one {needle} in the export, found {count}")
    return text.replace(needle, replacement)


def _drop_element(text: str, tag: str, handle_attr: str, handle: str, *, expect: int) -> str:
    """Remove every `<tag handle_attr="handle" .../>`, self-closing or not.

    The export is pretty-printed one element per line, so the whole line goes,
    indentation included. DOTALL covers the block form (an <objref> carrying a
    <region>); handles are unique, so the first closing tag is the right one.
    """
    pattern = re.compile(
        rf'[ \t]*<{tag} {handle_attr}="{re.escape(handle)}"(?:[^>]*?/>|[^>]*?>.*?</{tag}>)\r?\n',
        re.DOTALL,
    )
    text, count = pattern.subn("", text)
    if count != expect:
        raise ValueError(f"expected to remove {expect} <{tag}> for {handle}, removed {count}")
    return text


def apply_plan(text: str, repairs: list[Repair], deletions: list[Deletion]) -> str:
    """Rewrite srcs and cut dead media, leaving every other byte untouched."""
    for repair in repairs:
        text = _sub_once(text, f'src="{repair.old_src}"', f'src="{repair.new_src}"')
    for deletion in deletions:
        text = _drop_element(text, "object", "handle", deletion.handle, expect=1)
        if deletion.referenced_by:
            text = _drop_element(
                text, "objref", "hlink", deletion.handle, expect=len(deletion.referenced_by)
            )
    return text


def _kept_elements(root: ET.Element, dead_handles: set[str]) -> list[ET.Element]:
    """Every element except the media objects we deleted, their children, and
    the objrefs pointing at them — i.e. what must survive verbatim."""
    kept: list[ET.Element] = []

    def walk(element: ET.Element) -> None:
        tag = element.tag.split("}", 1)[-1]
        if tag == "object" and element.get("handle") in dead_handles:
            return
        if tag == "objref" and element.get("hlink") in dead_handles:
            return
        kept.append(element)
        for child in element:
            walk(child)

    walk(root)
    return kept


def verify(original: str, repaired: str, repairs: list[Repair], deletions: list[Deletion]) -> None:
    """Prove nothing but the intended srcs and deletions changed.

    Walks the surviving elements of both trees in lockstep, comparing tag, text
    and attributes. Anything else that moved raises before a byte is written.
    """
    expected_src = {r.old_src: r.new_src for r in repairs}
    dead = {d.handle for d in deletions}

    before = _kept_elements(ET.fromstring(original), dead)
    after = _kept_elements(ET.fromstring(repaired), set())
    if len(before) != len(after):
        raise ValueError(f"surviving element count changed: {len(before)} -> {len(after)}")

    for old, new in zip(before, after):
        if old.tag != new.tag or (old.text or "").strip() != (new.text or "").strip():
            raise ValueError(f"element content changed at <{old.tag}>")
        if list(old.attrib) != list(new.attrib):
            raise ValueError(f"attribute set changed at <{old.tag}>")
        for key, old_value in old.attrib.items():
            new_value = new.attrib[key]
            if old_value == new_value:
                continue
            if key == "src" and expected_src.get(old_value) == new_value:
                continue
            raise ValueError(
                f"unintended change at <{old.tag}> {key}: {old_value!r} -> {new_value!r}"
            )


def _print_diff(original: str, repaired: str) -> None:
    diff = difflib.unified_diff(
        original.splitlines(), repaired.splitlines(), "before", "after", lineterm="", n=0
    )
    for line in diff:
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the repaired file (backs the original up first). Without it, "
        "nothing is written — the plan and a diff are printed.",
    )
    parser.add_argument(
        "--input", type=Path, default=EXPORT_FILE, help=f"default: {rel(EXPORT_FILE)}"
    )
    args = parser.parse_args()

    original = args.input.read_text(encoding="utf-8")
    repairs, deletions, kept = plan(ET.fromstring(original))

    print(f"Export: {rel(args.input)}")

    print(f"\n{len(repairs)} link(s) to repoint at the renamed file:")
    for repair in repairs:
        print(f"  {repair.person_id}  {Path(repair.old_src).name}")
        print(f"        -> {Path(repair.new_src).name}")

    print(f"\n{len(deletions)} dead media object(s) to delete:")
    for deletion in deletions:
        who = ", ".join(deletion.referenced_by) if deletion.referenced_by else "nobody"
        print(f"  {Path(deletion.src).name}")
        print(f"        referenced by {who}; {deletion.reason}")

    if kept:
        print(f"\n{len(kept)} dead link(s) left untouched:")
        for report in kept:
            print(f"  {report}")

    if not repairs and not deletions:
        print("\nNothing to do.")
        return

    repaired = apply_plan(original, repairs, deletions)
    verify(original, repaired, repairs, deletions)
    print("\nVerified: the two files differ only in the changes listed above.")

    if not args.apply:
        print("\nDiff (dry run — nothing written; pass --apply to write):")
        _print_diff(original, repaired)
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup = BACKUP_DIR / f"{args.input.stem}.{stamp}.gramps"
    shutil.copy2(args.input, backup)
    args.input.write_text(repaired, encoding="utf-8")

    print(f"\nBacked up to {rel(backup)}")
    print(f"Wrote {rel(args.input)} — {len(repairs)} repaired, {len(deletions)} deleted.")


if __name__ == "__main__":
    main()
