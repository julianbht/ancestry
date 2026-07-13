"""Encrypt private data and sync it to Cloudflare R2 (and back).

R2 is the canonical, durable store for the project's private data (see
docs/going-public.md); local disk is a working *cache* of it. This CLI moves
that data: ``push`` encrypts + uploads new/changed local files, ``pull``
reconstructs local files from R2 (e.g. to hydrate a fresh machine).

The set of private data is the ``SYNC_ROOTS`` manifest below, split into three
categories (see docs/going-public.md): ``raw`` (original photos), ``curated``
(hand-made tree + labels), and ``derived`` (regenerable pipeline output — state
JSONs, step sidecars, and face embeddings). The encryption + transport
primitives live in ``pipeline.shared.r2``; this file is just the orchestration:
walk those roots, track what's uploaded, and drive the diff. State
(``data/state/r2_sync.json``) records each file's plaintext SHA-256 so re-running
``push`` only uploads what changed. Crypto never depends on that state — a
``pull`` works from R2 + passphrase alone.

Usage
-----
    uv run python scripts/sync_private.py push        # encrypt + upload new/changed
    uv run python scripts/sync_private.py pull         # download + decrypt missing
    uv run python scripts/sync_private.py status       # local vs remote counts
    uv run python scripts/sync_private.py push --limit 5 --dry-run   # smoke test
    uv run python scripts/sync_private.py pull --only curated   # tree + labels, skip raw photos
    uv run python scripts/sync_private.py push --only derived    # snapshot pipeline output

``--only {raw,curated,derived}`` restricts any command to one category.
``curated`` is the small hand-made data (Gramps tree, ground truth, CSVs);
``raw`` is the bulk of original photos; ``derived`` is the pipeline's regenerable
output (state + sidecars + embeddings). On a fresh pod that already has the
images, ``pull --only curated`` restores just the curated inputs without
re-downloading raw. On the cluster after a run, ``push --only derived`` snapshots
the results so a local machine can ``pull --only derived`` and skip recomputation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError

# load_dotenv() runs on import of the shared config module.
from pipeline.shared import (
    config as _config,
)  # noqa: F401  (import for .env side effect)
from pipeline.shared import r2
from pipeline.shared.hashing import sha256_file
from pipeline.shared.log import logger, setup
from pipeline.shared.paths import CURATED_DIR, DATA_DIR, GRAMPS_DIR, STEPS_DIR, rel
from pipeline.shared.state import STATE_DIR

# --- What gets synced -------------------------------------------------------
# The private data — see the category table in docs/going-public.md. Everything
# lives under the data root, so each root's logical R2 key is just its DATA_DIR-
# relative path (raw photos → "raw/...", curated → "curated/..."/"gramps/...",
# derived → "steps/..."/"state/..."). Each root is a tree (walked, optionally
# filtered by `patterns`) or a single file. gramps/graphs is derived-and-cheap,
# so it's excluded; the crop *images* under steps/ are excluded too (regenerable
# from the sidecars via the reconstruct_* scripts) — only the sidecars +
# embeddings are worth storing.

# The three categories from docs/going-public.md. They differ in size and how
# precious they are: `raw` is the bulk (original photos, ~GBs, irreplaceable),
# `curated` is the small hand-made stuff (tree, labels, CSVs, irreplaceable),
# `derived` is regenerable pipeline output (state, sidecars, embeddings) kept
# only to skip recomputation. `--only <category>` syncs one at a time.
RAW = "raw"
CURATED = "curated"
DERIVED = "derived"
CATEGORIES = (RAW, CURATED, DERIVED)


@dataclass(frozen=True)
class SyncRoot:
    """A private tree or single file mirrored to R2. `local` is made relative to
    DATA_DIR to form its logical R2 key (and AAD). For a directory, `patterns`
    (rglob globs) restricts which files are walked — empty means every file;
    `category` is one of CATEGORIES, so `--only` can sync a subset."""

    local: Path
    is_dir: bool
    category: str
    patterns: tuple[str, ...] = ()


SYNC_ROOTS: list[SyncRoot] = [
    SyncRoot(DATA_DIR / "raw", is_dir=True, category=RAW),  # the originals → "raw/..."
    # Gramps genealogy — hand-made, non-regenerable → "gramps/...".
    SyncRoot(GRAMPS_DIR / "database", is_dir=True, category=CURATED),
    SyncRoot(GRAMPS_DIR / "portraits", is_dir=True, category=CURATED),
    SyncRoot(GRAMPS_DIR / "documents", is_dir=True, category=CURATED),
    # Curated inputs — hand-made, non-regenerable → "curated/...".
    SyncRoot(CURATED_DIR / "face_annotation" / "ground_truth.json", False, CURATED),
    SyncRoot(CURATED_DIR / "frame_crop" / "ground_truth.json", False, CURATED),
    SyncRoot(CURATED_DIR / "rotate" / "rotations.csv", False, CURATED),
    SyncRoot(CURATED_DIR / "photo_backs.csv", False, CURATED),
    # Derived pipeline output — regenerable, snapshot to skip recomputation.
    # Per-step state files → "state/..." (r2_sync.json itself is never listed
    # here, so the tool never syncs its own bookkeeping).
    SyncRoot(STATE_DIR / "frame_crop.json", False, DERIVED),
    SyncRoot(STATE_DIR / "face_crop.json", False, DERIVED),
    SyncRoot(STATE_DIR / "face_recognition.json", False, DERIVED),
    # Step sidecars + embeddings → "steps/..." (crop images excluded via patterns).
    SyncRoot(STEPS_DIR / "frame_crop", True, DERIVED, patterns=("*.json",)),
    SyncRoot(STEPS_DIR / "face_crop", True, DERIVED, patterns=("faces.json",)),
    SyncRoot(
        STEPS_DIR / "face_recognition",
        True,
        DERIVED,
        patterns=("recognition.json", "embeddings.npy"),
    ),
]


def _selected_roots(only: str | None) -> list[SyncRoot]:
    """The sync roots for this run: all of them, or just one category (`--only`)."""
    if only is None:
        return SYNC_ROOTS
    return [root for root in SYNC_ROOTS if root.category == only]

STATE_FILE = STATE_DIR / "r2_sync.json"


# ---------------------------------------------------------------------------
# State (upload tracking — purely an optimization; crypto never depends on it)
# ---------------------------------------------------------------------------
def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"version": 1, "objects": {}}


def _save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _root_prefix(root: SyncRoot) -> str:
    """Logical R2 prefix for a root: its DATA_DIR-relative path (trailing slash
    for directories, so listing can't spill into a sibling name)."""
    logical = root.local.relative_to(DATA_DIR).as_posix()
    return logical + "/" if root.is_dir else logical


def _iter_root_files(root: SyncRoot) -> list[Path]:
    """Local files under a root (the file itself, or the tree). For a directory
    root, `patterns` (rglob globs) restricts which files match — empty walks the
    whole tree; multiple patterns union (e.g. sidecars + embeddings, no images).
    Missing roots yield nothing — a machine need not have every asset."""
    if root.is_dir:
        if not root.local.is_dir():
            logger.warning(f"skipping missing directory {rel(root.local)}")
            return []
        patterns = root.patterns or ("*",)
        matched = {p for pat in patterns for p in root.local.rglob(pat) if p.is_file()}
        return sorted(matched)
    if not root.local.is_file():
        logger.warning(f"skipping missing file {rel(root.local)}")
        return []
    return [root.local]


def _local_pairs(limit: int, roots: list[SyncRoot]) -> list[tuple[SyncRoot, Path]]:
    """(root, local file) across the given sync roots, optionally capped for smoke tests."""
    pairs = [(root, local) for root in roots for local in _iter_root_files(root)]
    return pairs[:limit] if limit else pairs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_push(args: argparse.Namespace) -> None:
    s = r2.load_settings()
    client = r2.make_client(s)
    state = _load_state()

    pairs = _local_pairs(args.limit, _selected_roots(args.only))

    uploaded = skipped = failed = 0
    for root, local in pairs:
        key = r2.r2_key(local)
        rec_key = local.relative_to(DATA_DIR).as_posix()
        digest = sha256_file(local)

        prior = state["objects"].get(rec_key)
        if not args.ignore_state and prior and prior.get("sha256") == digest:
            skipped += 1
            continue

        if args.dry_run:
            logger.info(f"[dry-run] would upload {rel(local)} -> {key}")
            uploaded += 1
            continue

        try:
            blob = r2.encrypt(local.read_bytes(), s.passphrase, r2.aad_for(local))
            client.put_object(Bucket=s.bucket, Key=key, Body=blob)
        except (ClientError, OSError) as e:
            logger.error(f"FAILED {rel(local)}: {e}")
            failed += 1
            continue

        state["objects"][rec_key] = {
            "sha256": digest,
            "r2_key": key,
            "size": local.stat().st_size,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_state(state)  # per-file, so a crash mid-run is recoverable
        uploaded += 1
        logger.info(f"uploaded {rel(local)} ({local.stat().st_size / 1e6:.1f} MB)")

    verb = "would upload" if args.dry_run else "uploaded"
    logger.info(
        f"Done — {len(pairs)} local files: {uploaded} {verb}, "
        f"{skipped} skipped (unchanged), {failed} failed"
    )


def cmd_pull(args: argparse.Namespace) -> None:
    s = r2.load_settings()
    client = r2.make_client(s)
    state = _load_state()

    pairs = [
        (root, key)
        for root in _selected_roots(args.only)
        for key in r2.list_keys(client, s.bucket, _root_prefix(root))
    ]
    if args.limit:
        pairs = pairs[: args.limit]

    downloaded = skipped = failed = 0
    for root, key in pairs:
        local = r2.local_from_key(key)
        rec_key = local.relative_to(DATA_DIR).as_posix()

        if not args.ignore_state and not args.force and local.exists():
            # Trust an existing local file whose hash matches recorded state.
            prior = state["objects"].get(rec_key)
            if prior and prior.get("sha256") == sha256_file(local):
                skipped += 1
                continue

        if args.dry_run:
            logger.info(f"[dry-run] would download {key} -> {rel(local)}")
            downloaded += 1
            continue

        try:
            blob = client.get_object(Bucket=s.bucket, Key=key)["Body"].read()
            plaintext = r2.decrypt(blob, s.passphrase, r2.aad_for(local))
        except (ClientError, ValueError) as e:
            logger.error(f"FAILED {key}: {e}")
            failed += 1
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(plaintext)
        state["objects"][rec_key] = {
            "sha256": sha256_file(local),
            "r2_key": key,
            "size": len(plaintext),
            "uploaded_at": state["objects"].get(rec_key, {}).get("uploaded_at"),
        }
        _save_state(state)
        downloaded += 1
        logger.info(f"downloaded {rel(local)} ({len(plaintext) / 1e6:.1f} MB)")

    verb = "would download" if args.dry_run else "downloaded"
    logger.info(
        f"Done — {len(pairs)} remote objects: {downloaded} {verb}, "
        f"{skipped} skipped (already local), {failed} failed"
    )


def cmd_status(args: argparse.Namespace) -> None:
    s = r2.load_settings()
    client = r2.make_client(s)

    roots = _selected_roots(args.only)
    local_keys = {r2.r2_key(local) for _, local in _local_pairs(0, roots)}
    remote_keys: set[str] = set()
    for root in roots:
        remote_keys.update(r2.list_keys(client, s.bucket, _root_prefix(root)))

    logger.info(f"Bucket: {s.bucket}   roots: {len(roots)}")
    logger.info(f"Local files:  {len(local_keys)}")
    logger.info(f"Remote blobs: {len(remote_keys)}")
    logger.info(f"Local-only (need push): {len(local_keys - remote_keys)}")
    logger.info(f"Remote-only (need pull): {len(remote_keys - local_keys)}")


# ---------------------------------------------------------------------------
def main() -> None:
    setup("r2_sync")
    # No local-existence guard here: `pull` legitimately runs on a fresh machine
    # where nothing exists locally yet. `push` warns per missing root instead.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_only(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--only",
            choices=CATEGORIES,
            default=None,
            help="sync only one category (e.g. 'curated' = tree + labels, skip raw photos)",
        )

    for name, fn, verb in (
        ("push", cmd_push, "upload"),
        ("pull", cmd_pull, "download"),
    ):
        p = sub.add_parser(name, help=f"{verb} new/changed files")
        add_only(p)
        p.add_argument(
            "--limit", type=int, default=0, help="process at most N files (smoke test)"
        )
        p.add_argument(
            "--ignore-state",
            action="store_true",
            help=f"re-{verb} everything, ignoring recorded state",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="show what would happen, transfer nothing",
        )
        if name == "pull":
            p.add_argument(
                "--force",
                action="store_true",
                help="overwrite local files even if they exist",
            )
        p.set_defaults(func=fn)

    sp = sub.add_parser("status", help="show local vs remote counts")
    add_only(sp)
    sp.set_defaults(func=cmd_status)

    args = parser.parse_args()
    try:
        args.func(args)
    except r2.MissingSecretError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
