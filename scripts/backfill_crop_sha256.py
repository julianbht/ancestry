"""Backfill content hashes into existing pipeline sidecars.

Several sidecars gained SHA-256 fields after they were first written. This
one-off script fills them by hashing the referenced files on disk — no models or
GPU needed, so it runs locally without re-running any step. The hash is a pure
function of the referenced bytes, so every pass produces the same value for the
same file (e.g. frame_crop's sha256 and face_crop's source_sha256 hash the same
frame, and come out identical).

Fields backfilled:
  - frame_crop  <crop>.json   sha256         (hash of crop_image, the frame)
  - face_crop   faces.json    source_sha256  (hash of source_image, the frame)
  - face_crop   faces.json    faces[].sha256 (hash of each crop_image)
  - face_recog  recognition.json recognitions[].crop_sha256 (hash of each crop_image)

IMPORTANT — this trusts that each file on disk is still the exact one its sidecar
describes. There was no hash before, so that can't be verified; it is the
"assuming the current JSONs all match" premise. Only run this if the producing
steps have NOT been re-run since these sidecars were written.

Note on recognition: this backfills only crop_sha256. The same schema change also
added `candidates` and `embeddings_file`, which are NOT reconstructable from disk
(they need the gallery distances / stored embeddings a re-run would produce). Old
recognition.json files therefore stay partially upgraded until face_recognition
is re-run.

Usage:
    uv run python scripts/backfill_crop_sha256.py [--force] [--dry-run]
"""

import argparse
import json
from pathlib import Path

from pipeline.shared.hashing import hash_crop_image
from pipeline.shared.paths import STEPS_DIR, rel as project_rel

FRAME_CROP_DIR = STEPS_DIR / "frame_crop"
FACE_CROP_DIR = STEPS_DIR / "face_crop"
RECOGNITION_DIR = STEPS_DIR / "face_recognition"


def _backfill_pass(
    sidecar_paths: list[Path],
    entries_key: str,
    hash_field: str,
    force: bool,
    dry_run: bool,
) -> None:
    """Add `hash_field` (hash of each entry's crop_image) to every entry under
    `entries_key` in each sidecar. Shared by both sidecar types, which differ only
    in those two key names."""
    n_files_updated = 0
    n_entries_hashed = 0
    n_entries_skipped = 0  # already had the hash (and not --force)
    n_crops_missing = 0    # crop file absent → hash recorded as null

    for sidecar_path in sidecar_paths:
        data = json.loads(sidecar_path.read_text())
        changed = False

        for entry in data.get(entries_key, []):
            if not force and entry.get(hash_field) is not None:
                n_entries_skipped += 1
                continue

            sha = hash_crop_image(entry["crop_image"])
            if sha is None:
                n_crops_missing += 1
                print(f"  WARN crop missing: {entry['crop_image']} ({project_rel(sidecar_path)})")
            entry[hash_field] = sha
            n_entries_hashed += 1
            changed = True

        if changed:
            n_files_updated += 1
            if not dry_run:
                sidecar_path.write_text(json.dumps(data, indent=2))

    action = "would update" if dry_run else "updated"
    print(
        f"  {action} {n_files_updated} file(s); {n_entries_hashed} entries hashed "
        f"({n_crops_missing} with crop missing → null), "
        f"{n_entries_skipped} skipped (already had hash)"
    )


def _backfill_toplevel_pass(
    sidecar_paths: list[Path],
    hash_field: str,
    source_field: str,
    force: bool,
    dry_run: bool,
) -> None:
    """Add a single top-level `hash_field` (hash of the file named by top-level
    `source_field`) to each sidecar. For sidecars that carry one hash of one
    referenced file (frame_crop's <crop>.json, face_crop's faces.json frame), as
    opposed to the per-entry lists handled by _backfill_pass."""
    n_files_updated = 0
    n_skipped = 0      # already had the hash (and not --force)
    n_missing = 0      # referenced file absent → hash recorded as null

    for sidecar_path in sidecar_paths:
        data = json.loads(sidecar_path.read_text())
        if not force and data.get(hash_field) is not None:
            n_skipped += 1
            continue

        sha = hash_crop_image(data[source_field])
        if sha is None:
            n_missing += 1
            print(f"  WARN file missing: {data[source_field]} ({project_rel(sidecar_path)})")
        data[hash_field] = sha
        n_files_updated += 1
        if not dry_run:
            sidecar_path.write_text(json.dumps(data, indent=2))

    action = "would update" if dry_run else "updated"
    print(
        f"  {action} {n_files_updated} file(s) ({n_missing} with file missing → null), "
        f"{n_skipped} skipped (already had hash)"
    )


def backfill(force: bool, dry_run: bool) -> None:
    frame_crop_sidecars = sorted(FRAME_CROP_DIR.rglob("*.json"))
    face_crop_sidecars = sorted(FACE_CROP_DIR.rglob("faces.json"))
    recognition_sidecars = sorted(RECOGNITION_DIR.rglob("recognition.json"))

    print(f"frame_crop: {len(frame_crop_sidecars)} sidecar(s) under {project_rel(FRAME_CROP_DIR)}")
    _backfill_toplevel_pass(frame_crop_sidecars, "sha256", "crop_image", force, dry_run)

    print(f"face_crop: {len(face_crop_sidecars)} faces.json under {project_rel(FACE_CROP_DIR)}")
    _backfill_toplevel_pass(face_crop_sidecars, "source_sha256", "source_image", force, dry_run)
    _backfill_pass(face_crop_sidecars, "faces", "sha256", force, dry_run)

    print(f"face_recognition: {len(recognition_sidecars)} recognition.json under {project_rel(RECOGNITION_DIR)}")
    _backfill_pass(recognition_sidecars, "recognitions", "crop_sha256", force, dry_run)

    if dry_run:
        print("\nDry run — no files written. Re-run without --dry-run to apply.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-hash and overwrite entries that already have a hash",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files",
    )
    args = parser.parse_args()
    backfill(args.force, args.dry_run)


if __name__ == "__main__":
    main()
