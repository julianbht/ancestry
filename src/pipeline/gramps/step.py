"""
Gramps step: merge our portraits and face labels into the Gramps family tree.

Reads two label sources and augments the Gramps XML export
(data/gramps/database/data.gramps) so that each person carries the faces we know
about:

  - hand-cropped portraits from data/gramps/portraits/, attached whole (they are
    already cropped to the face);
  - labelled faces from data/curated/face_annotation/ground_truth.json, attached
    as a *media reference with a region* — the face bbox in Gramps'
    integer-percent coordinates — on the photo they were found in.

Hand-cropped portraits always come first, so they win the profile thumbnail:
dropping a correctly-named file into data/gramps/portraits/ is enough to fix
someone's portrait, no annotation needed. See pipeline.gramps.plan for the
ordering rules and pipeline.gramps.media for why images are baked upright first.

This never touches the live Gramps SQLite database. It edits an *exported*
.gramps XML and writes a new augmented .gramps. Because the augmented file is a
strict superset of the export (same handles for the existing people), the
reliable way to load it is to IMPORT IT INTO A FRESH, EMPTY family tree.
Importing into the existing tree is NOT recommended: Gramps remaps colliding
handles on import, so it duplicates people rather than adding the new objrefs.

Idempotency: the expensive part is baking an upright copy of each referenced
image, and that is what the state file tracks. The XML itself is rebuilt from
the export on every run — it is a deterministic function of the labels, so
rewriting it is cheap and always correct.
"""

from loguru import logger

from pipeline.gramps import GRAMPS_DB_DIR
from pipeline.gramps.annotations import GROUND_TRUTH_FILE, load_face_annotations
from pipeline.gramps.config import GrampsConfig
from pipeline.gramps.export import GrampsExport
from pipeline.gramps.media import MEDIA_DIR, bake_upright, media_path
from pipeline.gramps.plan import MediaItem, Plan, build_plan
from pipeline.gramps.portraits import PORTRAITS_DIR, load_curated_portraits
from pipeline.shared import state as state_lib
from pipeline.shared.config import load as load_config
from pipeline.shared.log import setup
from pipeline.shared.paths import rel

EXPORT_FILE = GRAMPS_DB_DIR / "data.gramps"
AUGMENTED_FILE = GRAMPS_DB_DIR / "data.augmented.gramps"
STATE_FILE = state_lib.STATE_DIR / "gramps.json"

_IMPORT_HINT = (
    f"Import {rel(AUGMENTED_FILE)} into a NEW, empty Gramps tree "
    "(Family Trees -> Manage Family Trees -> New, Load, then Import...). "
    "Do NOT import into your existing tree: Gramps remaps colliding handles and "
    "would duplicate every person instead of adding the face regions."
)


def _log_plan(plan: Plan, config: GrampsConfig) -> None:
    stats = plan.stats
    logger.info(
        f"Plan: {plan.ref_count} media ref(s) across {len(plan.refs_by_person)} people "
        f"— {stats.curated_portraits} hand-cropped portrait(s), "
        f"{stats.ground_truth_faces} annotated face(s); "
        f"{len(plan.media)} media object(s)"
    )
    if config.include_faces == "portrait":
        logger.info(
            f"{stats.skipped_not_portrait} annotated face(s) skipped "
            "(not is_portrait, include_faces=portrait)"
        )
    logger.info(f"{stats.unlabelled_faces} annotated face(s) have no person_id")
    if stats.unknown_person_ids:
        logger.warning(
            f"{len(stats.unknown_person_ids)} labelled person id(s) are not in the "
            f"family tree and were dropped: {', '.join(sorted(stats.unknown_person_ids))}"
        )


def _bake_media(plan: Plan, config: GrampsConfig) -> tuple[int, int, int, int]:
    """Bake an upright copy of each planned image.

    Returns (baked, skipped, missing, failed). Missing sources are reported but
    not fatal: the media object is still written, so the export tells you which
    file to restore.
    """
    state = state_lib.load(STATE_FILE)
    baked = skipped = missing = failed = 0

    for item in plan.media.values():
        if config.max_files_to_bake is not None and baked >= config.max_files_to_bake:
            break
        if not config.ignore_state and state_lib.is_done(state, item.key):
            skipped += 1
            continue
        if not item.source.exists():
            logger.warning(f"Source image missing: {rel(item.source)}")
            missing += 1
            continue

        dest = media_path(item.key)
        try:
            bake_upright(item.source, dest, config.jpeg_quality)
            state_lib.mark_done(state, item.key, [str(dest)])
            state_lib.save(state, STATE_FILE)
            baked += 1
        except Exception as e:  # noqa: BLE001 — one bad image must not stop the run
            state_lib.mark_failed(state, item.key, str(e))
            state_lib.save(state, STATE_FILE)
            logger.error(f"Failed to bake {rel(item.source)}: {e}")
            failed += 1

    return baked, skipped, missing, failed


def _merge(plan: Plan, export: GrampsExport, config: GrampsConfig) -> tuple[int, int]:
    """Apply the plan to the export. Returns (media added, media pruned)."""
    if config.replace_media:
        export.clear_person_media(set(plan.refs_by_person))

    added: list[MediaItem] = export.add_media(list(plan.media.values()))
    for person_id, refs in plan.refs_by_person.items():
        export.add_person_media_refs(person_id, refs)

    pruned = export.prune_unreferenced_media() if config.replace_media else 0
    return len(added), pruned


def run() -> None:
    setup("gramps")

    try:
        config = load_config("gramps", GrampsConfig)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return

    logger.info(
        f"Config: include_curated_portraits={config.include_curated_portraits}, "
        f"include_faces={config.include_faces}, "
        f"mode={'replace-media' if config.replace_media else 'additive'}, "
        f"max_files_to_bake={config.max_files_to_bake or 'unlimited'}, "
        f"dry_run={config.dry_run}"
    )

    if not EXPORT_FILE.exists():
        logger.error(
            f"No Gramps export at {rel(EXPORT_FILE)} — export your tree from Gramps "
            "(Family Trees -> Export..., Gramps XML) and save it there first."
        )
        return

    try:
        export = GrampsExport.load(EXPORT_FILE)
    except ValueError as e:
        logger.error(f"Cannot read {rel(EXPORT_FILE)}: {e}")
        return

    logger.info(f"Source export: {rel(EXPORT_FILE)} ({len(export.person_ids)} people)")
    logger.info(f"Portraits    : {rel(PORTRAITS_DIR)}")
    logger.info(f"Ground truth : {rel(GROUND_TRUTH_FILE)}")

    plan = build_plan(
        annotations=load_face_annotations(),
        curated_portraits=load_curated_portraits(),
        known_person_ids=export.person_ids,
        include_faces=config.include_faces,
        include_curated_portraits=config.include_curated_portraits,
    )
    _log_plan(plan, config)

    if config.dry_run:
        logger.info("Done — dry run, nothing written")
        return

    baked, skipped, missing, failed = _bake_media(plan, config)
    added, pruned = _merge(plan, export, config)
    export.write(AUGMENTED_FILE)

    logger.info(f"Wrote {rel(AUGMENTED_FILE)}")
    logger.info(_IMPORT_HINT)
    logger.info(
        f"Done — {plan.ref_count} media refs attached to {len(plan.refs_by_person)} people "
        f"({plan.stats.curated_portraits} hand-cropped portraits, "
        f"{plan.stats.ground_truth_faces} annotated faces), "
        f"{added} media objects added"
        + (f", {pruned} unreferenced media pruned" if config.replace_media else "")
        + f"; images: {baked} baked upright into {rel(MEDIA_DIR)}, "
        f"{skipped} skipped (already baked), "
        f"{missing} missing (source file not on disk), {failed} failed (errors)"
    )


if __name__ == "__main__":
    run()
