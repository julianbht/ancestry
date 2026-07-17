"""Config schema for the gramps step."""

from typing import Annotated, Literal

from pydantic import Field

from pipeline.shared.config import StrictConfig


class GrampsConfig(StrictConfig):
    # --- What to merge ---
    # include_curated_portraits: attach the hand-cropped portraits from
    #   data/gramps/portraits/. They always become the person's first media
    #   reference, which is what Gramps shows as the profile thumbnail.
    include_curated_portraits: bool
    # include_faces: which labelled faces from ground_truth.json to attach.
    #   "portrait" — only faces flagged is_portrait.
    #   "all" — every labelled face, so each photo's "people in this image" is
    #   fully populated.
    include_faces: Literal["portrait", "all"]

    # --- Merge mode ---
    # replace_media: drop each touched person's existing media references before
    #   attaching ours, then prune any media object left unreferenced. Default is
    #   additive, which keeps media already in the export.
    replace_media: bool

    # --- Run control ---
    # max_files_to_bake: cap the images baked per run (null = unlimited). Media
    #   whose image isn't baked yet is still written to the XML, so a capped run
    #   produces an export with missing image files — for smoke tests only.
    max_files_to_bake: Annotated[int | None, Field(ge=1)]
    # ignore_state: re-bake images even if already marked done.
    ignore_state: bool
    # dry_run: report the plan; bake nothing and write no XML.
    dry_run: bool

    # JPEG quality for baked media copies (1-100). Ignored for PNG sources.
    jpeg_quality: Annotated[int, Field(ge=1, le=100)]
