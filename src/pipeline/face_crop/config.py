"""Config schema for the face_crop step.

Unlike frame_crop there is only one detection method (SAM), so its parameters
live in a nested ``sam:`` block of step.yaml rather than in a separate method
file — there is nothing to switch between.
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from pipeline.shared.config import StrictConfig
from pipeline.shared.paths import MODELS_DIR

# Reuse the same SAM 3 image checkpoint that frame_crop already downloads via
# scripts/setup_sam3.py. A derived filesystem path, so it keeps a default rather
# than being spelled out in YAML (see CLAUDE.md "No hidden defaults" exception).
_SAM3_CHECKPOINT = MODELS_DIR / "sam3" / "sam3.pt"


class SamConfig(StrictConfig):
    prompt: str
    score_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    device: str
    checkpoint_path: Path = _SAM3_CHECKPOINT


class DebugConfig(StrictConfig):
    # save_overlay: dump an annotated copy of each frame (all face boxes + scores)
    # to data/debug/face_crop/ for visual inspection.
    save_overlay: bool


class FaceCropConfig(StrictConfig):
    # --- File selection ---
    max_files_to_crop: Annotated[int | None, Field(ge=1)]
    only_file: str | None
    only_ground_truth: bool
    ignore_state: bool
    # skip_state_write: process every frame regardless of prior state and
    # don't record the outcome in data/state/ — next run will redo the same
    # frames.
    skip_state_write: bool
    # skip_output_write: detect and log, but don't write face crops or the
    # faces.json sidecar to disk. Combine with skip_state_write for a smoke
    # test that leaves no trace on disk.
    skip_output_write: bool

    # Discard detections whose box covers less than this fraction of the frame
    # area — guards against tiny spurious boxes. Set 0.0 to keep everything.
    min_area_fraction: Annotated[float, Field(ge=0.0, lt=1.0)]
    # Pad each detected face box outward by this fraction of its width/height on
    # every side before cropping (0.0 = the bare box). Crops are clamped to the
    # frame bounds.
    margin_frac: Annotated[float, Field(ge=0.0)]
    jpeg_quality: Annotated[int, Field(ge=1, le=100)]

    sam: SamConfig
    debug: DebugConfig
