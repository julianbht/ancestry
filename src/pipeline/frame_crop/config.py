"""Config schema for the frame_crop step."""

from typing import Annotated

from pydantic import Field, field_validator

from pipeline.shared.config import StrictConfig


class BlurConfig(StrictConfig):
    enabled: bool
    save_output: bool
    annotate: bool
    kernel_size: Annotated[int, Field(ge=1)]

    @field_validator("kernel_size")
    @classmethod
    def must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return v


class MorphologyConfig(StrictConfig):
    enabled: bool
    save_output: bool
    annotate: bool
    save_diff: bool
    kernel_size: Annotated[int, Field(ge=1)]
    iterations: Annotated[int, Field(ge=1)]


class AnnotationConfig(StrictConfig):
    scale: Annotated[float, Field(gt=0)]
    bold: bool


class DebugConfig(StrictConfig):
    save_quad: bool
    annotate_quad: bool
    annotation: AnnotationConfig


class FrameCropConfig(StrictConfig):
    max_files_to_crop: Annotated[int | None, Field(ge=1)]
    only_file: str | None
    only_ground_truth: bool
    method: str
    min_area_fraction: Annotated[float, Field(gt=0, lt=1)]
    max_area_fraction: Annotated[float, Field(gt=0, lt=1)]
    margin_px: Annotated[int, Field(ge=0)]
    inner_crop: bool
    jpeg_quality: Annotated[int, Field(ge=1, le=100)]
    ignore_state: bool
    # skip_state_write: process every file regardless of prior state and don't
    # record the outcome in data/state/ — next run will redo the same files.
    skip_state_write: bool
    # skip_output_write: detect and log, but don't write the cropped image or
    # its sidecar JSON to disk. Combine with skip_state_write for a smoke test
    # that leaves no trace on disk.
    skip_output_write: bool
    debug: DebugConfig
