"""Config loading and overrides for Label Studio helpers."""

from __future__ import annotations

from pathlib import Path

from pipeline.annotation.config import LabelStudioConfig
from pipeline.annotation.label_studio_models import (
    BuildSettings,
    NormalizeSettings,
    TrackingSettings,
)
from pipeline.annotation.label_studio_paths import resolve_path, resolve_paths
from pipeline.shared.config import load as load_config
from pipeline.shared.config import load_file as load_config_file
from pipeline.shared.paths import CONFIG_DIR


def load_annotation_config(config_path: Path | None) -> tuple[LabelStudioConfig, Path]:
    if config_path is None:
        import os
        env = os.environ.get("ENV", "prod")
        return load_config("annotation", LabelStudioConfig), (
            CONFIG_DIR / "annotation" / f"step.{env}.yaml"
        )
    resolved = resolve_path(config_path)
    return load_config_file(resolved, LabelStudioConfig), resolved


def apply_build_overrides(
    config: LabelStudioConfig,
    cases_override: list[Path] | None,
    existing_override: Path | None,
    output_override: Path | None,
) -> LabelStudioConfig:
    if cases_override is None and existing_override is None and output_override is None:
        return config

    build_updates: dict[str, Path | list[Path] | None] = {}
    if cases_override is not None:
        build_updates["cases_files"] = cases_override
    if existing_override is not None:
        build_updates["existing_tasks"] = existing_override
    if output_override is not None:
        build_updates["tasks_output"] = output_override

    build = config.build.model_copy(update=build_updates)
    return config.model_copy(update={"build": build})


def apply_normalize_overrides(
    config: LabelStudioConfig, export_override: Path | None, output_override: Path | None
) -> LabelStudioConfig:
    if export_override is None and output_override is None:
        return config

    normalize_updates: dict[str, Path | None] = {}
    if export_override is not None:
        normalize_updates["export_path"] = export_override
    if output_override is not None:
        normalize_updates["ground_truth_output"] = output_override

    normalize = config.normalize.model_copy(update=normalize_updates)
    return config.model_copy(update={"normalize": normalize})


def build_settings_from_config(config: LabelStudioConfig) -> BuildSettings:
    cases_files = resolve_paths(config.build.cases_files)
    tasks_output = resolve_path(config.build.tasks_output)
    existing_tasks = (
        resolve_path(config.build.existing_tasks)
        if config.build.existing_tasks is not None
        else None
    )
    return BuildSettings(
        cases_files=cases_files,
        tasks_output=tasks_output,
        existing_tasks=existing_tasks,
    )


def normalize_settings_from_config(config: LabelStudioConfig) -> NormalizeSettings:
    export_path = config.normalize.export_path
    if export_path is None:
        raise ValueError(
            "Export path is required (set normalize.export_path or pass --export)."
        )
    return NormalizeSettings(
        export_path=resolve_path(export_path),
        ground_truth_output=resolve_path(config.normalize.ground_truth_output),
    )


def tracking_settings_from_config(config: LabelStudioConfig) -> TrackingSettings:
    return TrackingSettings(
        enabled=config.tracking.enabled,
        runs_dir=resolve_path(config.tracking.runs_dir),
        frame_crop_config_path=(
            resolve_path(config.tracking.frame_crop_config_path)
            if config.tracking.frame_crop_config_path is not None
            else None
        ),
    )
