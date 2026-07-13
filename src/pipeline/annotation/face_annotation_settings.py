"""Config loading and overrides for face annotation helpers."""

from __future__ import annotations

from pathlib import Path

from pipeline.annotation.face_annotation_config import FaceAnnotationConfig
from pipeline.annotation.label_studio_paths import resolve_path
from pipeline.shared.config import load as load_config
from pipeline.shared.config import load_file as load_config_file
from pipeline.shared.paths import CONFIG_DIR


def load_face_annotation_config(
    config_path: Path | None,
) -> tuple[FaceAnnotationConfig, Path]:
    if config_path is None:
        import os

        env = os.environ.get("ENV", "prod")
        return load_config("face_annotation", FaceAnnotationConfig), (
            CONFIG_DIR / "face_annotation" / f"step.{env}.yaml"
        )
    resolved = resolve_path(config_path)
    return load_config_file(resolved, FaceAnnotationConfig), resolved


def apply_build_overrides(
    config: FaceAnnotationConfig,
    existing_override: Path | None,
    output_override: Path | None,
    ground_truth_override: Path | None = None,
) -> FaceAnnotationConfig:
    if (
        existing_override is None
        and output_override is None
        and ground_truth_override is None
    ):
        return config
    updates = {
        k: v
        for k, v in {
            "existing_tasks": existing_override,
            "tasks_output": output_override,
            "existing_ground_truth": ground_truth_override,
        }.items()
        if v is not None
    }
    return config.model_copy(update={"build": config.build.model_copy(update=updates)})


def apply_normalize_overrides(
    config: FaceAnnotationConfig,
    export_override: Path | None,
    output_override: Path | None,
) -> FaceAnnotationConfig:
    if export_override is None and output_override is None:
        return config
    updates = {
        k: v
        for k, v in {
            "export_path": export_override,
            "ground_truth_output": output_override,
        }.items()
        if v is not None
    }
    return config.model_copy(
        update={"normalize": config.normalize.model_copy(update=updates)}
    )


def resolved_tasks_output(config: FaceAnnotationConfig) -> Path:
    return resolve_path(config.build.tasks_output)


def resolved_existing_tasks(config: FaceAnnotationConfig) -> Path | None:
    if config.build.existing_tasks is None:
        return None
    return resolve_path(config.build.existing_tasks)


def resolved_existing_ground_truth(config: FaceAnnotationConfig) -> Path | None:
    if config.build.existing_ground_truth is None:
        return None
    return resolve_path(config.build.existing_ground_truth)


def resolved_export_path(config: FaceAnnotationConfig) -> Path:
    if config.normalize.export_path is None:
        raise ValueError(
            "Export path is required (set normalize.export_path or pass --export)."
        )
    return resolve_path(config.normalize.export_path)


def resolved_ground_truth_output(config: FaceAnnotationConfig) -> Path:
    return resolve_path(config.normalize.ground_truth_output)


def resolved_template_output(config: FaceAnnotationConfig) -> Path:
    return resolve_path(config.template_output)
