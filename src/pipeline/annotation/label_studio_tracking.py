"""Run manifest tracking for Label Studio helpers."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pipeline.annotation.label_studio_models import (
    RunConfigFiles,
    RunInputs,
    RunManifest,
    RunOutputs,
    RunResults,
    TrackingSettings,
)


def create_run_dir(runs_dir: Path, command: str) -> tuple[Path, str]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = runs_dir / f"{run_id}_{command}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_id


def snapshot_config(source: Path, run_dir: Path, name: str) -> str:
    dest = run_dir / name
    dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest.name


def write_run_manifest(
    tracking: TrackingSettings,
    command: Literal["build", "normalize"],
    label_name: str,
    inputs: RunInputs,
    outputs: RunOutputs,
    results: RunResults,
    annotation_config_path: Path,
    frame_crop_config_path: Path | None,
    argv: list[str] | None = None,
) -> None:
    if not tracking.enabled:
        return

    run_dir, run_id = create_run_dir(tracking.runs_dir, command)
    annotation_copy = snapshot_config(annotation_config_path, run_dir, "annotation.yaml")
    frame_crop_copy = None
    if frame_crop_config_path is not None and frame_crop_config_path.exists():
        frame_crop_copy = snapshot_config(
            frame_crop_config_path, run_dir, "frame_crop.yaml"
        )

    manifest = RunManifest(
        run_id=run_id,
        command=command,
        created_at=datetime.now(timezone.utc).isoformat(),
        argv=(argv or sys.argv[1:]),
        label_name=label_name,
        inputs=inputs,
        outputs=outputs,
        results=results,
        config_files=RunConfigFiles(
            annotation_config=annotation_copy,
            frame_crop_config=frame_crop_copy,
        ),
    )
    (run_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
