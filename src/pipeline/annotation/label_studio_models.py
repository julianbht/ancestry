"""Typed models for Label Studio helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class BuildSettings:
    cases_files: list[Path]
    tasks_output: Path
    existing_tasks: Path | None


@dataclass(frozen=True)
class NormalizeSettings:
    export_path: Path
    ground_truth_output: Path


@dataclass(frozen=True)
class TrackingSettings:
    enabled: bool
    runs_dir: Path
    frame_crop_config_path: Path | None


@dataclass(frozen=True)
class BuildOutcome:
    tasks_written: int
    skipped_existing: int


@dataclass(frozen=True)
class NormalizeOutcome:
    labels_written: int
    skipped: int


class LabelStudioTaskData(BaseModel):
    image: str
    key: str
    image_rel: str
    source: str


class LabelStudioTask(BaseModel):
    data: LabelStudioTaskData


class LabelStudioExistingTaskData(BaseModel):
    key: str | None = None

    model_config = ConfigDict(extra="ignore")


class LabelStudioExistingTask(BaseModel):
    data: LabelStudioExistingTaskData = Field(default_factory=LabelStudioExistingTaskData)

    model_config = ConfigDict(extra="ignore")


class LabelStudioExportData(BaseModel):
    key: str | None = None
    source: str | None = None
    image_rel: str | None = None

    model_config = ConfigDict(extra="ignore")


class LabelStudioResultValue(BaseModel):
    polygonlabels: list[str] = Field(default_factory=list)
    points: list[tuple[float, float]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class LabelStudioResult(BaseModel):
    type: str
    value: LabelStudioResultValue | None = None

    model_config = ConfigDict(extra="ignore")


class LabelStudioAnnotation(BaseModel):
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    result: list[LabelStudioResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class LabelStudioExportTask(BaseModel):
    id: int | None = None
    data: LabelStudioExportData = Field(default_factory=LabelStudioExportData)
    annotations: list[LabelStudioAnnotation] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class GroundTruthItem(BaseModel):
    image_rel: str
    image_size: tuple[int, int]
    quad: list[tuple[float, float]]
    source: Literal["label_studio"] = "label_studio"
    task_id: int | None = None
    annotation_id: int | None = None


class GroundTruthFile(BaseModel):
    version: int = 1
    items: dict[str, GroundTruthItem]


class RunInputs(BaseModel):
    cases_files: list[str] | None = None
    existing_tasks: str | None = None
    export_path: str | None = None


class RunOutputs(BaseModel):
    tasks_output: str | None = None
    ground_truth_output: str | None = None


class RunResults(BaseModel):
    tasks_written: int | None = None
    skipped_existing: int | None = None
    labels_written: int | None = None
    skipped_tasks: int | None = None


class RunConfigFiles(BaseModel):
    annotation_config: str
    frame_crop_config: str | None = None


class RunManifest(BaseModel):
    version: int = 1
    run_id: str
    command: Literal["build", "normalize"]
    created_at: str
    argv: list[str]
    label_name: str
    inputs: RunInputs
    outputs: RunOutputs
    results: RunResults
    config_files: RunConfigFiles
