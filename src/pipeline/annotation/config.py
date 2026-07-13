"""Config schema for Label Studio annotation helpers."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class LabelStudioBuildConfig(BaseModel):
    cases_files: list[Path] = Field(
        default_factory=lambda: [Path("config/frame_crop/label_cases.txt")]
    )
    existing_tasks: Path | None = None
    tasks_output: Path = Path("data/label_studio/frame_crop/tasks.json")


class LabelStudioNormalizeConfig(BaseModel):
    export_path: Path | None = None
    ground_truth_output: Path = Path("data/curated/frame_crop/ground_truth.json")


class LabelStudioTrackingConfig(BaseModel):
    enabled: bool = True
    runs_dir: Path = Path("data/label_studio/frame_crop/runs")
    frame_crop_config_path: Path | None = Path("config/frame_crop/step.prod.yaml")


class LabelStudioConfig(BaseModel):
    label_name: str = "photo"
    build: LabelStudioBuildConfig = LabelStudioBuildConfig()
    normalize: LabelStudioNormalizeConfig = LabelStudioNormalizeConfig()
    tracking: LabelStudioTrackingConfig = LabelStudioTrackingConfig()

    @field_validator("label_name")
    @classmethod
    def label_name_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("label_name must be non-empty")
        return value
