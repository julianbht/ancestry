from pathlib import Path

from pydantic import BaseModel, Field


class FaceBuildConfig(BaseModel):
    existing_tasks: Path | None = None
    # Images already in this ground-truth file are skipped when building a new
    # batch, so re-running build only proposes not-yet-annotated photos.
    existing_ground_truth: Path | None = Path("data/curated/face_annotation/ground_truth.json")
    tasks_output: Path = Path("data/label_studio/face_annotation/tasks.json")


class FaceNormalizeConfig(BaseModel):
    export_path: Path | None = None
    ground_truth_output: Path = Path("data/curated/face_annotation/ground_truth.json")


class FaceAnnotationConfig(BaseModel):
    gramps_source: str = "csv"
    # Generated from the private person list, so it belongs under the data root
    # (a regenerable Label Studio working file), never in public config/.
    template_output: Path = Path("data/label_studio/face_annotation/label_studio.xml")
    build: FaceBuildConfig = Field(default_factory=FaceBuildConfig)
    normalize: FaceNormalizeConfig = Field(default_factory=FaceNormalizeConfig)
