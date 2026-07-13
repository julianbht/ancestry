"""Typed models for face annotation Label Studio helpers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---- Task building ----

class FaceTaskData(BaseModel):
    image: str
    key: str
    image_rel: str
    source: str


class FacePredictionResult(BaseModel):
    """One pre-drawn rectangle attached to a task (a face_crop detection).

    Mirrors a Label Studio prediction region. x/y/width/height in value are
    percentages of the displayed image, matching the Rectangle control.
    """

    id: str
    type: str = "rectangle"
    from_name: str = "face"
    to_name: str = "image"
    original_width: int
    original_height: int
    image_rotation: float = 0.0
    value: dict


class FacePrediction(BaseModel):
    """A set of pre-drawn regions shown on a task before annotation."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str = "face_crop"
    result: list[FacePredictionResult] = Field(default_factory=list)


class FaceTask(BaseModel):
    data: FaceTaskData
    predictions: list[FacePrediction] = Field(default_factory=list)


class FaceExistingTaskData(BaseModel):
    key: str | None = None
    model_config = ConfigDict(extra="ignore")


class FaceExistingTask(BaseModel):
    data: FaceExistingTaskData = Field(default_factory=FaceExistingTaskData)
    model_config = ConfigDict(extra="ignore")


# ---- Label Studio export parsing ----

class LsRectangleValue(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    model_config = ConfigDict(extra="ignore")


class LsChoicesValue(BaseModel):
    choices: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class LsTaxonomyValue(BaseModel):
    # Each entry is a path from root to leaf, e.g. [["Boldt", "I0013: Karl Boldt"]]
    taxonomy: list[list[str]] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class LsTextAreaValue(BaseModel):
    text: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class LsNumberValue(BaseModel):
    number: float | None = None
    model_config = ConfigDict(extra="ignore")


class LsFaceResult(BaseModel):
    id: str
    type: str
    from_name: str = ""
    from_id: str | None = None
    value: dict = Field(default_factory=dict)
    model_config = ConfigDict(extra="ignore")


class LsFaceAnnotation(BaseModel):
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    was_cancelled: bool = False  # Label Studio sets this when the user clicks Skip
    result: list[LsFaceResult] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class LsFaceExportData(BaseModel):
    key: str | None = None
    source: str | None = None
    image_rel: str | None = None
    model_config = ConfigDict(extra="ignore")


class LsFaceExportTask(BaseModel):
    id: int | None = None
    data: LsFaceExportData = Field(default_factory=LsFaceExportData)
    annotations: list[LsFaceAnnotation] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


# ---- Ground truth ----

class FaceAnnotationGT(BaseModel):
    """A single annotated face within a photo."""
    # Pixel coordinates, top-left origin: (x, y, width, height)
    bbox_xywh: tuple[float, float, float, float]
    person_id: str | None   # Gramps ID (e.g. "I0004") or None for unknown
    face_note: str | None   # freehand note for this face (e.g. unrecognized person name)
    age: int | None
    is_portrait: bool


class FaceGroundTruthItem(BaseModel):
    image_rel: str
    image_size: tuple[int, int]     # (width, height) in pixels
    faces: list[FaceAnnotationGT]
    note: str | None = None
    task_id: int | None = None
    annotation_id: int | None = None


class FaceGroundTruthFile(BaseModel):
    version: int = 1
    items: dict[str, FaceGroundTruthItem]
