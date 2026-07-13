"""Config schema for the face_recognition step."""

from typing import Annotated

from pydantic import Field

from pipeline.shared.config import StrictConfig


class DebugConfig(StrictConfig):
    save_overlay: bool
    font_scale: Annotated[float, Field(gt=0.0)]
    abbreviate_surname: bool


class DeepFaceConfig(StrictConfig):
    # DeepFace embedding model. Good choices: "ArcFace", "Facenet512", "VGG-Face".
    model_name: str
    # Face detection/alignment backend used inside each crop before embedding.
    # "skip": treat the whole crop as-is (fastest, no landmark alignment).
    # "opencv" / "retinaface" / "mtcnn": re-detect within the crop for alignment.
    detector_backend: str


class InsightFaceConfig(StrictConfig):
    # Model pack to download and use. "buffalo_l" is the standard high-accuracy
    # choice; "antelopev2" is heavier but potentially more accurate.
    model_pack: str
    # Input resolution fed to the detector inside FaceAnalysis.prepare().
    # (640, 640) is the InsightFace default and works well for face crops.
    det_size: tuple[int, int]
    # Padding added around each face crop before detection, as a fraction of
    # max(height, width). Face crops are tight so InsightFace's internal detector
    # often misses them without margin. 0.4 = 40% padding on each side.
    # Set to 0.0 to disable. Good candidate for hpopt.
    pad_ratio: Annotated[float, Field(ge=0.0, le=2.0)]
    # Detector confidence floor passed to FaceAnalysis.prepare(). InsightFace's
    # default is 0.5; lowering it recovers weak/blurry/partial faces the detector
    # would otherwise miss (the main source of "no face detected" embedding
    # failures), at the cost of more spurious detections on non-face crops. Good
    # candidate for hpopt.
    det_thresh: Annotated[float, Field(gt=0.0, le=1.0)]


class FaceRecognitionConfig(StrictConfig):
    # --- File selection ---
    max_files_to_recognize: Annotated[int | None, Field(ge=1)]
    ignore_state: bool
    skip_state_write: bool
    skip_output_write: bool

    # --- Method ---
    # Which embedding backend to use: "deepface" or "insightface".
    # Method-specific params are loaded from config/face_recognition/<method>.yaml.
    method: str

    # --- Matching ---
    # Distance metric for gallery nearest-neighbour matching.
    # "cosine" | "euclidean" | "euclidean_l2"
    distance_metric: str
    # Maximum distance to be considered a match. Below threshold → "recognized";
    # at or above → "unknown". Typical ranges (cosine): DeepFace ArcFace 0.30–0.50,
    # InsightFace buffalo_l 0.40–0.60 — needs tuning per method.
    recognition_threshold: Annotated[float, Field(ge=0.0, le=2.0)]
    # How many gallery people to record per face in the sidecar's `candidates`
    # list, ranked by nearest distance. Cheap to store; powers Label Studio
    # review (pre-filled alternatives) and the #1→#2 ambiguity margin.
    top_k_candidates: Annotated[int, Field(ge=1)]

    # --- Gallery ---
    # Minimum overlap coefficient (intersection / area of the smaller box) between
    # a ground-truth bounding box and a face_crop detection (both in source-image
    # coordinates) to accept the crop as a gallery reference.
    gallery_overlap_threshold: Annotated[float, Field(ge=0.0, le=1.0)]

    debug: DebugConfig
