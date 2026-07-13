"""Master entry point — runs all pipeline steps in order.

Each step is idempotent and skips already-processed files, so this is safe
to re-run at any time. New files will be picked up automatically.

Usage:
    uv run python -m pipeline
"""

from pipeline.download.step import run as download
from pipeline.face_crop.step import run as face_crop
from pipeline.face_recognition.step import run as face_recognition
from pipeline.frame_crop.step import run as frame_crop
from pipeline.rotate.step import run as rotate


def main() -> None:
    download()
    rotate()
    frame_crop()
    face_crop()
    face_recognition()
    # age_estimation()


if __name__ == "__main__":
    main()
