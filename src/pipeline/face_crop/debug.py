"""Debug visualization for face_crop: draw all detected face boxes on a frame."""

from pathlib import Path

import cv2
import numpy as np

from pipeline.face_crop.detector import FaceDetection

_BOX_COLOR = (0, 255, 0)  # green (BGR)


def save_face_overlay(
    img: np.ndarray,
    faces: list[FaceDetection],
    prompt: str,
    out_path: Path,
) -> None:
    """Write a copy of img with every detected face box drawn and labelled with
    its index and score, plus the prompt in the top-left corner."""
    overlay = img.copy()
    img_h, img_w = img.shape[:2]
    scale = max(1.0, img_w / 1000)
    thickness = max(2, int(scale * 1.5))
    font = cv2.FONT_HERSHEY_SIMPLEX

    _draw_label(overlay, f"prompt: {prompt!r}", (12, int(40 * scale)), font, scale, thickness)

    for i, face in enumerate(faces):
        x0, y0, x1, y1 = (int(v) for v in face.box_xyxy)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), _BOX_COLOR, thickness)
        label = f"#{i} {face.score:.2f}"
        ty = y0 - 8 if y0 - 8 > int(30 * scale) else y1 + int(30 * scale)
        _draw_label(overlay, label, (x0, ty), font, scale, thickness)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _draw_label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    scale: float,
    thickness: int,
) -> None:
    """Black text on a filled white box so it stays legible over any background."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    pad = max(3, int(scale * 3))
    cv2.rectangle(
        img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (255, 255, 255), cv2.FILLED
    )
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
