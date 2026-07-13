from pathlib import Path

import cv2
import numpy as np

from pipeline.frame_crop.config import FrameCropConfig
from pipeline.frame_crop.methods.base import Detection

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _save(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _wrap_line(text: str, scale: float, thickness: int, max_w: int) -> list[str]:
    """Greedily break `text` on spaces so each piece fits within max_w pixels."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        trial = word if not current else f"{current} {word}"
        (trial_w, _), _ = cv2.getTextSize(trial, _FONT, scale, thickness)
        if trial_w <= max_w or not current:
            current = trial  # keep going (or accept an over-wide lone word)
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _annotate(
    img: np.ndarray,
    params: dict[str, object],
    scale: float = 1.0,
    bold: bool = False,
) -> np.ndarray:
    """Append a parameter bar below the image, wrapping lines too wide to fit."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    _, w = out.shape[:2]
    text_scale = max(0.8, w / 800) * scale
    thickness = 2 if bold else 1
    pad = 8
    max_w = w - 2 * pad
    lines: list[str] = []
    for key, value in params.items():
        lines.extend(_wrap_line(f"{key}: {value}", text_scale, thickness, max_w))
    if not lines:
        return out
    (_, text_h), baseline = cv2.getTextSize("Ag", _FONT, text_scale, thickness)
    line_gap = max(8, int(text_h * 0.4))
    bar_h = pad * 2 + len(lines) * text_h + (len(lines) - 1) * line_gap + baseline
    text_y = pad + text_h
    bar = np.full((bar_h, w, 3), 255, dtype=np.uint8)
    for line in lines:
        cv2.putText(
            bar, line, (pad, text_y), _FONT, text_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )
        text_y += text_h + line_gap
    return np.vstack((out, bar))


def _quad_params(detection: Detection, method: str) -> dict[str, object]:
    """Short, human-meaningful fields to show under the quad overlay.

    The full failure reason stays in the logs; here we show only the method, the
    text prompt and the detection confidence when available (SAM).
    """
    params: dict[str, object] = {"method": method}
    if detection.prompt is not None:
        params["prompt"] = detection.prompt
    if detection.score is not None:
        params["score"] = f"{detection.score:.2f}"
    return params


def save_debug_quad(
    img: np.ndarray,
    detection: Detection,
    config: FrameCropConfig,
    debug_dir: Path,
) -> None:
    annotated = img.copy()
    h, w = img.shape[:2]
    if detection.quad is not None:
        cv2.polylines(
            annotated, [detection.quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3
        )
    else:
        cv2.line(annotated, (0, 0), (w, h), (0, 0, 255), 3)
        cv2.line(annotated, (w, 0), (0, h), (0, 0, 255), 3)
    if config.debug.annotate_quad:
        annotated = _annotate(
            annotated,
            _quad_params(detection, config.method),
            scale=config.debug.annotation.scale,
            bold=config.debug.annotation.bold,
        )
    _save(debug_dir / "quad.jpg", annotated)
