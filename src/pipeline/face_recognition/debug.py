"""Debug visualization for face_recognition: draw every face box on its frame,
labelled with the matched person's name and distance, colour-coded by status.

Text is rendered with Pillow rather than cv2.putText: cv2's Hershey fonts can't
draw the accented characters (ä, ö, ü, ...) that show up in German given names
and surnames, while Pillow + a TrueType font handles them natively.
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.face_recognition.sidecar import RecognitionEntry, RecognitionStatus

# DejaVu Sans ships with matplotlib (already a project dependency) and covers
# the Latin-1 diacritics that cv2.putText cannot render.
_FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"

_COLORS = {
    RecognitionStatus.RECOGNIZED: (0, 200, 0),  # green (RGB)
    RecognitionStatus.UNKNOWN: (255, 170, 0),  # amber
    RecognitionStatus.NO_EMBEDDING: (255, 0, 0),  # red
}
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)


def save_recognition_overlay(
    img: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    recognitions: list[RecognitionEntry],
    id_to_name: dict[str, str],
    params_line: str,
    font_scale: float,
    out_path: Path,
) -> None:
    """Write a copy of img with every face box drawn and labelled with the
    matched person's name and distance, colour-coded by status, plus a
    run-parameters line (black on white) in the top-left corner."""
    overlay = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(str(_FONT_PATH), round(24 * font_scale))
    dist_font = ImageFont.truetype(str(_FONT_PATH), round(20 * font_scale))
    box_width = max(2, round(font_scale * 3))
    stroke_width = max(1, round(font_scale))

    _draw_label_with_background(draw, (12, 4), params_line, font)

    for box, rec in zip(boxes, recognitions):
        color = _COLORS[rec.status]
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1, y1), outline=color, width=box_width)

        name_lines = label_name(rec, id_to_name).split(" ", 1)
        _, top, _, bottom = font.getbbox(name_lines[0])
        line_h = bottom - top + 4

        name_y = y0 - line_h * len(name_lines) - 4
        name_y = name_y if name_y > 0 else y1 + 4
        for i, line in enumerate(name_lines):
            draw.text(
                (x0, name_y + i * line_h), line, font=font, fill=color,
                stroke_width=stroke_width, stroke_fill=_BLACK,
            )

        # Bottom-left line: distance score followed by the model's own short
        # gender/age estimate (e.g. "0.44  F 32"), each part shown only if present.
        bottom_parts = []
        if rec.distance is not None:
            bottom_parts.append(f"{rec.distance:.2f}")
        attr = " ".join(p for p in (rec.gender, str(rec.age) if rec.age is not None else None) if p)
        if attr:
            bottom_parts.append(attr)
        if bottom_parts:
            _, dist_top, _, dist_bottom = dist_font.getbbox("0.00")
            dist_y = y1 - (dist_bottom - dist_top) - 4
            draw.text(
                (x0 + 4, dist_y), "  ".join(bottom_parts), font=dist_font, fill=color,
                stroke_width=stroke_width, stroke_fill=_BLACK,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR))


def label_name(rec: RecognitionEntry, id_to_name: dict[str, str]) -> str:
    if rec.status == RecognitionStatus.RECOGNIZED:
        assert rec.person_id is not None
        return id_to_name.get(rec.person_id, rec.person_id)
    if rec.status == RecognitionStatus.UNKNOWN:
        return "unknown"
    return "no embedding"


def _draw_label_with_background(
    draw: ImageDraw.ImageDraw,
    org: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Black text on a filled white box so it stays legible over any background."""
    pad = 4
    bbox = draw.textbbox(org, text, font=font)
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=_WHITE)
    draw.text(org, text, font=font, fill=_BLACK)
