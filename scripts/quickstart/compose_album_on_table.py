"""Place the quickstart album photos on the table background.

The script keeps a copy of the source photos under ``quickstart/originals/album``
(ignored by git), composites each photo onto ``quickstart/wood-table-background.png``
with a small paper border and rotation, and writes:

- the curated images to ``quickstart/data/raw/album/``
- frame-crop ground truth to ``quickstart/data/curated/frame_crop/ground_truth.json``
- a reusable placement manifest to ``quickstart/data/curated/frame_crop/table_layouts.json``
- a contact-sheet preview to ``quickstart/data/debug/table-preview.jpg``

Usage:
    uv run python scripts/quickstart/compose_album_on_table.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from pipeline.shared.paths import PROJECT_ROOT

QUICKSTART_ROOT = PROJECT_ROOT / "quickstart"
SOURCE_ALBUM = QUICKSTART_ROOT / "data" / "raw" / "album"
ORIGINALS_ALBUM = QUICKSTART_ROOT / "originals" / "album"
BACKGROUND_PATH = QUICKSTART_ROOT / "wood-table-background.png"
GROUND_TRUTH_PATH = (
    QUICKSTART_ROOT / "data" / "curated" / "frame_crop" / "ground_truth.json"
)
LAYOUTS_PATH = (
    QUICKSTART_ROOT / "data" / "curated" / "frame_crop" / "table_layouts.json"
)
PREVIEW_PATH = QUICKSTART_ROOT / "data" / "debug" / "table-preview.jpg"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_LONG_EDGE_MIN_FRACTION = 0.25
TARGET_LONG_EDGE_MAX_FRACTION = 0.38
MAX_ROTATION_DEGREES = 4.0
TABLE_MARGIN_MIN_PX = 80
TABLE_MARGIN_FRACTION = 0.03
PAPER_BORDER_MIN_PX = 24
PAPER_BORDER_FRACTION = 0.045
SHADOW_X_OFFSET_PX = 18
SHADOW_Y_OFFSET_PX = 22
SHADOW_BLUR_RADIUS_PX = 18
SHADOW_ALPHA = 85


@dataclass(frozen=True)
class Layout:
    key: str
    source_rel: str
    output_rel: str
    image_size: tuple[int, int]
    quad: list[tuple[float, float]]
    angle_deg: float
    border_px: int
    print_size: tuple[int, int]
    center: tuple[float, float]


def _list_source_images(source_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _copy_source_album(source_dir: Path, originals_dir: Path) -> None:
    originals_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, originals_dir / path.name)


def _stable_rng(seed: int, name: str) -> random.Random:
    digest = sha1(f"{seed}:{name}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _rotate_point(
    x: float, y: float, cx: float, cy: float, angle_rad: float
) -> tuple[float, float]:
    dx = x - cx
    dy = y - cy
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (
        dx * cos_a - dy * sin_a + cx,
        dx * sin_a + dy * cos_a + cy,
    )


def _order_quad(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    tl = min(points, key=lambda p: (p[0] + p[1], p[1], p[0]))
    br = max(points, key=lambda p: (p[0] + p[1], p[1], p[0]))
    tr = min(points, key=lambda p: (p[1] - p[0], p[1], -p[0]))
    bl = max(points, key=lambda p: (p[1] - p[0], p[1], -p[0]))
    return [tl, tr, br, bl]


def _load_rgba(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGBA")


def _make_paper(
    src: Image.Image, rng: random.Random, bg_size: tuple[int, int]
) -> tuple[Image.Image, Layout, tuple[int, int]]:
    bg_w, bg_h = bg_size
    target_long_edge = rng.randint(
        int(bg_w * TARGET_LONG_EDGE_MIN_FRACTION),
        int(bg_w * TARGET_LONG_EDGE_MAX_FRACTION),
    )
    scale = target_long_edge / max(src.size)
    inner_w = max(1, round(src.size[0] * scale))
    inner_h = max(1, round(src.size[1] * scale))
    resized = src.resize((inner_w, inner_h), Image.Resampling.LANCZOS)

    border_px = max(
        PAPER_BORDER_MIN_PX, round(min(inner_w, inner_h) * PAPER_BORDER_FRACTION)
    )
    paper_w = inner_w + border_px * 2
    paper_h = inner_h + border_px * 2
    paper = Image.new("RGBA", (paper_w, paper_h), (244, 239, 230, 255))
    paper.paste(resized, (border_px, border_px), resized)

    angle_deg = rng.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
    rotated = paper.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )

    corners = [
        (0.0, 0.0),
        (paper_w * 1.0, 0.0),
        (paper_w * 1.0, paper_h * 1.0),
        (0.0, paper_h * 1.0),
    ]
    cx = paper_w / 2
    cy = paper_h / 2
    rotated_corners = [
        _rotate_point(x, y, cx, cy, math.radians(angle_deg)) for x, y in corners
    ]
    min_x = min(x for x, _ in rotated_corners)
    min_y = min(y for _, y in rotated_corners)
    max_x = max(x for x, _ in rotated_corners)
    max_y = max(y for _, y in rotated_corners)
    rotated_w = rotated.size[0]
    rotated_h = rotated.size[1]

    margin_x = max(TABLE_MARGIN_MIN_PX, round(bg_w * TABLE_MARGIN_FRACTION))
    margin_y = max(TABLE_MARGIN_MIN_PX, round(bg_h * TABLE_MARGIN_FRACTION))
    left_min = margin_x
    top_min = margin_y
    left_max = bg_w - margin_x - rotated_w
    top_max = bg_h - margin_y - rotated_h
    if left_max <= left_min or top_max <= top_min:
        raise ValueError("Table background is too small for the requested paper size")

    center_x = rng.uniform(left_min + rotated_w / 2, left_max + rotated_w / 2)
    center_y = rng.uniform(top_min + rotated_h / 2, top_max + rotated_h / 2)
    left = round(center_x - rotated_w / 2)
    top = round(center_y - rotated_h / 2)

    quad = _order_quad(
        [(x - min_x + left, y - min_y + top) for x, y in rotated_corners]
    )
    layout = Layout(
        key="",
        source_rel="",
        output_rel="",
        image_size=(bg_w, bg_h),
        quad=quad,
        angle_deg=round(angle_deg, 3),
        border_px=border_px,
        print_size=(paper_w, paper_h),
        center=(round(center_x, 2), round(center_y, 2)),
    )
    return rotated, layout, (left, top)


def _composite(
    background: Image.Image, paper: Image.Image, left: int, top: int
) -> Image.Image:
    canvas = background.copy()

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    alpha = paper.getchannel("A")
    shadow_mask = alpha.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS_PX))
    shadow_layer = Image.new("RGBA", paper.size, (0, 0, 0, SHADOW_ALPHA))
    shadow_layer.putalpha(shadow_mask)
    shadow.paste(shadow_layer, (left + SHADOW_X_OFFSET_PX, top + SHADOW_Y_OFFSET_PX), shadow_layer)

    canvas = Image.alpha_composite(canvas, shadow)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay.paste(paper, (left, top), paper)
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def _write_preview(outputs: list[tuple[str, Path]], preview_path: Path) -> None:
    if not outputs:
        return

    cols = 5
    thumb_w = 320
    thumb_h = 220
    label_h = 36
    rows = math.ceil(len(outputs) / cols)
    sheet = Image.new(
        "RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (245, 240, 232)
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for idx, (name, path) in enumerate(outputs):
        row = idx // cols
        col = idx % cols
        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)
        with Image.open(path) as img:
            thumb = ImageOps.contain(img.convert("RGB"), (thumb_w - 20, thumb_h - 20))
        px = x0 + (thumb_w - thumb.width) // 2
        py = y0 + (thumb_h - thumb.height) // 2
        sheet.paste(thumb, (px, py))
        draw.text((x0 + 8, y0 + thumb_h + 6), name, fill=(40, 32, 22), font=font)

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_path, quality=92)


def generate(seed: int) -> None:
    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(f"Missing background image: {BACKGROUND_PATH}")

    source_dir = (
        ORIGINALS_ALBUM
        if ORIGINALS_ALBUM.exists() and _list_source_images(ORIGINALS_ALBUM)
        else SOURCE_ALBUM
    )
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source album: {source_dir}")

    source_images = _list_source_images(source_dir)
    if not source_images:
        raise FileNotFoundError(f"No images found in {source_dir}")

    if source_dir == SOURCE_ALBUM:
        _copy_source_album(source_dir, ORIGINALS_ALBUM)

    background = _load_rgba(BACKGROUND_PATH)
    background_size = background.size

    layouts: dict[str, dict[str, object]] = {}
    gt_items: dict[str, dict[str, object]] = {}
    outputs: list[tuple[str, Path]] = []

    for src_path in source_images:
        rng = _stable_rng(seed, src_path.name)
        src_img = _load_rgba(src_path)
        paper, layout, (left, top) = _make_paper(src_img, rng, background_size)
        composed = _composite(background, paper, left, top)

        out_path = SOURCE_ALBUM / src_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed.save(out_path, quality=95, subsampling=0)

        key = f"album/{src_path.name}"
        quad = [[round(x, 2), round(y, 2)] for x, y in layout.quad]
        gt_items[key] = {
            "image_rel": f"raw/album/{src_path.name}",
            "image_size": [background_size[0], background_size[1]],
            "quad": quad,
        }
        layouts[key] = {
            "source_rel": f"originals/album/{src_path.name}",
            "output_rel": f"data/raw/album/{src_path.name}",
            "image_size": [background_size[0], background_size[1]],
            "quad": quad,
            "angle_deg": layout.angle_deg,
            "border_px": layout.border_px,
            "print_size": [layout.print_size[0], layout.print_size[1]],
            "center": [layout.center[0], layout.center[1]],
        }
        outputs.append((src_path.name, out_path))

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.write_text(
        json.dumps({"version": 1, "items": gt_items}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    LAYOUTS_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "background_rel": "wood-table-background.png",
                "items": layouts,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_preview(outputs, PREVIEW_PATH)

    print(
        f"Wrote {len(outputs)} composed image(s) to {SOURCE_ALBUM.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Wrote frame-crop ground truth to {GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(f"Wrote placement manifest to {LAYOUTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote preview to {PREVIEW_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic placement seed (default: 42)",
    )
    args = parser.parse_args()
    generate(args.seed)


if __name__ == "__main__":
    main()
