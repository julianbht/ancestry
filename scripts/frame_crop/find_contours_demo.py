"""
Demonstrates cv2.findContours behaviour on synthetic binary images.

Cases:
  1. single line       - 1px horizontal line, 256px wide
  2. closed rectangle  - rectangular outline, fully closed
  3. open rectangle    - same outline but with a 30px gap on one side
  4. filled rectangle  - solid filled rectangle

Each output image has two panels side by side:
  left  - original binary image (what findContours receives)
  right - contours drawn in random colors on a black background

Saved to data/debug/find_contours_demo/.

Usage:
    uv run python scripts/frame_crop/find_contours_demo.py
"""

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "debug" / "find_contours_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 600, 400
IMAGE_AREA = W * H


def _report(name: str, img: np.ndarray) -> None:
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)

    print(f"\n-- {name} --")
    print(f"  contours found : {len(contours)}")
    if areas:
        print(f"  largest area   : {areas[0]:.0f} px2  ({areas[0]/IMAGE_AREA:.2%} of image)")
        print(f"  all areas      : {[f'{a:.0f}' for a in areas[:10]]}"
              + (" ..." if len(areas) > 10 else ""))
    else:
        print("  (no contours)")

    # Left panel: original binary image
    left = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Right panel: contours in random colors on black background
    right = np.zeros_like(left)
    rng = np.random.default_rng(seed=42)
    for c in contours:
        color = tuple(int(x) for x in rng.integers(80, 256, size=3))
        cv2.drawContours(right, [c], -1, color, 1)

    divider = np.full((H, 4, 3), 80, dtype=np.uint8)
    combined = np.hstack([left, divider, right])

    # Label bar above panels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_bar = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
    for text, x in [("original", 10), ("contours", W + 14)]:
        cv2.putText(label_bar, text, (x, 24), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Stats bar below panels
    stats_bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
    stats = (
        f"{name}  |  contours: {len(contours)}  |  "
        f"largest: {areas[0]:.0f} px2 ({areas[0]/IMAGE_AREA:.2%})"
        if areas else f"{name}  |  no contours"
    )
    cv2.putText(stats_bar, stats, (10, 20), font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    out = np.vstack([label_bar, combined, stats_bar])

    filename = name.lower().replace(" ", "_") + ".jpg"
    cv2.imwrite(str(OUT_DIR / filename), out)
    print(f"  saved -> {OUT_DIR / filename}")


# -- 1. Single horizontal line ------------------------------------------------
img = np.zeros((H, W), dtype=np.uint8)
cv2.line(img, (50, H // 2), (50 + 256, H // 2), 255, 1)
_report("single line", img)

# -- 2. Closed rectangle ------------------------------------------------------
img = np.zeros((H, W), dtype=np.uint8)
cv2.rectangle(img, (80, 80), (W - 80, H - 80), 255, 1)
_report("closed rectangle", img)

# -- 3. Open rectangle (30px gap on top edge) ---------------------------------
img = np.zeros((H, W), dtype=np.uint8)
cv2.rectangle(img, (80, 80), (W - 80, H - 80), 255, 1)
gap_cx = W // 2
img[80, gap_cx - 15 : gap_cx + 15] = 0
_report("open rectangle", img)

# -- 4. Filled rectangle ------------------------------------------------------
img = np.zeros((H, W), dtype=np.uint8)
cv2.rectangle(img, (80, 80), (W - 80, H - 80), 255, cv2.FILLED)
_report("filled rectangle", img)

print(f"\nAll images written to {OUT_DIR}")
