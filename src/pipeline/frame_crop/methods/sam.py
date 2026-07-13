from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from pydantic import Field

from pipeline.frame_crop.detect import quad_from_mask
from pipeline.frame_crop.methods.base import Detection
from pipeline.shared.config import StrictConfig
from pipeline.shared.paths import MODELS_DIR

_SAM3_CHECKPOINT = MODELS_DIR / "sam3" / "sam3.pt"


class SamConfig(StrictConfig):
    prompt: str
    score_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    # Path to the SAM 3 image checkpoint. Keeps its default (a derived filesystem
    # path) rather than being spelled out in YAML — see setup_sam3.py.
    checkpoint_path: Path = _SAM3_CHECKPOINT
    device: str


class SamMethod:
    def __init__(
        self,
        config: SamConfig,
        min_area_fraction: float,
        max_area_fraction: float,
    ) -> None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise RuntimeError(
                "SAM3 is not installed. Run: uv run python scripts/setup_sam3.py"
            ) from e

        if not config.checkpoint_path.exists():
            raise RuntimeError(
                f"SAM3 checkpoint not found at {config.checkpoint_path}. "
                "Run: uv run python scripts/setup_sam3.py"
            )

        model = build_sam3_image_model(
            checkpoint_path=str(config.checkpoint_path),
            load_from_HF=False,
            device=config.device,
            eval_mode=True,
        )
        self._processor = Sam3Processor(model)
        self._config = config
        self._min_area_fraction = min_area_fraction
        self._max_area_fraction = max_area_fraction

    def detect(
        self, img: np.ndarray, debug_dir: Path | None = None
    ) -> Detection:
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # SAM3's image model force-casts backbone features to bfloat16 internally
        # (sam3_image.py), assuming inference runs under AMP autocast — as every
        # SAM3 example notebook does. Without this context the bf16 features hit
        # float32 weights: "mat1 and mat2 must have the same dtype" (facebookresearch/sam3#507).
        device_type = "cuda" if "cuda" in self._config.device else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            state = self._processor.set_image(pil_img)
            output = self._processor.set_text_prompt(state=state, prompt=self._config.prompt)

        masks = output["masks"]   # (N, 1, H, W) bool or float tensor
        scores = output["scores"] # (N,) float tensor
        boxes = output["boxes"]   # (N, 4) xyxy in original-image pixels

        if masks is None or len(masks) == 0:
            return Detection(None, "sam: no masks returned", prompt=self._config.prompt)

        scores_np = scores.cpu().float().numpy()
        masks_np = masks.cpu().float().numpy()
        boxes_np = boxes.cpu().float().numpy()

        # Try masks in descending score order; return the first that fits area constraints
        order = scores_np.argsort()[::-1]
        candidates = [int(i) for i in order if scores_np[i] >= self._config.score_threshold]

        if debug_dir is not None and candidates:
            self._save_mask_debug(img, masks_np, scores_np, boxes_np, candidates, debug_dir)

        if not candidates:
            return Detection(
                None,
                f"sam: no mask above score_threshold {self._config.score_threshold} "
                f"(best {scores_np[order[0]]:.2f})",
                prompt=self._config.prompt,
            )

        best_reason: str | None = None
        for idx in candidates:
            mask_uint8 = self._mask_to_uint8(masks_np[idx])
            quad, reason = quad_from_mask(mask_uint8, self._min_area_fraction, self._max_area_fraction)
            if quad is not None:
                return Detection(
                    quad=quad,
                    info="sam",
                    score=float(scores_np[idx]),
                    box_xyxy=tuple(float(v) for v in boxes_np[idx]),
                    prompt=self._config.prompt,
                )
            if best_reason is None:  # reason for the highest-scoring candidate
                best_reason = reason

        return Detection(
            None,
            f"sam: {best_reason} (best score {scores_np[candidates[0]]:.2f})",
            prompt=self._config.prompt,
        )

    @staticmethod
    def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
        # SAM3 returns masks as (N, 1, H, W); drop the singleton channel dim
        # so findContours gets a 2-D image (it rejects 3-D input).
        return (np.squeeze(mask) > 0.5).astype(np.uint8) * 255

    # Overlay colors (BGR). Bright, high-contrast hues that stand out against
    # the wooden-table background the prints are photographed on.
    _MASK_TINT = (0, 255, 0)      # green   — SAM's segmentation mask
    _SAM_BOX_COLOR = (255, 0, 255)  # magenta — SAM's axis-aligned box (raw output)
    _FIT_BOX_COLOR = (255, 255, 0)  # cyan    — minAreaRect fitted to the mask (crop quad)

    def _save_mask_debug(
        self,
        img: np.ndarray,
        masks_np: np.ndarray,
        scores_np: np.ndarray,
        boxes_np: np.ndarray,
        candidates: list[int],
        debug_dir: Path,
    ) -> None:
        """Save two debug overlays of the SAM step. Both show the segmentation
        mask (green) with a prompt header and per-candidate score/area labels;
        they differ only in the box drawn:
          - sam_box.jpg — SAM's raw axis-aligned box (magenta).
          - fit_box.jpg — the rotated minAreaRect fitted to the mask (cyan), the
            quad the crop is derived from.
        Kept as separate images so each can stand alone on a presentation slide."""
        img_h, img_w = img.shape[:2]
        image_area = img_h * img_w
        scale = max(1.0, img_w / 1000)
        thickness = max(2, int(scale * 1.5))
        font = cv2.FONT_HERSHEY_SIMPLEX

        sam_overlay = img.copy()  # mask + SAM's axis-aligned box
        fit_overlay = img.copy()  # mask + fitted rotated crop box

        # Prompt header, top-left. Reserve the band it occupies so per-box labels
        # drop below it rather than overlapping it.
        (_, th), base = cv2.getTextSize("Ag", font, scale, thickness)
        header = f"prompt: {self._config.prompt!r}"
        header_baseline = int(th * 1.8)
        header_bottom = header_baseline + base + int(th * 0.4)
        self._draw_label(sam_overlay, header, (12, header_baseline), font, scale, thickness)
        self._draw_label(fit_overlay, header, (12, header_baseline), font, scale, thickness)

        for idx in candidates:
            mask_uint8 = self._mask_to_uint8(masks_np[idx])
            tint = np.zeros_like(img)
            tint[mask_uint8 > 0] = self._MASK_TINT
            sam_overlay = cv2.addWeighted(sam_overlay, 1.0, tint, 0.4, 0)
            fit_overlay = cv2.addWeighted(fit_overlay, 1.0, tint, 0.4, 0)

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            frac = cv2.contourArea(c) / image_area
            label = f"score={scores_np[idx]:.2f} area={frac:.0%}"

            # SAM's raw axis-aligned box (magenta), anchored at its top-left.
            x0, y0, x1, y1 = (int(v) for v in boxes_np[idx])
            cv2.rectangle(sam_overlay, (x0, y0), (x1, y1), self._SAM_BOX_COLOR, thickness)
            pos = self._label_pos(x0, y0, label, font, scale, thickness, img_w, img_h, header_bottom)
            self._draw_label(sam_overlay, label, pos, font, scale, thickness)

            # Our fitted rotated box (cyan), anchored at its topmost corner.
            box = cv2.boxPoints(cv2.minAreaRect(cv2.convexHull(c))).astype(np.int32)
            cv2.polylines(fit_overlay, [box.reshape(-1, 1, 2)], True, self._FIT_BOX_COLOR, thickness)
            tx, ty = box[box[:, 1].argmin()]
            pos = self._label_pos(int(tx), int(ty), label, font, scale, thickness, img_w, img_h, header_bottom)
            self._draw_label(fit_overlay, label, pos, font, scale, thickness)

        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "sam_box.jpg"), sam_overlay)
        cv2.imwrite(str(debug_dir / "fit_box.jpg"), fit_overlay)

    @staticmethod
    def _label_pos(
        x: int,
        y: int,
        label: str,
        font: int,
        scale: float,
        thickness: int,
        img_w: int,
        img_h: int,
        top_margin: int = 0,
    ) -> tuple[int, int]:
        """Clamp a label anchored at (x, y) fully inside the image: clamp x, and
        drop the label below the anchor when there isn't room for it above the
        reserved `top_margin` band (used to keep labels clear of the header)."""
        (tw, th), base = cv2.getTextSize(label, font, scale, thickness)
        tx = int(min(max(x, 0), max(0, img_w - tw)))
        ty = y - 10
        if ty - th < top_margin:
            ty = y + th + 12
        ty = int(min(max(ty, top_margin + th), img_h - base))
        return (tx, ty)

    @staticmethod
    def _draw_label(
        img: np.ndarray,
        text: str,
        org: tuple[int, int],
        font: int,
        scale: float,
        thickness: int,
    ) -> None:
        """Draw black text on a filled white box so it stays legible over any
        background (mask tint, table, photo) — the box guarantees contrast,
        which plain or outlined text on a busy background does not."""
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        x, y = org
        pad = max(3, int(scale * 3))
        cv2.rectangle(
            img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), (255, 255, 255), cv2.FILLED
        )
        cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
