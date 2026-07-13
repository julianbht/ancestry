"""
Frame crop step: detect and crop the physical photo print from the input image.

For each JPEG in data/raw/, uses the rotated version from data/rotated/ if one
exists, otherwise falls back to the raw file. Detects the rectangular photo
boundary and crops an axis-aligned rectangle around it, saved in data/frames/.

The detection method is selected via config (step.yaml: method: canny|saturation|sam).
Each method has its own config file (e.g. config/frame_crop/canny.yaml).

If no quadrilateral is detected, the full image is saved unchanged and the
state entry records crop_found=false.
"""

import json
import shutil
from pathlib import Path

import cv2
from loguru import logger

from pipeline.frame_crop.config import FrameCropConfig
from pipeline.frame_crop.methods.base import DetectionMethod
from pipeline.frame_crop.methods.canny import CannyConfig, CannyMethod
from pipeline.frame_crop.methods.sam import SamConfig, SamMethod
from pipeline.frame_crop.methods.saturation import SaturationConfig, SaturationMethod
from pipeline.frame_crop.processing import CropDecision, crop_image
from pipeline.frame_crop.sidecar import DetectionSidecar, FrameCropSidecar
from pipeline.shared import state as state_lib
from pipeline.shared.config import load as load_config
from pipeline.shared.config import load_file
from pipeline.shared.hashing import sha256_file
from pipeline.shared.log import setup
from pipeline.shared.paths import CONFIG_DIR, CURATED_DIR, DATA_DIR, DEBUG_DIR, STEPS_DIR, rel as project_rel

RAW_DIR = DATA_DIR / "raw"
ROTATED_DIR = STEPS_DIR / "rotate"
FRAMES_DIR = STEPS_DIR / "frame_crop"
STEP_DEBUG_DIR = DEBUG_DIR / "frame_crop"
STATE_FILE = state_lib.STATE_DIR / "frame_crop.json"


def _resolve_input(file_rel: Path) -> Path:
    rotated = ROTATED_DIR / file_rel
    return rotated if rotated.exists() else RAW_DIR / file_rel


def _write_sidecar(out_path: Path, input_path: Path, decision: CropDecision) -> None:
    """Persist the detection metadata next to each crop so downstream steps can
    consume it (score, model box, quad). One JSON per output image.

    crop_rect_xywh is the resolved crop rect in source_image's own pixel
    coordinates — downstream steps add it to their own locally-detected boxes
    to express them in that same shared coordinate space, without needing to
    recompute it from the quad and current config themselves."""
    sidecar = FrameCropSidecar(
        source_image=str(project_rel(input_path)),
        crop_image=str(project_rel(out_path)),
        crop_found=decision.crop_found,
        crop_rect_xywh=decision.crop_rect_xywh,
        detection=DetectionSidecar.from_detection(decision.detection),
        sha256=sha256_file(out_path),
    )
    out_path.with_suffix(".json").write_text(sidecar.model_dump_json(indent=2))


def _build_method(config: FrameCropConfig) -> DetectionMethod:
    name = config.method
    if name == "canny":
        method_config = load_file(CONFIG_DIR / "frame_crop" / "canny.yaml", CannyConfig)
        return CannyMethod(
            method_config,
            config.min_area_fraction,
            config.max_area_fraction,
            config.debug.annotation,
        )
    if name == "saturation":
        method_config = load_file(
            CONFIG_DIR / "frame_crop" / "saturation.yaml", SaturationConfig
        )
        return SaturationMethod(
            method_config,
            config.min_area_fraction,
            config.max_area_fraction,
            config.debug.annotation,
        )
    if name == "sam":
        method_config = load_file(CONFIG_DIR / "frame_crop" / "sam.yaml", SamConfig)
        return SamMethod(method_config, config.min_area_fraction, config.max_area_fraction)
    raise ValueError(f"Unknown detection method: {name!r}")


def run() -> None:
    setup("frame_crop")

    try:
        config = load_config("frame_crop", FrameCropConfig)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return

    logger.info(
        f"Config: method={config.method}, "
        f"max_files={config.max_files_to_crop or 'unlimited'}, "
        f"area=({config.min_area_fraction}–{config.max_area_fraction}), "
        f"margin_px={config.margin_px}"
    )

    if STEP_DEBUG_DIR.exists():
        shutil.rmtree(STEP_DEBUG_DIR)

    all_raws = sorted(RAW_DIR.rglob("*.jpg"))

    if config.only_file:
        target = RAW_DIR / config.only_file
        all_raws = [p for p in all_raws if p == target]
        if not all_raws:
            logger.error(f"only_file={config.only_file!r} not found in {RAW_DIR}")
            return

    if config.only_ground_truth:
        gt_path = CURATED_DIR / "frame_crop" / "ground_truth.json"
        gt = json.loads(gt_path.read_text())
        gt_keys = {Path(k) for k in gt["items"]}
        all_raws = [p for p in all_raws if p.relative_to(RAW_DIR) in gt_keys]

    logger.info(f"Processing {len(all_raws)} JPEG(s)")

    state = state_lib.load(STATE_FILE)

    # Build the detection method only when at least one file still needs
    # processing. The SAM method pulls in torch + a multi-GB checkpoint; a
    # fully-processed run (e.g. the quickstart data) skips it entirely, so the
    # pipeline runs end-to-end without SAM/torch installed. The loop below still
    # visits every file and logs it as already-done, so the run stays informative.
    def _needs_processing(raw_path: Path) -> bool:
        key = raw_path.relative_to(RAW_DIR).as_posix()
        return config.ignore_state or not state_lib.is_done(state, key)

    method: DetectionMethod | None = None
    if any(_needs_processing(p) for p in all_raws):
        try:
            method = _build_method(config)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error(str(e))
            return
    else:
        logger.info(
            f"All files already processed — {config.method} detector not loaded "
            "(nothing to detect)"
        )

    total_cropped = 0
    total_uncropped = 0
    total_skipped = 0
    total_failed = 0
    total_attempted = 0

    for idx, raw_path in enumerate(all_raws, start=1):
        if (
            config.max_files_to_crop is not None
            and total_attempted >= config.max_files_to_crop
        ):
            break

        progress = f"[{idx}/{len(all_raws)}]"
        file_rel = raw_path.relative_to(RAW_DIR)
        key = file_rel.as_posix()

        if not config.ignore_state and state_lib.is_done(state, key):
            logger.debug(f"{progress} Skipping {project_rel(raw_path)} (already done)")
            total_skipped += 1
            continue

        total_attempted += 1
        assert method is not None  # built above whenever any file needs work
        input_path = _resolve_input(file_rel)
        out_path = FRAMES_DIR / file_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = cv2.imread(str(input_path))
            if img is None:
                raise ValueError(f"Could not read image: {project_rel(input_path)}")

            debug_dir = STEP_DEBUG_DIR / file_rel.with_suffix("")
            decision = crop_image(img, method, config, debug_dir)

            if decision.crop_found:
                logger.success(f"{progress} Cropped {project_rel(input_path)} (via {decision.detection.info})")
                total_cropped += 1
            else:
                logger.warning(
                    f"{progress} No quad detected in {project_rel(input_path)} "
                    f"({decision.detection.info}) — saving full image"
                )
                total_uncropped += 1

            if not config.skip_output_write:
                cv2.imwrite(
                    str(out_path),
                    decision.result,
                    [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
                )
                _write_sidecar(out_path, input_path, decision)

            if not config.skip_state_write:
                outputs = [str(out_path)] if not config.skip_output_write else []
                state_lib.mark_done(state, key, outputs)
                state["processed"][key]["crop_found"] = decision.crop_found
                state_lib.save(state, STATE_FILE)

        except Exception as e:
            if not config.skip_state_write:
                state_lib.mark_failed(state, key, str(e))
                state_lib.save(state, STATE_FILE)
            logger.error(f"{progress} Failed {project_rel(input_path)}: {e}")
            total_failed += 1

    if total_attempted > 0:
        crop_rate = 100 * total_cropped / total_attempted
        uncropped_rate = 100 * total_uncropped / total_attempted
        outcome = (
            f"{total_attempted} attempted: "
            f"{total_cropped} cropped ({crop_rate:.0f}%), "
            f"{total_uncropped} no frame detected — full image saved ({uncropped_rate:.0f}%)"
        )
    else:
        outcome = "0 attempted"
    logger.info(
        f"Done — {outcome}; "
        f"{total_skipped} skipped (already done), {total_failed} failed (errors)"
    )


if __name__ == "__main__":
    run()
