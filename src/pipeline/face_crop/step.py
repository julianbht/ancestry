"""
Face crop step: detect every face in each frame-cropped photo and save each
face as its own image.

Input is the output of frame_crop (data/steps/frame_crop/). For each frame, SAM
detects all faces above a score threshold; each face box is cropped and written
to its own file under data/steps/face_crop/<frame>/face_NN.jpg, alongside a
faces.json sidecar capturing the box coordinates, scores and prompt for every
detection (consumed downstream by face_recognition / age_estimation).

Each face's box is recorded twice: box_xyxy in frame-local pixels (this frame's
own coordinate system) and box_xyxy_source, the same box shifted by frame_crop's
crop_rect_xywh offset into the *source image's* pixel coordinates — the same
space ground_truth.json's bbox_xywh is already in. This lets a ground-truth
face be matched to a face_crop output crop with a plain IoU check, no further
coordinate transform needed.

Photos that are the *back* of another photo (handwritten notes, listed in
data/curated/photo_backs.csv) carry no faces and are excluded.
"""

import csv
import json
import shutil
from pathlib import Path

import cv2
from loguru import logger

from pipeline.face_crop.config import FaceCropConfig
from pipeline.face_crop.cropping import crop_face
from pipeline.face_crop.debug import save_face_overlay
from pipeline.face_crop.detector import FaceDetection, SamFaceDetector
from pipeline.face_crop.sidecar import FaceCropEntry, FaceCropSidecar
from pipeline.frame_crop.sidecar import FrameCropSidecar
from pipeline.shared import state as state_lib
from pipeline.shared.config import load as load_config
from pipeline.shared.hashing import sha256_file
from pipeline.shared.log import setup
from pipeline.shared.paths import CURATED_DIR, DEBUG_DIR, STEPS_DIR, rel as project_rel

FRAMES_DIR = STEPS_DIR / "frame_crop"
FACES_DIR = STEPS_DIR / "face_crop"
STEP_DEBUG_DIR = DEBUG_DIR / "face_crop"
STATE_FILE = state_lib.STATE_DIR / "face_crop.json"

PHOTO_BACKS_CSV = CURATED_DIR / "photo_backs.csv"
GROUND_TRUTH_FILE = CURATED_DIR / "face_annotation" / "ground_truth.json"


def _load_back_photo_keys() -> set[str]:
    """Relative keys ('<folder>/<file>.jpg') of photos that are the back of
    another photo and must be skipped. Missing CSV → no exclusions."""
    if not PHOTO_BACKS_CSV.exists():
        return set()
    keys: set[str] = set()
    with PHOTO_BACKS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            folder = (row.get("foldername") or "").strip()
            back = (row.get("file-with-note-on-back") or "").strip()
            if folder and back:
                keys.add(f"{folder}/{back}")
    return keys


def _ground_truth_keys() -> set[str]:
    gt = json.loads(GROUND_TRUTH_FILE.read_text())
    return set(gt["items"].keys())


def _load_frame_crop_sidecar(frame_path: Path) -> FrameCropSidecar:
    """Read frame_crop's sidecar for this frame. It carries both the crop offset
    (crop_rect_xywh, whose x/y shift faces into source-image space) and the sha256
    of the frame's bytes that face_crop verifies its input against."""
    sidecar_path = frame_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"Missing frame_crop sidecar: {project_rel(sidecar_path)} "
            "(rerun frame_crop to regenerate it)"
        )
    return FrameCropSidecar.model_validate_json(sidecar_path.read_text())


def _write_sidecar(
    out_dir: Path,
    source_image: Path,
    source_sha256: str,
    image_size: tuple[int, int],
    prompt: str,
    faces: list[FaceDetection],
    crop_paths: list[Path],
    frame_offset: tuple[int, int],
) -> Path:
    """Write faces.json describing every detected face in one frame."""
    offset_x, offset_y = frame_offset
    sidecar = FaceCropSidecar(
        source_image=str(project_rel(source_image)),
        source_sha256=source_sha256,
        image_size=image_size,
        prompt=prompt,
        face_count=len(faces),
        faces=[
            FaceCropEntry(
                index=i,
                box_xyxy=face.box_xyxy,
                score=face.score,
                box_xyxy_source=(
                    face.box_xyxy[0] + offset_x,
                    face.box_xyxy[1] + offset_y,
                    face.box_xyxy[2] + offset_x,
                    face.box_xyxy[3] + offset_y,
                ),
                crop_image=str(project_rel(crop_path)),
                sha256=sha256_file(crop_path),
            )
            for i, (face, crop_path) in enumerate(zip(faces, crop_paths))
        ],
    )
    sidecar_path = out_dir / "faces.json"
    sidecar_path.write_text(sidecar.model_dump_json(indent=2))
    return sidecar_path


def run() -> None:
    setup("face_crop")

    try:
        config = load_config("face_crop", FaceCropConfig)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return

    logger.info(
        f"Config: prompt={config.sam.prompt!r}, "
        f"score_threshold={config.sam.score_threshold}, "
        f"min_area_fraction={config.min_area_fraction}, "
        f"margin_frac={config.margin_frac}, "
        f"max_files={config.max_files_to_crop or 'unlimited'}"
    )

    if STEP_DEBUG_DIR.exists():
        shutil.rmtree(STEP_DEBUG_DIR)

    all_frames = sorted(FRAMES_DIR.rglob("*.jpg"))

    if config.only_file:
        target = FRAMES_DIR / config.only_file
        all_frames = [p for p in all_frames if p == target]
        if not all_frames:
            logger.error(f"only_file={config.only_file!r} not found in {FRAMES_DIR}")
            return

    if config.only_ground_truth:
        gt_keys = _ground_truth_keys()
        all_frames = [p for p in all_frames if p.relative_to(FRAMES_DIR).as_posix() in gt_keys]

    back_keys = _load_back_photo_keys()
    kept_frames: list[Path] = []
    total_excluded = 0
    for p in all_frames:
        if p.relative_to(FRAMES_DIR).as_posix() in back_keys:
            total_excluded += 1
        else:
            kept_frames.append(p)

    logger.info(
        f"Processing {len(kept_frames)} frame(s); "
        f"{total_excluded} excluded (photo backs)"
    )

    state = state_lib.load(STATE_FILE)

    # Load SAM (torch + a multi-GB checkpoint) only when at least one frame still
    # needs processing. A fully-processed run — e.g. the quickstart data, where
    # every crop already exists — skips the load entirely, so the pipeline runs
    # end-to-end without SAM/torch installed. The loop below still visits every
    # frame and logs it as already-done, so the run stays informative.
    def _needs_processing(frame_path: Path) -> bool:
        key = frame_path.relative_to(FRAMES_DIR).as_posix()
        return config.ignore_state or not state_lib.is_done(state, key)

    detector: SamFaceDetector | None = None
    if any(_needs_processing(p) for p in kept_frames):
        try:
            detector = SamFaceDetector(config.sam)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error(str(e))
            return
    else:
        logger.info("All frames already processed — SAM not loaded (nothing to detect)")

    total_faces = 0
    total_with_faces = 0
    total_no_face = 0
    total_skipped = 0
    total_failed = 0
    total_attempted = 0

    for idx, frame_path in enumerate(kept_frames, start=1):
        if (
            config.max_files_to_crop is not None
            and total_attempted >= config.max_files_to_crop
        ):
            break

        progress = f"[{idx}/{len(kept_frames)}]"
        file_rel = frame_path.relative_to(FRAMES_DIR)
        key = file_rel.as_posix()

        if not config.ignore_state and state_lib.is_done(state, key):
            logger.debug(f"{progress} Skipping {project_rel(frame_path)} (already done)")
            total_skipped += 1
            continue

        total_attempted += 1
        assert detector is not None  # built above whenever any frame needs work
        out_dir = FACES_DIR / file_rel.with_suffix("")

        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                raise ValueError(f"Could not read image: {project_rel(frame_path)}")

            fc_sidecar = _load_frame_crop_sidecar(frame_path)
            source_sha256 = sha256_file(frame_path)
            if source_sha256 != fc_sidecar.sha256:
                raise ValueError(
                    f"frame_crop output changed since its sidecar was written: "
                    f"{project_rel(frame_path)} (sidecar sha256 {fc_sidecar.sha256[:12]}…, "
                    f"file {source_sha256[:12]}…) — rerun frame_crop, then face_crop"
                )
            frame_offset = (fc_sidecar.crop_rect_xywh[0], fc_sidecar.crop_rect_xywh[1])
            img_h, img_w = img.shape[:2]
            frame_area = img_h * img_w
            faces = [
                f
                for f in detector.detect(img)
                if f.area >= config.min_area_fraction * frame_area
            ]

            if config.debug.save_overlay:
                save_face_overlay(
                    img, faces, config.sam.prompt, STEP_DEBUG_DIR / file_rel
                )

            if faces:
                logger.success(
                    f"{progress} {len(faces)} face(s) in {project_rel(frame_path)} "
                    f"(scores {', '.join(f'{f.score:.2f}' for f in faces)})"
                )
                total_with_faces += 1
                total_faces += len(faces)
            else:
                logger.warning(f"{progress} No face detected in {project_rel(frame_path)}")
                total_no_face += 1

            outputs: list[str] = []
            if not config.skip_output_write:
                # Clear any prior output for this frame so a re-run can't leave
                # stale crops behind (e.g. when fewer faces are detected this time).
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                crop_paths: list[Path] = []
                for i, face in enumerate(faces):
                    crop = crop_face(img, face, config.margin_frac)
                    crop_path = out_dir / f"face_{i:02d}.jpg"
                    cv2.imwrite(
                        str(crop_path),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
                    )
                    crop_paths.append(crop_path)

                sidecar_path = _write_sidecar(
                    out_dir, frame_path, source_sha256, (img_w, img_h), config.sam.prompt,
                    faces, crop_paths, frame_offset,
                )
                outputs = [str(p) for p in crop_paths] + [str(sidecar_path)]

            if not config.skip_state_write:
                state_lib.mark_done(state, key, outputs)
                state["processed"][key]["face_count"] = len(faces)
                state_lib.save(state, STATE_FILE)

        except Exception as e:
            if not config.skip_state_write:
                state_lib.mark_failed(state, key, str(e))
                state_lib.save(state, STATE_FILE)
            logger.error(f"{progress} Failed {project_rel(frame_path)}: {e}")
            total_failed += 1

    if total_attempted > 0:
        with_rate = 100 * total_with_faces / total_attempted
        no_face_rate = 100 * total_no_face / total_attempted
        outcome = (
            f"{total_attempted} attempted: "
            f"{total_faces} faces from {total_with_faces} frames with faces ({with_rate:.0f}%), "
            f"{total_no_face} with no face detected ({no_face_rate:.0f}%)"
        )
    else:
        outcome = "0 attempted"
    logger.info(
        f"Done — {outcome}; "
        f"{total_excluded} excluded (photo backs), "
        f"{total_skipped} skipped (already done), {total_failed} failed (errors)"
    )


if __name__ == "__main__":
    run()
