"""
Face recognition step: identify each face crop against a gallery built from
ground-truth annotations.

For every detected face (produced by face_crop), extracts an embedding via the
configured method (DeepFace or InsightFace) and finds the nearest neighbour in
the gallery. Faces within recognition_threshold of the closest gallery entry
are assigned that person's ID; everything else is marked unknown.

The gallery is built once per run from ground_truth.json: for each annotated
face with a person_id, the best-matching face_crop detection (by IoU on
source-image coordinates) becomes a reference embedding for that person.

Input:  data/steps/face_crop/<folder>/<stem>/faces.json + face_NN.jpg crops
Output: data/steps/face_recognition/<folder>/<stem>/recognition.json
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from pipeline.face_recognition.config import FaceRecognitionConfig
from pipeline.face_recognition.debug import label_name, save_recognition_overlay
from pipeline.face_recognition.gallery import build_gallery
from pipeline.face_recognition.methods.base import EmbeddingMethod
from pipeline.face_recognition.recognizer import FaceRecognizer
from pipeline.face_recognition.sidecar import RecognitionEntry, RecognitionSidecar, RecognitionStatus
from pipeline.gramps import Person, load_family_tree
from pipeline.shared import state as state_lib
from pipeline.shared.config import load as load_config, load_file
from pipeline.shared.log import setup
from pipeline.shared.paths import CONFIG_DIR, CURATED_DIR, DEBUG_DIR, MODELS_DIR, PROJECT_ROOT, STEPS_DIR, rel as project_rel

FACES_DIR = STEPS_DIR / "face_crop"
RECOGNITION_DIR = STEPS_DIR / "face_recognition"
STEP_DEBUG_DIR = DEBUG_DIR / "face_recognition"
STATE_FILE = state_lib.STATE_DIR / "face_recognition.json"
GROUND_TRUTH_FILE = CURATED_DIR / "face_annotation" / "ground_truth.json"


def _build_method(config: FaceRecognitionConfig) -> EmbeddingMethod:
    method_config_path = CONFIG_DIR / "face_recognition" / f"{config.method}.yaml"
    if config.method == "deepface":
        from pipeline.face_recognition.config import DeepFaceConfig
        from pipeline.face_recognition.methods.deepface import DeepFaceMethod

        mc = load_file(method_config_path, DeepFaceConfig)
        return DeepFaceMethod(
            model_name=mc.model_name,
            detector_backend=mc.detector_backend,
            models_dir=MODELS_DIR / "deepface",
        )
    if config.method == "insightface":
        from pipeline.face_recognition.config import InsightFaceConfig
        from pipeline.face_recognition.methods.insightface import InsightFaceMethod

        mc = load_file(method_config_path, InsightFaceConfig)
        return InsightFaceMethod(
            model_pack=mc.model_pack,
            models_dir=MODELS_DIR / "insightface",
            det_size=mc.det_size,
            pad_ratio=mc.pad_ratio,
            det_thresh=mc.det_thresh,
        )
    raise ValueError(f"Unknown face recognition method: {config.method!r}. Expected 'deepface' or 'insightface'.")


def _display_name(person: Person, abbreviate_surname: bool) -> str:
    """First given name + surname (dropping middle given names), e.g. "Renate
    Boldt", or "Renate B." when abbreviate_surname is set."""
    first_given = person.given.split()[0] if person.given else ""
    surname = f"{person.surname[0]}." if abbreviate_surname and person.surname else person.surname
    return " ".join(p for p in (first_given, surname) if p) or person.id


def run() -> None:
    setup("face_recognition")

    try:
        config = load_config("face_recognition", FaceRecognitionConfig)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return

    logger.info(
        f"Config: method={config.method!r}, "
        f"metric={config.distance_metric!r}, threshold={config.recognition_threshold}, "
        f"gallery_overlap={config.gallery_overlap_threshold}, "
        f"max_files={config.max_files_to_recognize or 'unlimited'}"
    )

    if STEP_DEBUG_DIR.exists():
        shutil.rmtree(STEP_DEBUG_DIR)

    all_sidecars = sorted(FACES_DIR.rglob("faces.json"))
    state = state_lib.load(STATE_FILE)

    def _needs_processing(sidecar_path: Path) -> bool:
        frame_key = sidecar_path.parent.relative_to(FACES_DIR).as_posix()
        return config.ignore_state or not state_lib.is_done(state, frame_key)

    # Building the embedding model downloads + loads weights, and building the
    # gallery embeds every labeled reference crop. Do this only when at least one
    # frame still needs recognition; a fully-processed run (e.g. the quickstart
    # data) skips it all, so the pipeline runs end-to-end without the model. The
    # loop below still visits every frame and logs it as already-done, so the run
    # stays informative.
    recognizer: FaceRecognizer | None = None
    gallery: dict[str, list[np.ndarray]] | None = None
    id_to_name: dict[str, str] | None = None
    params_line: str | None = None

    if any(_needs_processing(s) for s in all_sidecars):
        method = _build_method(config)
        try:
            gallery = build_gallery(
                GROUND_TRUTH_FILE,
                FACES_DIR,
                config.gallery_overlap_threshold,
                method,
            )
        except Exception as e:
            logger.error(f"Gallery build failed: {e}")
            return

        if not gallery:
            logger.error(
                "Gallery is empty — no reference crops could be matched. "
                "Check that ground_truth.json entries overlap with face_crop output "
                "and that gallery_overlap_threshold is not too high."
            )
            return

        gallery_crop_count = sum(len(v) for v in gallery.values())
        logger.info(f"Gallery ready: {len(gallery)} people, {gallery_crop_count} reference crops")

        recognizer = FaceRecognizer(
            gallery=gallery,
            method=method,
            distance_metric=config.distance_metric,
            threshold=config.recognition_threshold,
            top_k=config.top_k_candidates,
        )

        id_to_name = {
            p.id: _display_name(p, config.debug.abbreviate_surname) for p in load_family_tree()
        }
        params_line = (
            f"method={method.name} "
            f"metric={config.distance_metric} threshold={config.recognition_threshold}"
        )
    else:
        logger.info(
            "All frames already recognized — embedding model & gallery not loaded "
            "(nothing to recognize)"
        )

    total_attempted = 0
    total_skipped = 0
    total_failed = 0
    total_recognized = 0
    total_unknown = 0
    total_no_embedding = 0

    for idx, sidecar_path in enumerate(all_sidecars, start=1):
        if (
            config.max_files_to_recognize is not None
            and total_attempted >= config.max_files_to_recognize
        ):
            break

        progress = f"[{idx}/{len(all_sidecars)}]"

        # Key is the frame directory relative to FACES_DIR, e.g.
        # "3fBXinPqztWt3CP/20260421_185307"
        frame_key = sidecar_path.parent.relative_to(FACES_DIR).as_posix()

        if not config.ignore_state and state_lib.is_done(state, frame_key):
            logger.debug(f"{progress} Skipping {project_rel(sidecar_path)} (already done)")
            total_skipped += 1
            continue

        total_attempted += 1
        # All built above whenever any frame needs work (this branch only runs
        # for frames that need work).
        assert recognizer is not None and gallery is not None
        assert id_to_name is not None and params_line is not None

        try:
            sidecar_data = json.loads(sidecar_path.read_text())
            faces_raw = sidecar_data.get("faces", [])
        except Exception as e:
            logger.warning(
                f"{progress} Could not parse {project_rel(sidecar_path)}: {e} — skipping frame"
            )
            if not config.skip_state_write:
                state_lib.mark_failed(state, frame_key, str(e))
                state_lib.save(state, STATE_FILE)
            total_failed += 1
            continue

        recognitions: list[RecognitionEntry] = []
        # Row i aligns with recognitions[i]; None for faces whose embedding failed.
        embeddings_rows: list[np.ndarray | None] = []
        for face in faces_raw:
            crop_image = face["crop_image"]
            result = recognizer.recognize(Path(crop_image))
            recognitions.append(
                RecognitionEntry(
                    face_index=face["index"],
                    crop_image=crop_image,
                    person_id=result.person_id,
                    distance=result.distance,
                    status=result.status,
                    candidates=result.candidates,
                    age=result.age,
                    gender=result.gender,
                    det_score=result.det_score,
                    # Carried from face_crop's faces.json (FaceCropEntry.sha256);
                    # face_crop always writes it and the backfill fills legacy
                    # sidecars, so a missing key is a real error worth surfacing.
                    crop_sha256=face["sha256"],
                )
            )
            embeddings_rows.append(result.embedding)
            if result.status == RecognitionStatus.RECOGNIZED:
                total_recognized += 1
            elif result.status == RecognitionStatus.UNKNOWN:
                total_unknown += 1
            else:
                total_no_embedding += 1

        if recognitions:
            summaries = []
            for rec in recognitions:
                text = label_name(rec, id_to_name)
                if rec.distance is not None:
                    text += f" ({rec.distance:.2f})"
                summaries.append(text)
            logger.success(f"{progress} {sidecar_data['source_image']}: {', '.join(summaries)}")
        else:
            logger.debug(f"{progress} {sidecar_data['source_image']}: no faces")

        if config.debug.save_overlay and recognitions:
            frame_path = PROJECT_ROOT / sidecar_data["source_image"]
            img = cv2.imread(str(frame_path))
            if img is None:
                logger.warning(f"Debug overlay: could not read {project_rel(frame_path)}")
            else:
                boxes = [tuple(face["box_xyxy"]) for face in faces_raw]
                save_recognition_overlay(
                    img, boxes, recognitions, id_to_name, params_line,
                    config.debug.font_scale, STEP_DEBUG_DIR / f"{frame_key}.jpg",
                )

        outputs: list[str] = []
        if not config.skip_output_write:
            out_dir = RECOGNITION_DIR / frame_key
            out_dir.mkdir(parents=True, exist_ok=True)

            # Persist query embeddings (the expensive artifact) aligned by row to
            # recognitions; failed embeddings become a NaN row. Skip the file only
            # when no face in this frame embedded (dim unknown, nothing to store).
            embeddings_file: str | None = None
            dim = next((e.shape[0] for e in embeddings_rows if e is not None), None)
            if dim is not None:
                arr = np.full((len(embeddings_rows), dim), np.nan, dtype=np.float32)
                for i, e in enumerate(embeddings_rows):
                    if e is not None:
                        arr[i] = e
                emb_path = out_dir / "embeddings.npy"
                np.save(emb_path, arr)
                embeddings_file = str(project_rel(emb_path))

            rec_sidecar = RecognitionSidecar(
                source_sidecar=str(project_rel(sidecar_path)),
                model=method.name,
                detector_backend=config.method,
                distance_metric=config.distance_metric,
                threshold=config.recognition_threshold,
                gallery_size=len(gallery),
                embeddings_file=embeddings_file,
                recognitions=recognitions,
            )
            rec_path = out_dir / "recognition.json"
            rec_path.write_text(rec_sidecar.model_dump_json(indent=2))
            outputs = [str(rec_path)]
            if embeddings_file is not None:
                outputs.append(str(emb_path))

        if not config.skip_state_write:
            state_lib.mark_done(state, frame_key, outputs)
            state_lib.save(state, STATE_FILE)

    total_faces = total_recognized + total_unknown + total_no_embedding
    if total_attempted > 0 and total_faces > 0:
        recog_rate = 100 * total_recognized / total_faces
        unknown_rate = 100 * total_unknown / total_faces
        outcome = (
            f"{total_attempted} frames, {total_faces} faces: "
            f"{total_recognized} recognized ({recog_rate:.0f}%), "
            f"{total_unknown} unknown ({unknown_rate:.0f}%)"
        )
        if total_no_embedding:
            outcome += f", {total_no_embedding} embedding failures"
    else:
        outcome = "0 attempted"

    logger.info(
        f"Done — {outcome}; "
        f"{total_skipped} skipped (already done), {total_failed} failed (errors)"
    )


if __name__ == "__main__":
    run()
