import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import optuna
import wandb
from loguru import logger

from pipeline.experiments.face_recognition.metrics import evaluate_loo
from pipeline.face_recognition.gallery import MatchedFace
from pipeline.face_recognition.methods.base import EmbeddingMethod
from pipeline.face_recognition.methods.deepface import DeepFaceMethod
from pipeline.face_recognition.methods.insightface import InsightFaceMethod
from pipeline.shared.paths import DATA_DIR

# Shared across both methods.
DISTANCE_METRICS = ["cosine", "euclidean", "euclidean_l2"]

# --- DeepFace search space (see pipeline/face_recognition/config.py) ---
MODEL_NAMES = ["ArcFace", "Facenet512", "VGG-Face"]
# "skip" treats the crop as-is (fastest); the rest re-detect/align within the crop.
DETECTOR_BACKENDS = ["skip", "opencv", "retinaface", "mtcnn"]

# --- InsightFace search space ---
MODEL_PACKS = ["buffalo_l", "antelopev2"]
# Square detector input sizes for FaceAnalysis.prepare(). Tuples aren't valid
# Optuna categoricals, so we search the side length and build (n, n).
DET_SIZES = [480, 640, 800]

# Model weights live alongside the production step's downloads
# (pipeline/face_recognition/step.py:_build_method) so the search reuses them.
DEEPFACE_MODELS_DIR = DATA_DIR / "models" / "deepface"
INSIGHTFACE_MODELS_DIR = DATA_DIR / "models" / "insightface"

# Objective weighting. Fbeta with beta < 1 favours precision over recall, i.e.
# gallery purity (only label a face when confident) over completeness. beta=0.5
# weights precision ~2x recall; set beta=1.0 for plain F1 (equal weight).
FBETA = 0.5


@dataclass
class MethodTrial:
    """One method instance proposed for a trial: how to cache its embeddings,
    how to build it, and the hyperparams to log."""

    cache_key: tuple  # identifies the embedding output; equal keys reuse the cache
    build: Callable[[], EmbeddingMethod]
    params: dict[str, object]  # method-specific hyperparams, for W&B + logging


def _suggest_deepface(trial: optuna.Trial) -> MethodTrial:
    model_name = trial.suggest_categorical("model_name", MODEL_NAMES)
    detector_backend = trial.suggest_categorical("detector_backend", DETECTOR_BACKENDS)
    return MethodTrial(
        cache_key=("deepface", model_name, detector_backend),
        build=lambda: DeepFaceMethod(model_name, detector_backend, DEEPFACE_MODELS_DIR),
        params={"model_name": model_name, "detector_backend": detector_backend},
    )


def _suggest_insightface(trial: optuna.Trial) -> MethodTrial:
    model_pack = trial.suggest_categorical("model_pack", MODEL_PACKS)
    det_size = trial.suggest_categorical("det_size", DET_SIZES)
    # Padding and det_thresh both change which faces detect (and thus the embedding
    # output), so both are part of the cache key. Stepped (not continuous) so
    # revisited values still hit the embedding cache.
    pad_ratio = trial.suggest_float("pad_ratio", 0.0, 1.0, step=0.1)
    # Detector confidence floor. Searched below the 0.5 default to recover the
    # weak/blurry/partial faces behind most "no face detected" failures; the
    # objective now charges recall for those failures, so the search trades
    # recovered coverage against any spurious detections lower thresholds admit.
    det_thresh = trial.suggest_float("det_thresh", 0.1, 0.6, step=0.05)
    return MethodTrial(
        cache_key=("insightface", model_pack, det_size, pad_ratio, det_thresh),
        build=lambda: InsightFaceMethod(
            model_pack=model_pack,
            models_dir=INSIGHTFACE_MODELS_DIR,
            det_size=(det_size, det_size),
            pad_ratio=pad_ratio,
            det_thresh=det_thresh,
        ),
        params={
            "model_pack": model_pack,
            "det_size": det_size,
            "pad_ratio": pad_ratio,
            "det_thresh": det_thresh,
        },
    )


_SUGGEST_METHOD: dict[str, Callable[[optuna.Trial], MethodTrial]] = {
    "deepface": _suggest_deepface,
    "insightface": _suggest_insightface,
}


def _fbeta(precision: float, recall: float, beta: float) -> float:
    """Micro-averaged Fbeta. Returns 0.0 if either component is 0 (a threshold
    so tight nothing is labelled -> recall 0, or so loose everything is
    mislabelled -> precision 0), which is what keeps the optimiser honest."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


_THRESHOLD_RANGES: dict[str, tuple[float, float]] = {
    "cosine": (0.1, 0.9),
    "euclidean_l2": (0.3, 1.5),
    "euclidean": (5.0, 30.0),
}


def _suggest_threshold(trial: optuna.Trial, distance_metric: str) -> float:
    # Each metric has its own natural scale, so the threshold range is keyed
    # off the metric rather than searched as one shared range (same trick as
    # canny_search's threshold_low/threshold_delta branching).
    if distance_metric == "cosine":
        return trial.suggest_float("threshold_cosine", 0.1, 0.9)
    if distance_metric == "euclidean_l2":
        return trial.suggest_float("threshold_euclidean_l2", 0.3, 1.5)
    return trial.suggest_float("threshold_euclidean", 5.0, 30.0)


def _normalize_threshold(threshold: float, distance_metric: str) -> float:
    lo, hi = _THRESHOLD_RANGES[distance_metric]
    return (threshold - lo) / (hi - lo)


def make_objective(
    matches: list[MatchedFace],
    method: str,
    wandb_project: str,
    run_group: str,
) -> Callable[[optuna.Trial], float]:
    suggest_method = _SUGGEST_METHOD[method]

    # Embedding is the slow part (this is what got slow when detector_backend
    # was turned on); distance_metric/threshold are nearly free to vary once
    # embeddings exist. Cache per distinct embedding-defining combo
    # (MethodTrial.cache_key) so each combo is only ever embedded once across the
    # whole study, no matter how many trials revisit it.
    embedding_cache: dict[tuple, dict[str, np.ndarray | None]] = {}

    def _get_embeddings(method_trial: MethodTrial) -> dict[str, np.ndarray | None]:
        key = method_trial.cache_key
        if key not in embedding_cache:
            logger.info(
                f"Embedding {len(matches)} GT face(s) with {method_trial.params} "
                "(cached for every future trial with this combo)..."
            )
            embedder = method_trial.build()
            embeddings: dict[str, np.ndarray | None] = {}
            for idx, m in enumerate(matches, start=1):
                start = time.perf_counter()
                result = embedder.embed(m.crop_path)
                embeddings[str(m.crop_path)] = result.embedding if result is not None else None
                elapsed = time.perf_counter() - start
                logger.info(
                    f"[{idx}/{len(matches)}] embedded {m.crop_path} ({elapsed:.1f}s)"
                )
            embedding_cache[key] = embeddings
        return embedding_cache[key]

    def objective(trial: optuna.Trial) -> float:
        method_trial = suggest_method(trial)
        distance_metric = trial.suggest_categorical("distance_metric", DISTANCE_METRICS)
        threshold = _suggest_threshold(trial, distance_metric)

        embeddings = _get_embeddings(method_trial)
        result = evaluate_loo(matches, embeddings, distance_metric, threshold)
        fbeta = _fbeta(result.precision, result.recall, FBETA)

        trial.set_user_attr("precision", result.precision)
        trial.set_user_attr("recall", result.recall)
        trial.set_user_attr("specificity", result.specificity)
        trial.set_user_attr("n_known_queries", result.n_known_queries)
        trial.set_user_attr("n_negative_queries", result.n_negative_queries)
        trial.set_user_attr("n_emitted_labels", result.n_emitted_labels)

        run = wandb.init(
            project=wandb_project,
            group=run_group,
            name=f"face-recognition-trial-{trial.number:04d}",
            config={
                **method_trial.params,
                "distance_metric": distance_metric,
                "threshold": threshold,
                "threshold_normalized": _normalize_threshold(threshold, distance_metric),
            },
            reinit="create_new",
        )
        run.log(
            {
                "fbeta": fbeta,
                "precision": result.precision,
                "recall": result.recall,
                "specificity": result.specificity,
                "n_known_queries": result.n_known_queries,
                "n_negative_queries": result.n_negative_queries,
                "n_emitted_labels": result.n_emitted_labels,
            }
        )
        run.finish()

        return fbeta

    return objective
