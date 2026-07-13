import time
from collections.abc import Callable

import numpy as np
import optuna
import wandb
from loguru import logger

from pipeline.experiments.frame_crop.metrics import GTItem, detection_rate_at_iou, polygon_iou
from pipeline.frame_crop.methods.sam import SamMethod

SAM_PROMPTS = [
    "photo print",
    "old photograph",
    "rectangular photograph",
    "photograph",
    "photograph with people",
    "printed photograph",
    "paper photo",
    "vintage photograph",
    "scanned photo print",
    "image",
    "image with people",
    "rectangular image",
    "rectangular with people",
]


def make_objective(
    gt_items: list[GTItem],
    images: dict[str, np.ndarray],
    sam_method: SamMethod,
    wandb_project: str,
    run_group: str,
    prompts: list[str] = SAM_PROMPTS,
    iou_threshold: float = 0.75,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        prompt = trial.suggest_categorical("prompt", prompts)
        score_threshold = trial.suggest_float("score_threshold", 0.2, 0.8)

        # Swap prompt/threshold without reloading the model
        sam_method._config = sam_method._config.model_copy(
            update={"prompt": prompt, "score_threshold": score_threshold}
        )

        ious = []
        t0 = time.perf_counter()
        for item in gt_items:
            detection = sam_method.detect(images[item.image_rel])
            iou = (
                polygon_iou(detection.quad, item.quad)
                if detection.quad is not None
                else 0.0
            )
            ious.append(iou)
        elapsed = time.perf_counter() - t0
        logger.info(f"Trial {trial.number}: {len(ious)} images in {elapsed:.1f}s ({elapsed / len(ious):.2f}s/image)")

        detection_rate = detection_rate_at_iou(ious, iou_threshold)
        mean_iou = float(np.mean(ious))
        trial.set_user_attr("mean_iou", mean_iou)

        run = wandb.init(
            project=wandb_project,
            group=run_group,
            name=f"sam-trial-{trial.number:04d}",
            config=trial.params,
            reinit="create_new",
        )
        run.log({"detection_rate": detection_rate, "mean_iou": mean_iou, "prompt": prompt, "score_threshold": score_threshold})
        run.finish()

        return detection_rate

    return objective
