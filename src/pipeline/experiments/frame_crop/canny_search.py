from collections.abc import Callable

import numpy as np
import optuna
import wandb

from pipeline.experiments.frame_crop.metrics import GTItem, detection_rate_at_iou, polygon_iou
from pipeline.frame_crop.config import AnnotationConfig, BlurConfig, MorphologyConfig
from pipeline.frame_crop.methods.canny import CannyConfig, CannyMethod

_ANNOTATION = AnnotationConfig(scale=1.0, bold=False)


def _suggest_config(trial: optuna.Trial) -> CannyConfig:
    threshold_low = trial.suggest_float("threshold_low", 10.0, 200.0)
    threshold_high = threshold_low + trial.suggest_float("threshold_delta", 10.0, 300.0)
    return CannyConfig(
        threshold1=threshold_low,
        threshold2=threshold_high,
        save_output=False,
        annotate=False,
        save_contours=False,
        blur=BlurConfig(
            enabled=True,
            save_output=False,
            annotate=False,
            kernel_size=trial.suggest_categorical("blur_kernel_size", [3, 5, 7, 9, 11]),
        ),
        morphology=MorphologyConfig(
            enabled=True,
            save_output=False,
            annotate=False,
            save_diff=False,
            kernel_size=trial.suggest_categorical("morph_kernel_size", [3, 5, 7]),
            iterations=trial.suggest_int("morph_iterations", 1, 15),
        ),
    )


def make_objective(
    gt_items: list[GTItem],
    images: dict[str, np.ndarray],
    min_area_fraction: float,
    max_area_fraction: float,
    wandb_project: str,
    run_group: str,
    iou_threshold: float = 0.75,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        config = _suggest_config(trial)
        method = CannyMethod(config, min_area_fraction, max_area_fraction, _ANNOTATION)

        ious = []
        for item in gt_items:
            detection = method.detect(images[item.image_rel])
            iou = polygon_iou(detection.quad, item.quad) if detection.quad is not None else 0.0
            ious.append(iou)

        detection_rate = detection_rate_at_iou(ious, iou_threshold)
        mean_iou = float(np.mean(ious))
        trial.set_user_attr("mean_iou", mean_iou)

        wandb_config = {**trial.params, "threshold_high": config.threshold2}
        wandb_config.pop("threshold_delta")
        run = wandb.init(
            project=wandb_project,
            group=run_group,
            name=f"canny-trial-{trial.number:04d}",
            config=wandb_config,
            reinit="create_new",
        )
        run.log({"detection_rate": detection_rate, "mean_iou": mean_iou})
        run.finish()

        return detection_rate

    return objective
