import sys
from datetime import datetime
from pathlib import Path

import click
import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from pipeline.experiments.frame_crop.metrics import GTItem, load_ground_truth
from pipeline.experiments.study import create_study, log_best
from pipeline.shared.paths import CURATED_DIR, DATA_DIR, STEPS_DIR

RAW_DIR = DATA_DIR / "raw"
ROTATED_DIR = STEPS_DIR / "rotate"
GT_PATH = CURATED_DIR / "frame_crop" / "ground_truth.json"
MIN_AREA_FRACTION = 0.15
MAX_AREA_FRACTION = 0.95


def _resolve_input(image_rel: str) -> Path:
    rel = Path(image_rel).relative_to("raw")
    rotated = ROTATED_DIR / rel
    return rotated if rotated.exists() else RAW_DIR / rel


def _load_images(gt_items: list[GTItem]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for item in gt_items:
        path = _resolve_input(item.image_rel)
        img = cv2.imread(str(path))
        if img is None:
            logger.error(f"Could not read image: {path}")
            sys.exit(1)
        images[item.image_rel] = img
    return images


@click.command()
@click.option("--method", type=click.Choice(["canny", "sam"]), required=True)
@click.option("--n-trials", default=100, show_default=True)
@click.option("--wandb-project", default="ancestry-frame-crop-hpopt", show_default=True)
@click.option(
    "--run-group", default=None, help="W&B group name. Defaults to a timestamp."
)
@click.option("--iou-threshold", default=0.75, show_default=True)
def main(
    method: str,
    n_trials: int,
    wandb_project: str,
    run_group: str | None,
    iou_threshold: float,
) -> None:
    if run_group is None:
        run_group = datetime.now().strftime("%Y%m%d-%H%M%S")

    gt_items = load_ground_truth(GT_PATH)
    logger.info(f"Loaded {len(gt_items)} GT items, loading images...")
    images = _load_images(gt_items)

    logger.info(f"W&B project={wandb_project!r}, group={run_group!r}")
    study = create_study(f"frame-crop-{method}", "maximize")

    if method == "canny":
        from pipeline.experiments.frame_crop.canny_search import make_objective

        objective = make_objective(
            gt_items,
            images,
            MIN_AREA_FRACTION,
            MAX_AREA_FRACTION,
            wandb_project,
            run_group,
            iou_threshold,
        )
    else:
        from pipeline.experiments.frame_crop.sam_search import (
            SAM_PROMPTS,
            make_objective,
        )
        from pipeline.frame_crop.methods.sam import SamConfig, SamMethod

        sam_config = SamConfig(
            prompt=SAM_PROMPTS[0], score_threshold=0.5, device="cuda"
        )
        logger.info("Loading SAM model (once for all trials)...")
        sam_method = SamMethod(sam_config, MIN_AREA_FRACTION, MAX_AREA_FRACTION)
        objective = make_objective(
            gt_items,
            images,
            sam_method,
            wandb_project,
            run_group,
            iou_threshold=iou_threshold,
        )

    logger.info(f"Starting Optuna study: method={method}, n_trials={n_trials}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log_best(study, "detection_rate")
