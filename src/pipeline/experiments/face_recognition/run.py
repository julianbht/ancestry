"""Hyperparameter search for face_recognition, scored by leave-one-out Fbeta
(micro-averaged precision/recall) against the face_annotation ground truth
(see metrics.py for why LOO and search.py for the Fbeta weighting).

One study per embedding method (--method), since DeepFace and InsightFace have
different hyperparameter spaces; the distance_metric and threshold are searched
for both. See search.py for each method's search space.

Usage:
    uv run ancestry-face-recognition-hpopt --method deepface --n-trials 50
    uv run ancestry-face-recognition-hpopt --method insightface --n-trials 50
    uv run ancestry-face-recognition-hpopt --n-trials 2 --wandb-project ancestry-face-recognition-hpopt-dev
"""

from datetime import datetime
from pathlib import Path

import click
import wandb
from codecarbon import EmissionsTracker
from dotenv import load_dotenv
from loguru import logger

from pipeline.experiments.face_recognition.search import make_objective
from pipeline.experiments.study import create_study, log_best
from pipeline.face_recognition.gallery import match_ground_truth
from pipeline.shared.log import setup
from pipeline.shared.paths import CURATED_DIR, STEPS_DIR

load_dotenv()

logger.info("Imports done")

GT_PATH = CURATED_DIR / "face_annotation" / "ground_truth.json"
FACES_DIR = STEPS_DIR / "face_crop"
RESULTS_DIR = Path(__file__).parent / "results"
# Fixed rather than searched — this controls GT-to-detection matching, not the
# recognizer itself, and the current production value is already generous.
OVERLAP_THRESHOLD = 0.30


@click.command()
@click.option(
    "--method",
    type=click.Choice(["deepface", "insightface"]),
    default="deepface",
    show_default=True,
    help="Embedding method to tune. Run once per method.",
)
@click.option("--n-trials", default=50, show_default=True)
@click.option(
    "--wandb-project", default="ancestry-face-recognition-hpopt", show_default=True
)
@click.option(
    "--run-group", default=None, help="W&B group name. Defaults to a timestamp."
)
def main(method: str, n_trials: int, wandb_project: str, run_group: str | None) -> None:
    setup(f"face-recognition-hpopt-{method}")

    if run_group is None:
        run_group = f"{method}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    logger.info("Loading ground truth and matching face crops...")
    matches = match_ground_truth(GT_PATH, FACES_DIR, OVERLAP_THRESHOLD)
    n_labeled = sum(1 for m in matches if m.person_id)
    logger.info(
        f"Loaded {len(matches)} GT face(s) matched to face_crop output "
        f"({n_labeled} labeled, {len(matches) - n_labeled} unidentified)"
    )

    logger.info(f"Method={method!r}, W&B project={wandb_project!r}, group={run_group!r}")
    study = create_study(f"face-recognition-{method}", "maximize")

    objective = make_objective(matches, method, wandb_project, run_group)

    logger.info(f"Starting Optuna study: n_trials={n_trials}")
    # Measure the whole-study energy/CO2, not per trial: embeddings are cached
    # across trials (search.py), so a single trial's cost depends on cache hits,
    # not its hyperparameters — only the study total is meaningful.
    logger.info("Starting CodeCarbon emissions tracker...")
    tracker = EmissionsTracker(
        project_name=f"face-recognition-hpopt-{method}",
        output_dir=str(RESULTS_DIR),
        output_file="emissions.csv",
        log_level="error",
        save_to_file=True,
        allow_multiple_runs=True,
    )
    logger.info("Tracker started, launching Optuna study...")
    tracker.start()
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    finally:
        tracker.stop()

    _report_emissions(tracker, method, n_trials, wandb_project, run_group)
    log_best(study, "fbeta")


def _report_emissions(
    tracker: EmissionsTracker,
    method: str,
    n_trials: int,
    wandb_project: str,
    run_group: str,
) -> None:
    """Log the study's total energy/CO2 to loguru and a dedicated W&B run.
    CodeCarbon already wrote the raw breakdown to results/emissions.csv."""
    data = tracker.final_emissions_data
    if data is None:
        logger.warning("CodeCarbon produced no emissions data; skipping report.")
        return

    metrics = {
        "duration_s": data.duration,
        "energy_consumed_kwh": data.energy_consumed,
        "cpu_energy_kwh": data.cpu_energy,
        "gpu_energy_kwh": data.gpu_energy,
        "ram_energy_kwh": data.ram_energy,
        "emissions_kg": data.emissions,
        "emissions_rate_kg_per_s": data.emissions_rate,
    }
    logger.info(
        f"Done — energy {data.energy_consumed:.4f} kWh "
        f"(cpu {data.cpu_energy:.4f}, gpu {data.gpu_energy:.4f}, "
        f"ram {data.ram_energy:.4f}), {data.emissions:.4f} kg CO2eq "
        f"over {data.duration:.0f}s; breakdown in {RESULTS_DIR / 'emissions.csv'}"
    )

    run = wandb.init(
        project=wandb_project,
        group=run_group,
        name=f"emissions-{run_group}",
        job_type="emissions",
        config={"method": method, "n_trials": n_trials},
        reinit="create_new",
    )
    run.log(metrics)
    run.finish()


if __name__ == "__main__":
    main()
