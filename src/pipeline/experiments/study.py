import optuna
import wandb
from loguru import logger


def create_study(study_name: str, direction: str) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(direction=direction, study_name=study_name)


def log_best(study: optuna.Study, value_name: str = "value") -> None:
    best = study.best_trial
    attrs = ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in best.user_attrs.items()
    )
    attrs_part = f"{attrs}, " if attrs else ""
    logger.info(
        f"Best trial #{best.number}: {value_name}={best.value:.4f}, "
        f"{attrs_part}params={best.params}"
    )
