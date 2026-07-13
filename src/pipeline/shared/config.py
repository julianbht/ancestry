"""Generic config loading utility shared across all pipeline steps.

Each step defines its own Pydantic config schema in its own folder.
This module provides the load() function to validate a step's YAML config,
and _validate() as a shared helper.

Usage:
    from pipeline.shared.config import load
    from pipeline.download.config import DownloadConfig

    config = load("download", DownloadConfig)
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from pipeline.shared.paths import CONFIG_DIR

# .env is loaded in pipeline.shared.paths (imported above), which runs before any
# config or env var is read. That single load point lets a local .env set ENV=dev
# to default away from the prod configs without affecting containers, which never
# ship a .env (see deployment-env skill).


class StrictConfig(BaseModel):
    """Base class for every config schema in the project.

    Two rules, both enforced at load time:
      - No defaults: every field a schema declares must be present in the YAML.
        Schemas therefore declare fields without default values; a missing field
        is a validation error, not a silent fallback.
      - No extras: unknown keys are rejected (``extra="forbid"``), so a typo'd or
        stale key crashes instead of being ignored.

    The result is that each config file fully and exactly specifies its config —
    nothing is hidden behind a default, and what you read is what runs.
    """

    model_config = ConfigDict(extra="forbid")


def load[T: BaseModel](step: str, model: type[T]) -> T:
    """Load and validate a step's config/<step>/step.<ENV>.yaml.

    ENV defaults to "prod" if not set.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If validation fails, with a human-readable message.
    """
    env = os.environ.get("ENV", "prod")
    config_file = CONFIG_DIR / step / f"step.{env}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    raw = yaml.safe_load(config_file.read_text()) or {}
    return _validate(raw, model, config_file)


def load_file[T: BaseModel](config_file: Path, model: type[T]) -> T:
    """Load and validate an explicit YAML config file.

    Args:
        config_file: Full path to the YAML config file.
        model: Pydantic model class to validate against.

    Returns:
        A validated instance of the model.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If validation fails, with a human-readable message.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    raw = yaml.safe_load(config_file.read_text()) or {}
    return _validate(raw, model, config_file)


def _validate[T: BaseModel](raw: dict, model: type[T], source) -> T:
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        lines = [f"Invalid config in {source}:"]
        for err in e.errors():
            field = " -> ".join(str(x) for x in err["loc"])
            lines.append(f"  {field}: {err['msg']}")
        raise ValueError("\n".join(lines)) from e
