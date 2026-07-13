"""Canonical paths for the project, derived from this file's location.

All pipeline modules should import paths from here rather than constructing
them with relative Path() calls, so the code works regardless of the working
directory from which it is invoked.

DATA_DIR is redirectable via the ANCESTRY_DATA_DIR env var so the pipeline can
run against an alternate data tree (e.g. the committed quickstart/ dummy set)
without any code change. Everything else is derived from the repo location.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Loaded here — the most foundational module, imported before any path or env
# var is read — so DATA_DIR below can be driven from .env and every step picks
# up secrets the same way regardless of entry point. load_dotenv() is idempotent
# and does not override variables already set in the real environment.
load_dotenv()

# This file lives at src/pipeline/shared/paths.py, so the repo root is four
# levels up (shared → pipeline → src → repo root).
PROJECT_ROOT = Path(__file__).parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"

# Real data lives in data/ under the repo. ANCESTRY_DATA_DIR overrides the root —
# now the single switch for ALL private data (raw, curated, derived). A relative
# override anchors at the repo root (not the cwd, so it's launch-independent); an
# absolute one replaces it outright — pathlib's `/` drops the left operand when
# the right is absolute, so no is_absolute() branch is needed. expanduser() covers
# a leading ~ (e.g. from .env, which python-dotenv does not expand).
_data_override = os.environ.get("ANCESTRY_DATA_DIR", "").strip() or "data"
DATA_DIR = PROJECT_ROOT / Path(_data_override).expanduser()

STEPS_DIR = DATA_DIR / "steps"  # step-processed images (rotated, frames, faces, …)
DEBUG_DIR = DATA_DIR / "debug"  # throwaway debug output — safe to delete

# Hand-made, non-regenerable inputs: ground-truth labels, rotation + photo-back
# CSVs, download config. Lives under the data root so a single ANCESTRY_DATA_DIR
# switch redirects it too (e.g. to the committed quickstart/ dummy tree).
CURATED_DIR = DATA_DIR / "curated"

# Gramps genealogy (database/, portraits/, documents/, graphs/) — a curated
# input, but kept separate from the label-annotation artifacts under curated/.
GRAMPS_DIR = DATA_DIR / "gramps"

# Label Studio working files: generated annotation tasks and raw project exports.
# Regenerable scaffolding — the distilled labels live in curated/.
LABEL_STUDIO_DIR = DATA_DIR / "label_studio"

# Model checkpoints (SAM 3, InsightFace, DeepFace) are large shared binaries, not
# data — so MODELS_DIR is anchored at the real data/models and deliberately does
# NOT follow ANCESTRY_DATA_DIR. Every data root (including a redirected quickstart/)
# reads the one shared install; in prod PROJECT_ROOT is /app, so this stays the
# PVC's /app/data/models where scripts/setup_sam3.py installs them.
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def rel(path: Path) -> Path:
    """Return path relative to PROJECT_ROOT for log messages (VSCode-clickable).

    Falls back to the path unchanged when it lies outside the repo — possible
    now that DATA_DIR may be redirected to an external location.
    """
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path
