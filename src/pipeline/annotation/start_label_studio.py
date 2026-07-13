"""Start Label Studio with local file serving enabled."""

from __future__ import annotations

import os
import subprocess

from dotenv import load_dotenv

from pipeline.shared.paths import DATA_DIR


def start_label_studio(args: list[str]) -> int:
    load_dotenv()
    os.environ.setdefault("LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED", "true")
    os.environ.setdefault("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", str(DATA_DIR))
    completed = subprocess.run(["label-studio", *args], env=os.environ)
    return completed.returncode
