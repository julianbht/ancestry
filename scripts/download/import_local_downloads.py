"""
One-time script: import locally available zip files as if they were downloaded by the pipeline.

For each entry in MAPPING below, copies the local file to data/raw/<token>/<filename>
and marks it as done in data/state/download.json.

Usage:
    uv run python scripts/import_local_downloads.py
"""

import shutil

from pipeline.shared import state as state_lib
from pipeline.shared.paths import DATA_DIR, DEBUG_DIR

STATE_FILE = state_lib.STATE_DIR / "download.json"
RAW_DIR = DATA_DIR / "raw"
TMP_DIR = DEBUG_DIR

MAPPING = [
    ("9JRaB5HNcYfZYMa", "Batch 1", TMP_DIR / "family-part-001.zip"),
    ("zemaAxcRyw3NqXb", "Batch 2", TMP_DIR / "family-part-002.zip"),
    ("CdSg23WdQcjMDrZ", "Batch 3", TMP_DIR / "family-part-003.zip"),
    ("FCXZ97GMYoZfET9", "Batch 4", TMP_DIR / "family-part-004.zip"),
    ("3fBXinPqztWt3CP", "Batch 5", TMP_DIR / "family-part-005.zip"),
    ("KGQ6aDsd2aaKoMX", "Batch 6", TMP_DIR / "family-part-006.zip"),
    ("PKRGk5ArAJnotHc", "Batch 7", TMP_DIR / "family-part-007.zip"),
    ("8a6stfqDGdg3MHX", "Batch 8", TMP_DIR / "family-part-008.zip"),
    ("EjBRbsZ4L3kT36j", "Batch 9", TMP_DIR / "family-part-009.zip"),
    ("GRNn5dqntxTo6Am", "Batch 10", TMP_DIR / "family-part-010.zip"),
    ("XfBqCe7qtcbfMLb", "Batch 11", TMP_DIR / "family-part-011.zip"),
    ("6FTSxPHGgTHarEB", "Batch 12", TMP_DIR / "family-part-012.zip"),
    ("fQizjwKj2MaqQQC", "Batch 13", TMP_DIR / "family-part-013.zip"),
    ("EcGMKXPB8XXMrSj", "Batch 14", TMP_DIR / "family-part-014.zip"),
    ("QBiiSmFjkyomJ82", "Batch 15", TMP_DIR / "family-part-015.zip"),
    ("6q3zseyDCsTpmcd", "Batch 16", TMP_DIR / "family-part-016.zip"),
    ("4G9qGpi2oxwkPpq", "Batch 17", TMP_DIR / "family-part-017.zip"),
]


def main() -> None:
    state = state_lib.load(STATE_FILE)

    for token, description, local_path in MAPPING:
        if local_path is None:
            print(f"[SKIP] {description} ({token}) — no local_path set")
            continue

        src = local_path
        if not src.exists():
            print(f"[ERROR] {description} ({token}) — file not found: {src}")
            continue

        filename = src.name
        key = f"{token}/{filename}"

        if state_lib.is_done(state, key):
            print(f"[SKIP] {description} ({token}) — already in state")
            continue

        dest = RAW_DIR / token / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[COPY] {description} ({token}) — {src} -> {dest}")
        shutil.copy2(src, dest)

        state_lib.mark_done(state, key, [str(dest)])
        state_lib.save(state, STATE_FILE)
        print(f"[DONE] {description} ({token})")

    print("Import complete.")


if __name__ == "__main__":
    main()
