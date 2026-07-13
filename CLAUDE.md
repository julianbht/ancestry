# Ancestry Photo Pipeline

## Project Goal

A pipeline that processes old family photos to perform face recognition and age estimation. Photos are scans-of-prints photographed by smartphone.

## Pipeline Overview

```
[Nextcloud] → download → rotate → frame_crop → face_crop → face_recognition → age_estimation
```

Each step is **idempotent**: re-running a step only processes files that have not been processed yet. This is critical because the photo collection grows incrementally and steps must be rerunnable for newly added photos without redoing completed work.

### Steps

1. **download** — Fetch photos from university Nextcloud via WebDAV. Track which remote files have been downloaded.
2. **rotate** — Fix photos with incorrect orientation. Driven by `data/curated/rotate/rotations.csv` (filename → apparent_rotation). Use `scripts/detect_rotated.py` to auto-populate the CSV from EXIF tags (phone-specific heuristic); manual entries can be added for photos from other devices.
3. **frame_crop** — Detect and crop the physical photo print out of the smartphone image.
4. **face_crop** — Detect all faces in each cropped frame and save each face as its own image.
5. **face_recognition** — Identify each face crop: match against known labeled examples (a few reference photos per person at different ages), or mark as unknown.
6. **age_estimation** — Estimate the age of each detected face.

## Directory Structure

```
config/                 # committed, PUBLIC — pipeline configuration only
  <step>/               # one folder per step
    step.yaml           # step config (validated by Pydantic on load)
    ...                 # additional public step config (label templates, cases…)
src/                    # Python source (src layout — installed as packages)
  pipeline/             # the batch pipeline
    shared/             # utilities shared across all steps (paths, state, log, config loader)
    <step>/             # one folder per step
      __init__.py       # empty
      step.py           # contains the step's run() function
      config.py         # step-specific Pydantic config schemas (omit if step has no config)
  web/                  # FastAPI viewer over the pipeline outputs (imports pipeline.shared)
scripts/                # one-off helper scripts (not part of the pipeline)
quickstart/             # committed, PUBLIC — dummy data tree (see docs/going-public.md)
  data/                 # a stand-in data root; run on it via ANCESTRY_DATA_DIR=quickstart/data
data/                   # gitignored — ALL private data; redirectable via ANCESTRY_DATA_DIR
  raw/                  # originals (downloaded or dropped in locally)
  gramps/               # Gramps genealogy: database/, portraits/, documents/, graphs/ (backed up to R2)
  curated/              # hand-made, non-regenerable annotation inputs (backed up to R2)
    face_annotation/ground_truth.json   # face labels + ages + is_portrait flag
    frame_crop/ground_truth.json         # print-quad labels
    rotate/rotations.csv                 # filename -> apparent_rotation
    photo_backs.csv                      # front photo -> back-note photo (backs excluded from face_crop)
    download/shares.json                 # Nextcloud share config
  steps/                # step-processed images (one subfolder per step, named after the step)
    rotate/             # rotation-corrected versions
    frame_crop/         # frame-cropped versions of raw photos
    face_crop/          # individual face crops, one file per face
  state/                # JSON state files tracking pipeline progress
  logs/                 # one log file per pipeline run
  label_studio/         # Label Studio working files: generated tasks + raw project exports (regenerable)
  debug/                # throwaway debug output (previews, intermediate images) — safe to delete
```

**Data root**: everything private lives under `data/`. The `ANCESTRY_DATA_DIR`
env var redirects that root (absolute path used as-is; relative anchored at the
repo root), so `ANCESTRY_DATA_DIR=quickstart/data` runs the whole project on the
committed dummy tree. `config/` is public and always at the repo root. Model
checkpoints (`data/models/`, via `MODELS_DIR`) are the one exception: large shared
binaries anchored at the real `data/models` that deliberately do *not* follow the
redirect, so every data root (including quickstart) uses one install.

## State Tracking

Each pipeline step maintains a JSON state file in `data/state/`. The general schema is:

```json
{
  "version": 1,
  "processed": {
    "<input_file_identifier>": {
      "status": "done" | "failed" | "skipped",
      "output": ["<output_file_path>", ...],
      "timestamp": "<ISO8601>",
      "error": null | "<error message>"
    }
  }
}
```

A step reads its state file on startup and skips any input already marked `"done"`. On success it writes the entry; on failure it writes the error and continues to the next file. This allows partial runs and crash recovery.

~2000 files total — JSON state files are fine at this scale.

## Coordinate Spaces

Several steps detect rectangles within an image that is itself a crop of an earlier step's output (`frame_crop` crops the source photo; `face_crop` detects faces within that crop). Each step's sidecar JSON expresses its detections in two coordinate systems: its own **local** pixels (the image it actually ran on), and **source-image** pixels — the same coordinate system `ground_truth.json`'s `bbox_xywh` annotations use (drawn on `data/raw/` or `data/steps/rotate/`, whichever was the actual input to `frame_crop`).

The convention: each step persists the resolved offset of its own crop within its input image (`frame_crop`'s `crop_rect_xywh`), and the next step adds that one offset to its own local detections to produce the source-image-space box (`face_crop`'s `box_xyxy_source`). No step ever needs to walk back further than its immediate predecessor — every sidecar is already fully resolved into source-image space, so coordinates compose by simple addition rather than by replaying the pipeline's geometry from scratch. This only works because `frame_crop` strictly crops (never resizes or warps) — a fixed pixel scale means a translation is enough, no scale factor needed. (This is why the perspective-warp crop mode was removed: keeping crops to axis-aligned translations is what keeps this scheme simple.)

Payoff: a ground-truth-labeled face can be matched to a `face_crop` output crop with a plain IoU check on two boxes already in the same coordinate space — no transform code needed at use time.

## Face Recognition Approach

- **Known people**: provide a small set of reference face crops per person (labeled, potentially across different ages).
- **Unknown people**: faces that do not match any known person above a confidence threshold are flagged as `unknown`.
- Recognition results feed into Label Studio for human review and correction.

## Running the Pipeline

The project uses `uv` for dependency management and running scripts.

Run the full pipeline (all steps in order):

```bash
uv run python -m pipeline
```

Or run a single step individually:

```bash
uv run ancestry-download
uv run ancestry-rotate
```

Re-running any step or the full pipeline is always safe — already-processed files are skipped.

## Configuration

Each step has its own config folder at `config/<step>/`. The main config file is always `step.yaml`, validated against a Pydantic schema in `src/pipeline/<step>/config.py`. Steps may have additional config files (e.g. `shares.json` for the download step).

- **No hidden defaults**: config schemas extend `StrictConfig` (`src/pipeline/shared/config.py`), which forbids extra keys and declares every field *without* a default. Each YAML must therefore spell out every value — a missing field or a typo'd key fails validation and crashes the step with a clear message, rather than silently falling back. What you read in the YAML is exactly what runs. The one allowed exception is a derived filesystem path (e.g. SAM's `checkpoint_path`), which may keep a computed default rather than being hardcoded into YAML.
- `.env` — **gitignored** — contains secrets (currently only `NEXTCLOUD_PASSWORD`). Must be created manually on each machine after `git clone`.

## Step Conventions

Every step follows the same structural pattern:

- **State key**: relative path from the step's input directory (e.g. `data/raw/`), so it stays portable across machines.
- **Idempotency**: check `state_lib.is_done(state, key)` before processing; skip if already done.
- **Config**: load via `load_config("<step>", ConfigClass)` — always include `max_files_to_<step>` for smoke-testing and `ignore_state` for re-running without clearing state.
- **Per-file state save**: write state after every file (not at the end), so a crash is recoverable.
- **Run summary**: log a final `Done — ...` line that a reader can understand without knowing the code. Guidelines, not a rigid format:
  - Use outcome labels that are self-explanatory. Avoid internal jargon like "fallback" — say what actually happened (e.g. "no frame detected — full image saved").
  - Distinguish *processed-this-run* outcomes from *skipped* (already done) and *failed* (errored). Annotate counts where the meaning isn't obvious (e.g. `0 skipped (already done)`).
  - For steps with a quality metric (e.g. detection success rate), show the rate as a percentage of attempted files — not of skipped — and lead with the attempted count so the percentages have an obvious denominator.
  - Err on the side of clarity over terseness; a slightly longer line that's unambiguous beats a compact one that needs the source to decode.

  Example with a quality metric:
  ```
  Done — 50 attempted: 24 cropped (48%), 26 no frame detected — full image saved (52%); 0 skipped (already done), 0 failed (errors)
  ```
  Simple pass/fail step:
  ```
  Done — 45 extracted, 200 skipped (already done), 0 failed
  ```

## Code Style

- **Pylance warnings**: fix them when straightforward (e.g. adding type annotations). Leave them if fixing requires awkward workarounds or if a third-party library simply doesn't provide stubs.
- **Separation of concerns**: keep I/O, business logic, and state management distinct. Steps should not reach into each other's responsibilities.
- **Software patterns**: make use of common software patterns where they improve the readability of the code without overcomplicating it.