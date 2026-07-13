# Annotation

## Start Label Studio

```bash
uv run ancestry-label-studio start
```

This sets up local file serving from `data/` automatically.

## Frame crop annotation

```bash
# Build tasks
uv run ancestry-label-studio build

# After exporting from Label Studio:
uv run ancestry-label-studio normalize --export <path-to-export.json>
```

Ground truth is written to `data/curated/frame_crop/ground_truth.json`.

## Face annotation

```bash
# Generate Label Studio XML template (re-run if family tree changes)
uv run ancestry-face-annotation generate-template

# Build tasks. Images already in ground_truth.json are skipped, so this only
# proposes not-yet-annotated photos. Each task is pre-drawn with face_crop's
# detected boxes (Label Studio predictions) on the full source image — you
# just assign person/age/portrait and fix the rare bad box.
uv run ancestry-face-annotation build

# Build a random, reproducible batch of 100 not-yet-annotated images:
uv run ancestry-face-annotation build --sample 100 --seed 42

# Force-include specific keys even if already annotated (e.g. to redo a mistake):
uv run ancestry-face-annotation build --sample 100 --seed 42 \
    --include EjBRbsZ4L3kT36j/20260428_172241.jpg

# After exporting from Label Studio, merge it into the ground truth:
uv run ancestry-face-annotation normalize --export <path-to-export.json>
```

Ground truth is written to `data/curated/face_annotation/ground_truth.json`.
Each annotated photo contains a list of faces with: person ID, age, and portrait flag.

`normalize` **merges** into the existing ground truth (keyed by raw-relative
path): new photos are added and re-annotated photos overwrite their old entry,
so curating in several rounds is safe and nothing is lost. To redo a photo,
include it in a new batch (`--include`) and re-annotate it; the corrected entry
replaces the old one on normalize.
