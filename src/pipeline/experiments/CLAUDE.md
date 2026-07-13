# Experiments

Hyperparameter search for pipeline steps using Optuna (search) + W&B (logging).

## Structure

```
pipeline/experiments/
  study.py                  # shared: create_study(), log_best()
  frame_crop/
    run.py                  # CLI entry point
    metrics.py              # GTItem, load_ground_truth(), polygon_iou(), accuracy_at_iou()
    canny_search.py         # Optuna objective for the Canny method
    sam_search.py           # Optuna objective for the SAM method
    results/                # best configs + CSV/SVG exports per run (README.md summarizes)
  face_recognition/
    run.py                  # CLI entry point
    metrics.py              # evaluate_loo() — leave-one-out accuracy against ground_truth.json
    search.py               # Optuna objective: model_name, detector_backend, distance_metric, threshold
    results/                # best configs per run (README.md summarizes)
```

Each step's `results/` folder holds its hpopt outcomes (best params, run timings, and
any CSV/SVG/PNG exports), summarized in a `results/README.md`. The top-level
`README.md` only links to them.

## Adding a new step

1. Create `pipeline/experiments/<step>/` with:
   - `metrics.py` — load GT, define the IoU/accuracy metric for that step's output type
   - `<method>_search.py` — one per detection method; define `make_objective()`
   - `run.py` — CLI entry point that calls `create_study()` and `study.optimize()`
2. Add a script entry to `pyproject.toml`

## Ground truth format

Each step's GT lives in `config/<step>/ground_truth.json`. The frame_crop GT uses polygon quads; face_annotation uses bounding boxes. `metrics.py` owns the GT loading and IoU logic for each step.

## Running

```bash
# Local (Canny — no GPU needed)
uv run ancestry-frame-crop-hpopt --method canny --n-trials 150

# Cluster (SAM — GPU required)
./scripts/deployment/jobs/hpopt/run-hpopt-sam.sh

# Cluster (face_recognition — GPU required; one study per embedding method)
./scripts/deployment/jobs/hpopt/run-hpopt-face-recognition-deepface.sh
./scripts/deployment/jobs/hpopt/run-hpopt-face-recognition-insightface.sh

# Test / dev run
uv run ancestry-frame-crop-hpopt --method canny --n-trials 3 --wandb-project ancestry-frame-crop-hpopt-dev
```

## W&B

Each trial creates one W&B run in the `ancestry-frame-crop-hpopt` project. Requires `WANDB_API_KEY` in `.env` locally or in the `dsw-ancestry-secrets` k8s secret on the cluster.
