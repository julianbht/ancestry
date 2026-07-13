# Hyperparameter Experiments

This module finds the best configuration for each pipeline step by running many parameter combinations against a labelled ground truth dataset.

## How it works

### The problem

Steps like `frame_crop` have tuneable parameters (edge detection thresholds, blur kernel sizes, SAM text prompts) that significantly affect detection accuracy. Rather than guessing good values by hand, we search over the parameter space automatically.

### Optuna — the search engine

[Optuna](https://optuna.org) manages the search loop. Each iteration is called a **trial**. In a trial, Optuna proposes a set of parameter values, the objective function evaluates how well those parameters perform, and Optuna records the result.

After a few random trials, Optuna switches to **TPE** (Tree-structured Parzen Estimator) — a Bayesian method that learns which regions of the parameter space tend to score well and proposes values there more often. This finds good configurations much faster than an exhaustive grid search.

```
Optuna study
  ├── trial 0:  random params  →  accuracy 0.13  →  "explore"
  ├── trial 1:  random params  →  accuracy 0.09  →  "explore"
  ├── trial 2:  random params  →  accuracy 0.27  →  "explore"
  ...
  ├── trial 12: TPE proposal   →  accuracy 0.61  →  "exploit"
  └── trial N:  best so far
```

The objective function for each step:
1. Builds a detector with the trial's parameters
2. Runs it on every image in the ground truth set
3. Computes polygon IoU between the predicted quad and the labelled quad
4. Returns **accuracy** = fraction of images with IoU ≥ 0.75

### W&B — experiment tracking

[Weights & Biases](https://wandb.ai) records every trial so you can inspect results in the browser. Each trial creates one W&B run containing:

- The parameter values that were tried (logged as run config)
- `accuracy` — fraction of GT images correctly detected at IoU ≥ 0.75
- `mean_iou` — average IoU across all GT images

W&B lets you sort runs by accuracy, plot accuracy vs. individual parameters (e.g. "how does threshold1 affect accuracy?"), and compare the best and worst trials side-by-side.

## frame_crop search spaces

### Canny (`--method canny`)

| Parameter | Range | Notes |
|---|---|---|
| `threshold1` | 10 – 200 | Lower hysteresis threshold for edge detection |
| `threshold2` | 10 – 400 | Upper hysteresis threshold |
| `blur_kernel_size` | 3, 5, 7, 9, 11 | Gaussian blur before edge detection |
| `morph_kernel_size` | 3, 5, 7 | Morphological closing kernel (connects broken edges) |
| `morph_iterations` | 1 – 15 | How many times morphology is applied |

~150 trials is enough. Each trial takes ~5 seconds locally (no GPU needed).

### SAM (`--method sam`)

| Parameter | Range | Notes |
|---|---|---|
| `prompt` | 13 candidate strings | The text description SAM uses to locate the photo print |
| `score_threshold` | 0.2 – 0.8 | Minimum SAM confidence to accept a mask |

The SAM model is loaded once before the search starts and reused across all trials (loading it per trial would take ~30s each). ~50 trials is sufficient since the prompt space is small. Requires GPU.

## face_recognition search space

Unlike frame_crop, face_recognition's ground truth is small (a few dozen
identified people, most with only a handful of photos), so there's no room
for a fixed train/test split — instead the objective runs a **leave-one-out**
(LOO) pass: every labeled face is held out and queried against the gallery
built from every *other* labeled face exactly once per trial. See
`metrics.py` for why both a "known" (recall) and a "negative" (specificity)
side are scored — recall alone would let the search push the threshold
arbitrarily loose.

| Parameter | Values | Notes |
|---|---|---|
| `model_name` | ArcFace, Facenet512, VGG-Face | DeepFace embedding model |
| `detector_backend` | skip, opencv, retinaface, mtcnn | re-detection/alignment inside the crop before embedding |
| `distance_metric` | cosine, euclidean, euclidean_l2 | nearest-neighbour distance |
| `threshold` | metric-dependent range | match cutoff; searched per-metric since each has its own scale |

`(model_name, detector_backend)` combos are embedded once and cached for the
rest of the study — `distance_metric`/`threshold` are then nearly free to
vary, so re-embedding (the slow part once `detector_backend` is on) only
happens once per unique combo, not once per trial.

One Optuna study per embedding method, so run once per `--method`:

```bash
uv run ancestry-face-recognition-hpopt --method deepface --n-trials 50
uv run ancestry-face-recognition-hpopt --method insightface --n-trials 50

# Test / dev run
uv run ancestry-face-recognition-hpopt --method deepface --n-trials 2 --wandb-project ancestry-face-recognition-hpopt-dev
```

On the cluster these are two overlays (mirrors the canny/sam split):

```bash
./scripts/deployment/jobs/hpopt/run-hpopt-face-recognition-deepface.sh
./scripts/deployment/jobs/hpopt/run-hpopt-face-recognition-insightface.sh
```

## Metric detail

For each image the detector produces a quadrilateral (4 corner points). The ground truth is also a quad annotated in Label Studio. IoU is computed as:

```
IoU = intersection_area / union_area
```

using OpenCV's `intersectConvexConvex`. A detection is considered correct if IoU ≥ 0.75.

## Results

Per-step results (best configs, run timings, and CSV/SVG exports) live in each
step's `results/` folder:

- **frame_crop** (Canny + SAM 3.1) — [`frame_crop/results/README.md`](frame_crop/results/README.md)
- **face_recognition** — [`face_recognition/results/README.md`](face_recognition/results/README.md)

