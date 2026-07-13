# Deployment

## Image Hierarchy

```
Dockerfile.cuda-base  →  julianbht/ancestry-pipeline:cuda-base
    ├── Dockerfile.ssh-cuda  →  :ssh-cuda   (dev: SSH access, Claude CLI, git clone at boot)
    └── Dockerfile.job-cuda  →  :job-cuda   (prod: code baked in, generic GPU runner)
```

`cuda-base` provides Python 3.12, uv, ffmpeg, libsm6/libxext6, git. torch is **not** in any image — it lives on the PVC venv (see below).

## PVC Venv Strategy

Both cuda images point uv at `/app/data/.venv` (on the PVC) via `UV_PROJECT_ENVIRONMENT`. This means:

- torch + SAM3 are installed once by `setup_sam3.py` (idempotent) and persist across pod restarts
- `UV_NO_SYNC=1` prevents implicit `uv run` syncs from pruning those out-of-band packages
- `uv sync --frozen --no-dev --inexact` is used instead of a plain sync — `--inexact` preserves packages not in `uv.lock`
- `UV_LINK_MODE=copy` silences hardlink warnings (overlay FS vs PVC are different filesystems)

## Job Startup Sequence (`job-cuda`)

The run command is specified in each job's k8s manifest — the image has no baked-in CMD. The typical sequence:

```
uv sync --frozen --no-dev --inexact   # install project deps, preserve torch/SAM3
uv run python scripts/setup_sam3.py  # idempotent: clone + install SAM3 if not on PVC (GPU jobs only)
uv run <entry-point>                  # run the pipeline step or experiment
```

## Kaniko Build Jobs

| Manifest | Builds | When to run |
|---|---|---|
| `kaniko-build-cuda-base-job.yml` | `cuda-base` | CUDA version or apt packages change |
| `kaniko-build-job.yml` | `ssh-cuda` | SSH image changes (entrypoint, sshd_config) |
| `kaniko-build-job-cuda.yml` | `job-cuda` | Pipeline code, config, or scripts change |

Build order: `cuda-base` must exist before building either dependent image.

## K8s Manifests (Kustomize)

Manifests use [Kustomize](https://kustomize.io/) with a shared base per resource type and per-job/deployment overlays. Apply with `kubectl apply -k <overlay-dir>`.

```
k8s/
  jobs/
    base/             # shared Job spec (image, volumes, NEXTCLOUD_PASSWORD, ZIP_PASSWORD)
    frame-crop/       # ancestry-frame-crop, GPU
    face-crop/        # ancestry-face-crop, GPU
    face-recognition/ # ancestry-face-recognition, GPU
    hpopt-canny/      # ancestry-frame-crop-hpopt --method canny, CPU only, adds WANDB_API_KEY
    hpopt-sam/        # ancestry-frame-crop-hpopt --method sam, GPU, adds WANDB_API_KEY
    hpopt-face-recognition-deepface/      # ancestry-face-recognition-hpopt --method deepface, GPU, adds WANDB_API_KEY
    hpopt-face-recognition-insightface/   # ancestry-face-recognition-hpopt --method insightface, GPU, adds WANDB_API_KEY
  ssh/
    base/             # shared Deployment spec (all secrets, volumes, GPU, nodeSelector)
    cuda/             # ssh-cuda image
    dev/              # ssh image (no CUDA libs)
```

`data-pvc.yml` — the shared PVC mounted at `/app/data` in all pods.

## Helper Scripts

- `scripts/deployment/jobs/run-step.sh <step> [--env dev]` — apply + follow logs for any step under `deployment/k8s/jobs/` (e.g. `frame-crop`, `face-crop`). Fails with the list of available steps if `<step>` is missing or unknown.
- `scripts/deployment/jobs/hpopt/run-hpopt-canny.sh` / `run-hpopt-sam.sh` / `run-hpopt-face-recognition-deepface.sh` / `run-hpopt-face-recognition-insightface.sh` — same idea, kept as dedicated scripts since they're experiments rather than pipeline steps. One overlay per embedding method (mirrors the canny/sam split), since each method is a separate Optuna study.
- `scripts/deployment/ssh/ssh-connect.sh --cuda | --dev` — apply SSH deployment + port-forward
- All job scripts delegate to `rerun-job.sh <job-name> <kustomize-dir> [--env dev]`

## Environment Selection

The SSH image sets `ENV=dev` (via profile.d); the job image sets `ENV=prod` (via Docker ENV). See the `deployment-env` skill for details on how steps pick their config file.
