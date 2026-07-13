---
name: deployment-env
description: How the ancestry pipeline selects its dev/prod environment and configs across SSH deployments vs Jobs. Use when adding an env var, a step config, a new Dockerfile/k8s manifest, or when debugging why a step loaded the wrong step.<env>.yaml or an env var is missing over SSH.
---

# Deployment environment & config selection

## The one rule

**The image owns config; k8s owns secrets.**

- **Non-secret, determined by *which image you run*** (`ENV`, the `UV_*` constants) → hardcode in the **Dockerfile**.
- **Secret or cluster-specific** (`*_PASSWORD`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, `GITHUB_TOKEN`) → **k8s manifest** via `secretKeyRef`. Never hardcode these.

The mapping is intentional: the **ssh image IS dev**, the **job image IS prod**. That fact lives in the image, not in a manifest you must remember to set.

## How a step picks its config

`src/pipeline/shared/config.py` resolves the config file purely from the `ENV` env var:

```python
env = os.environ.get("ENV", "prod")          # default is prod (safe for a research pipeline)
config_file = CONFIG_DIR / step / f"step.{env}.yaml"
```

So a step loads `config/<step>/step.dev.yaml` when `ENV=dev`, else `step.prod.yaml`. If the matching file is missing, it raises `FileNotFoundError`. **Every step that calls `load_config` must therefore have BOTH `step.dev.yaml` and `step.prod.yaml`.** (`rotate` is exempt — it never loads a config.)

## Why `ENV` must be set via `/etc/profile.d`, not k8s `env:` (for SSH)

The SSH images run `sshd`, and you SSH in to run steps (`uv run ...`). **sshd does NOT pass its own environment to the sessions it spawns** — it builds a fresh environment per login from sshd defaults, PAM/`/etc/environment`, `AcceptEnv` (only `LANG LC_*` here), and the login shell sourcing `/etc/profile` → `/etc/profile.d/*.sh`.

So a k8s `env:` value or Docker `ENV` reaches **PID 1 (the entrypoint)** but **not your SSH shell**. Anything an interactive SSH session needs must be in `/etc/profile.d/`. Some vars (`PATH`, `UV_PROJECT_ENVIRONMENT`) need *both* the entrypoint and the SSH shell, so they appear twice — that's inherent, not a mistake.

`ENV` is set in profile.d per image:
- `Dockerfile.ssh`, `Dockerfile.ssh-cuda` → `export ENV=dev`

`Dockerfile.job-cuda` sets `ENV=prod` via the Docker `ENV` instruction directly — it has no `sshd`/login shell to worry about, since the run command is supplied straight into PID 1 via the k8s manifest.

`config.py`'s default of `prod` is the safety net: a shell that somehow lost `ENV` falls back to prod, never silently runs dev. (Historically the job "worked" only because of this default — now it's explicit.)

## Checklists

**Adding a pipeline step that loads config:** create BOTH `config/<step>/step.dev.yaml` and `config/<step>/step.prod.yaml`, even if identical for now.

**Adding a non-secret env var the SSH shell needs:** add `export FOO=...` to a `/etc/profile.d/*.sh` in the relevant Dockerfile(s). If the entrypoint also needs it, add a Docker `ENV FOO=...` too. Do NOT add it to the k8s manifest.

**Adding a secret:** add it to the k8s `secrets` and reference it with `secretKeyRef` in the manifest(s). It will reach PID 1 but NOT SSH sessions — that's accepted; we don't write secrets into profile.d on disk.

## Gotcha

Hardcoding the selector reads as `ENV ENV=dev` if you use the Docker `ENV` instruction (keyword collision with the var name). Use profile.d (`export ENV=dev`) to avoid it. If the collision ever bothers you, rename the var to `PIPELINE_ENV` — it's one line in `config.py` plus the profile.d scripts.
