"""
One-time setup for SAM 3.1: clone the repo, patch and install it into the
system Python environment, and pre-download model checkpoints to /data/models/sam3/.

Run with:
    export HF_TOKEN=<your_token>
    uv run python scripts/setup_sam3.py

Safe to re-run — each step checks whether its work is already done.
"""

import os
import subprocess
import sys
from pathlib import Path

SAM3_REPO_URL = "https://github.com/facebookresearch/sam3.git"
SAM3_CODE_DIR = Path("/app/data/sam3")
SAM3_MODEL_DIR = Path("/app/data/models/sam3")
# build_sam3_image_model builds the SAM 3 *image* architecture (dual-neck, convs 0-3).
# Its matching weights are facebook/sam3's sam3.pt. The SAM 3.1 multiplex checkpoint is
# a *video* model (tri-neck, convs 0-2) and loads with convs.3 randomly initialized
# (the missing-keys warning) -> no detections. Frame crop is single-image, so use sam3.pt.
HF_IMAGE_REPO = "facebook/sam3"
IMAGE_CKPT_NAME = "sam3.pt"

# torch is not in the base image — uv must install it into the project venv.
# Pre-installing the cu126 build here prevents SAM3's dependency resolution (via timm)
# from picking the latest torch build (cu13x), which requires CUDA driver 13.x and is
# incompatible with the cluster's CUDA 12.x drivers.
_TORCH_VERSION = "2.7.0"
_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"


def run(cmd: list[str], **kwargs) -> None:
    print(f"+ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def step_clone() -> None:
    if SAM3_CODE_DIR.exists():
        print(f"[clone] Already present at {SAM3_CODE_DIR} — pulling latest.")
        run(["git", "-C", str(SAM3_CODE_DIR), "pull"])
    else:
        print(f"[clone] Cloning into {SAM3_CODE_DIR} ...")
        run(["git", "clone", SAM3_REPO_URL, str(SAM3_CODE_DIR)])


def _patch_toml(label: str, old: str, new: str) -> None:
    toml_path = SAM3_CODE_DIR / "pyproject.toml"
    original = toml_path.read_text()
    patched = original.replace(old, new)
    if patched == original:
        print(f"[patch] {label} already patched or not found — skipping.")
    else:
        toml_path.write_text(patched)
        print(f"[patch] {label}")


def step_patch_einops() -> None:
    """einops is used unconditionally in sam/rope.py but only declared in [notebooks] extras."""
    _patch_toml(
        "Moved einops to core dependencies.",
        '"huggingface_hub",\n]',
        '"huggingface_hub",\n    "einops",\n]',
    )


def step_patch_pycocotools() -> None:
    """pycocotools is imported unconditionally via model_builder -> train/data chain but only declared in dev/notebooks extras."""
    _patch_toml(
        "Moved pycocotools to core dependencies.",
        '"einops",\n]',
        '"einops",\n    "pycocotools",\n]',
    )


def step_patch_psutil() -> None:
    """psutil is imported unconditionally in sam3_video_predictor.py but not declared as a dependency."""
    _patch_toml(
        "Moved psutil to core dependencies.",
        '"pycocotools",\n]',
        '"pycocotools",\n    "psutil",\n]',
    )


def step_patch_numpy() -> None:
    """SAM pins numpy<2 but the pipeline requires numpy>=2 (opencv, label-studio). SAM runs fine on 2.x."""
    _patch_toml(
        "Relaxed numpy pin from <2 to >=2.0.0.",
        "numpy>=1.26,<2",
        "numpy>=2.0.0",
    )


def step_patch_pkg_resources() -> None:
    """setuptools 81+ removed pkg_resources; replace the three calls with importlib.resources."""
    mb_path = SAM3_CODE_DIR / "sam3" / "model_builder.py"
    original = mb_path.read_text()

    # Replace import
    patched = original.replace(
        "import pkg_resources\n",
        "from pathlib import Path as _Path\n_SAM3_PKG_DIR = _Path(__file__).parent\n",
    )
    # Replace all three identical resource_filename calls
    patched = patched.replace(
        'pkg_resources.resource_filename(\n            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"\n        )',
        'str(_SAM3_PKG_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz")',
    )

    if patched == original:
        print("[patch] pkg_resources already patched or pattern not found — skipping.")
    else:
        mb_path.write_text(patched)
        print("[patch] Replaced pkg_resources with pathlib-based resource lookup.")



def step_install_torch() -> None:
    expected = f"{_TORCH_VERSION}+cu126"
    result = subprocess.run(["uv", "pip", "show", "torch"], capture_output=True, text=True)
    if result.returncode == 0:
        ver_line = next((l for l in result.stdout.splitlines() if l.startswith("Version:")), "")
        ver = ver_line.split(": ", 1)[1].strip() if ver_line else ""
        if ver == expected:
            print(f"[torch] Already installed: {ver} — skipping.")
            return
        print(f"[torch] ERROR: expected torch {expected} but found {ver}.", file=sys.stderr)
        sys.exit(1)
    print(f"[torch] Not found — installing {expected} ...")
    run([
        "uv", "pip", "install",
        f"torch=={expected}", "torchvision==0.22.0+cu126",
        "--index-url", _TORCH_INDEX_URL,
    ])


def step_install() -> None:
    print("[install] Installing SAM3 (editable) into the system Python environment ...")
    run(["uv", "pip", "install", "-e", str(SAM3_CODE_DIR)])


def _checkpoint_present() -> bool:
    return (SAM3_MODEL_DIR / IMAGE_CKPT_NAME).exists()


def step_hf_login() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[hf-login] ERROR: HF_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)
    print("[hf-login] Saving HF token to local cache ...")
    print("+ hf auth login --token <redacted>")
    subprocess.run(["hf", "auth", "login", "--token", token], check=True)


def step_download() -> None:
    SAM3_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if _checkpoint_present():
        print(f"[download] {IMAGE_CKPT_NAME} already present — skipping.")
        return

    print(f"[download] Downloading {HF_IMAGE_REPO}/{IMAGE_CKPT_NAME} to {SAM3_MODEL_DIR} ...")
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=HF_IMAGE_REPO,
        filename=IMAGE_CKPT_NAME,
        local_dir=str(SAM3_MODEL_DIR),
    )
    print("[download] Done.")


def step_verify() -> None:
    print("[verify] Checking torch + model signature ...")
    # Import in-process: torch/sam3 were just installed into this interpreter's
    # venv, so a subprocess only adds a second cold torch import (slow) and dumps
    # the source via run()'s "+ ..." echo. Importing here verifies the same thing.
    import inspect

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
    from sam3.model_builder import build_sam3_image_model

    if torch.cuda.is_available():
        print(f"  torch {torch.__version__}, device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  torch {torch.__version__}, device: cpu (no CUDA)")
    print(f"  build_sam3_image_model signature: {inspect.signature(build_sam3_image_model)}")


def main() -> None:
    print("=== SAM 3.1 setup ===\n")
    step_clone()
    print()
    step_patch_einops()
    print()
    step_patch_pycocotools()
    print()
    step_patch_psutil()
    print()
    step_patch_numpy()
    print()
    step_patch_pkg_resources()
    print()
    step_install_torch()
    print()
    step_install()
    print()
    if not _checkpoint_present():
        step_hf_login()
        print()
    step_download()
    print()
    step_verify()
    print()
    print("=== Setup complete ===")
    print(f"  Code:        {SAM3_CODE_DIR}")
    print(f"  Checkpoints: {SAM3_MODEL_DIR}")


if __name__ == "__main__":
    main()
