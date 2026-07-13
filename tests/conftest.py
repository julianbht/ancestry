"""Shared pytest fixtures/setup.

torch is a SAM3 dependency installed out-of-band (scripts/setup_sam3.py),
normally only present on the cluster. face_crop.detector imports it at module
level just to define FaceDetection's type hints' neighbours, so importing
pipeline.face_crop.step for its non-SAM logic fails on a plain dev machine
without it. Stub it out so those tests can run anywhere; SAM3 itself is never
exercised by these tests (SamFaceDetector imports it lazily, inside __init__).
"""

import sys
import types

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = types.ModuleType("torch")
