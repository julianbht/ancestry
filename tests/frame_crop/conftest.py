import numpy as np
import pytest

from pipeline.frame_crop.config import FrameCropConfig

# Axis-aligned quad inside a 1000x800 image: x 100-700, y 150-900.
SAMPLE_QUAD = np.array([[100, 150], [700, 150], [700, 900], [100, 900]], dtype=np.float32)


@pytest.fixture
def sample_quad() -> np.ndarray:
    return SAMPLE_QUAD.copy()


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.zeros((1000, 800, 3), dtype=np.uint8)


@pytest.fixture
def make_config():
    def _make(margin_px: int = 0, inner_crop: bool = True) -> FrameCropConfig:
        return FrameCropConfig.model_validate(
            {
                "max_files_to_crop": None,
                "only_file": None,
                "only_ground_truth": False,
                "method": "sam",
                "min_area_fraction": 0.15,
                "max_area_fraction": 0.95,
                "margin_px": margin_px,
                "inner_crop": inner_crop,
                "jpeg_quality": 100,
                "ignore_state": False,
                "skip_state_write": False,
                "skip_output_write": False,
                "debug": {
                    "save_quad": False,
                    "annotate_quad": False,
                    "annotation": {"scale": 1.0, "bold": False},
                },
            }
        )

    return _make
