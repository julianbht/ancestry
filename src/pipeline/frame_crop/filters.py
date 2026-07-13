import cv2
import numpy as np

from pipeline.frame_crop.config import BlurConfig, MorphologyConfig


def apply_blur(img: np.ndarray, config: BlurConfig) -> np.ndarray:
    if config.enabled:
        return cv2.GaussianBlur(img, (config.kernel_size, config.kernel_size), 0)
    return img


def apply_morphology(mask: np.ndarray, config: MorphologyConfig) -> np.ndarray:
    ks = config.kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.iterations)
