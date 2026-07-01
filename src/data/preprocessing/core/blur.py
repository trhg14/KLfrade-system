"""
Core preprocessing operations - Blur module.
"""

import cv2
import numpy as np
from typing import Tuple


def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma: float = 0) -> np.ndarray:
    """Apply Gaussian blur to an image."""
    return cv2.GaussianBlur(image, kernel_size, sigma)


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median blur to an image."""
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """Apply bilateral filter to an image."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
