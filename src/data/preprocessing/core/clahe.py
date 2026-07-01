"""
Core preprocessing operations - CLAHE module.
"""

import cv2
import numpy as np
from typing import Tuple


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def adaptive_histogram_equalization(
    image: np.ndarray,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply adaptive histogram equalization without contrast clipping."""
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply standard (global) histogram equalization."""
    return cv2.equalizeHist(image)
