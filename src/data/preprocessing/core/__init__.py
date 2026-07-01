"""Core preprocessing operations."""

from .base import (
    load_image,
    save_image,
    resize_image,
    normalize_image,
    scale_bounding_box,
    get_image_size,
)

from .blur import (
    gaussian_blur,
    median_blur,
    bilateral_filter,
)

from .clahe import (
    apply_clahe,
    adaptive_histogram_equalization,
    histogram_equalization,
)

from .augmentation import (
    horizontal_flip,
    vertical_flip,
    adjust_brightness,
    adjust_contrast,
)

__all__ = [
    # Base operations
    "load_image",
    "save_image",
    "resize_image",
    "normalize_image",
    "scale_bounding_box",
    "get_image_size",
    # Blur operations
    "gaussian_blur",
    "median_blur",
    "bilateral_filter",
    # CLAHE operations
    "apply_clahe",
    "adaptive_histogram_equalization",
    "histogram_equalization",
    # Augmentation operations
    "horizontal_flip",
    "vertical_flip",
    "adjust_brightness",
    "adjust_contrast",
]
