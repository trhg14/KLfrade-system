"""
Core preprocessing operations - Base module.

Provides fundamental image operations:
- Loading and saving images
- Resizing
- Normalization
- Bounding box scaling
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_image(image_path: str, mode: str = "grayscale") -> np.ndarray:
    """
    Load image from path.

    Args:
        image_path: Path to image file
        mode: 'grayscale' or 'rgb'

    Returns:
        Image as numpy array
    """
    if mode == "grayscale":
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    elif mode == "rgb":
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return image


def save_image(image: np.ndarray, output_path: str, format: str = "png") -> None:
    """
    Save image to path.

    Args:
        image: Image as numpy array
        output_path: Path to save image
        format: Output format ('png', 'jpg')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure correct extension
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{format}")

    cv2.imwrite(str(output_path), image)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size.

    Args:
        image: Input image
        target_size: (width, height) tuple

    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image (uint8)

    Returns:
        Normalized image (float32)
    """
    return image.astype(np.float32) / 255.0


def scale_bounding_box(
    bbox: list, original_size: Tuple[int, int], new_size: Tuple[int, int]
) -> list:
    """
    Scale bounding box coordinates from original size to new size.

    Args:
        bbox: [cx, cy, w, h] in normalized coordinates [0, 1]
        original_size: (width, height) of original image
        new_size: (width, height) of resized image

    Returns:
        Scaled bbox in normalized coordinates
    """
    # YOLO format is already normalized, so no scaling needed
    # This function is kept for API compatibility
    return bbox


def get_image_size(image: np.ndarray) -> Tuple[int, int]:
    """
    Get image size (width, height).

    Args:
        image: Input image

    Returns:
        (width, height) tuple
    """
    if len(image.shape) == 2:  # Grayscale
        h, w = image.shape
    else:  # Color
        h, w, _ = image.shape
    return (w, h)
