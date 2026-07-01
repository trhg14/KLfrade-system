"""
Core preprocessing operations - Augmentation module.

Provides data augmentation operations with label handling.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def horizontal_flip(
    image: np.ndarray, labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Flip image horizontally and adjust labels.

    Args:
        image: Input image
        labels: List of YOLO format labels (optional)

    Returns:
        Tuple of (flipped_image, adjusted_labels)
    """
    flipped_image = cv2.flip(image, 1)

    if labels is None:
        return flipped_image, None

    new_labels = []
    for label in labels:
        parts = label.strip().split()
        if len(parts) >= 5:
            class_id = parts[0]
            cx, cy, w, h = map(float, parts[1:5])
            # Flip horizontal: cx' = 1 - cx
            new_cx = 1.0 - cx
            new_label = f"{class_id} {new_cx} {cy} {w} {h}"
            new_labels.append(new_label)

    return flipped_image, new_labels


def vertical_flip(
    image: np.ndarray, labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Flip image vertically and adjust labels.

    Args:
        image: Input image
        labels: List of YOLO format labels (optional)

    Returns:
        Tuple of (flipped_image, adjusted_labels)
    """
    flipped_image = cv2.flip(image, 0)

    if labels is None:
        return flipped_image, None

    new_labels = []
    for label in labels:
        parts = label.strip().split()
        if len(parts) >= 5:
            class_id = parts[0]
            cx, cy, w, h = map(float, parts[1:5])
            # Flip vertical: cy' = 1 - cy
            new_cy = 1.0 - cy
            new_label = f"{class_id} {cx} {new_cy} {w} {h}"
            new_labels.append(new_label)

    return flipped_image, new_labels


def adjust_brightness(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    Adjust image brightness.

    Args:
        image: Input image (uint8)
        factor: Brightness factor (>1 = brighter, <1 = darker)

    Returns:
        Adjusted image
    """
    adjusted = image.astype(np.float32) * factor
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    Adjust image contrast.

    Args:
        image: Input image (uint8)
        factor: Contrast factor (>1 = more contrast, <1 = less contrast)

    Returns:
        Adjusted image
    """
    mean = np.mean(image)
    adjusted = (image.astype(np.float32) - mean) * factor + mean
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)
