"""
Augmentation for Data Balancing

Functions for applying augmentations (flip, etc.) while adjusting YOLO labels.
"""

from pathlib import Path

import numpy as np
from typing import List, Tuple


def flip_horizontal(
    image: np.ndarray, labels: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Flip image horizontally and adjust YOLO bounding boxes.

    Args:
        image: Input image (numpy array)
        labels: List of YOLO label strings (format: "class_id x_center y_center width height")

    Returns:
        Tuple of (flipped_image, adjusted_labels)
    """
    flipped_image = np.fliplr(image)

    # Adjust labels
    flipped_labels = []
    for label in labels:
        parts = label.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Flip x_center (mirror across vertical axis)
        x_center_flipped = 1.0 - x_center

        # Format label
        flipped_label = f"{class_id} {x_center_flipped:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        flipped_labels.append(flipped_label)

    return flipped_image, flipped_labels


def flip_vertical(image: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Flip image vertically and adjust YOLO bounding boxes.

    Args:
        image: Input image (numpy array)
        labels: List of YOLO label strings

    Returns:
        Tuple of (flipped_image, adjusted_labels)
    """
    flipped_image = np.flipud(image)

    # Adjust labels
    flipped_labels = []
    for label in labels:
        parts = label.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Flip y_center (mirror across horizontal axis)
        y_center_flipped = 1.0 - y_center

        # Format label
        flipped_label = f"{class_id} {x_center:.6f} {y_center_flipped:.6f} {width:.6f} {height:.6f}\n"
        flipped_labels.append(flipped_label)

    return flipped_image, flipped_labels


def augment_with_label_adjustment(
    image_path: str, label_path: str, augmentation_type: str = "flip_h"
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply augmentation and return augmented image + adjusted labels.

    Args:
        image_path: Path to input image
        label_path: Path to YOLO label file
        augmentation_type: Type of augmentation ('flip_h', 'flip_v')

    Returns:
        Tuple of (augmented_image, adjusted_labels)
    """
    from src.data.image_io import read_image

    image = read_image(Path(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Load labels
    with open(label_path, "r") as f:
        labels = f.readlines()

    # Apply augmentation
    if augmentation_type == "flip_h":
        return flip_horizontal(image, labels)
    elif augmentation_type == "flip_v":
        return flip_vertical(image, labels)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
