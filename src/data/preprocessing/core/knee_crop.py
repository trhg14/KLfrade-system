"""
Knee cropping preprocessing operations.

Crops knee regions from full X-ray images using existing knee bounding box labels.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import centralized YOLO utilities
from src.data.utils.yolo_utils import (
    load_yolo_boxes,
    save_yolo_boxes,
    yolo_to_pixel_box,
    expand_box_to_square,
)


# Note: load_yolo_boxes, yolo_to_pixel_box, and expand_box_to_square
# are now imported from src.data.utils.yolo_utils


def transform_labels_to_crop_space(
    labels: List[Dict],
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
    img_w: int,
    img_h: int,
) -> List[Dict]:
    """
    Transform labels from full image space to crop space.

    Args:
        labels: List of label dicts with 'class_id', 'x', 'y', 'w', 'h' (normalized)
        crop_x1, crop_y1, crop_x2, crop_y2: Crop region in pixels
        img_w, img_h: Original image dimensions

    Returns:
        Transformed labels in crop space (normalized [0,1])
    """
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w <= 0 or crop_h <= 0:
        return []

    transformed = []

    for label in labels:
        # Convert to pixel coordinates in full image
        x_center = label["x"] * img_w
        y_center = label["y"] * img_h
        width = label["w"] * img_w
        height = label["h"] * img_h

        # Check if label center is within crop region (with tolerance)
        if not (
            crop_x1 - width <= x_center <= crop_x2 + width
            and crop_y1 - height <= y_center <= crop_y2 + height
        ):
            continue  # Skip labels outside crop

        # Transform to crop coordinates
        new_x_center = x_center - crop_x1
        new_y_center = y_center - crop_y1

        # Clamp box boundaries
        box_x1 = max(0, new_x_center - width / 2)
        box_y1 = max(0, new_y_center - height / 2)
        box_x2 = min(crop_w, new_x_center + width / 2)
        box_y2 = min(crop_h, new_y_center + height / 2)

        # Recalculate after clamping
        clamped_w = box_x2 - box_x1
        clamped_h = box_y2 - box_y1
        clamped_x_center = (box_x1 + box_x2) / 2
        clamped_y_center = (box_y1 + box_y2) / 2

        # Normalize to crop space
        norm_x = np.clip(clamped_x_center / crop_w, 0.0, 1.0)
        norm_y = np.clip(clamped_y_center / crop_h, 0.0, 1.0)
        norm_w = np.clip(clamped_w / crop_w, 0.01, 1.0)
        norm_h = np.clip(clamped_h / crop_h, 0.01, 1.0)

        transformed.append(
            {
                "class_id": label["class_id"],
                "x": norm_x,
                "y": norm_y,
                "w": norm_w,
                "h": norm_h,
            }
        )

    return transformed


def crop_knee_from_image(
    image: np.ndarray,
    knee_label_path: Path,
    kl_label_path: Optional[Path] = None,
    margin: float = 0.15,
    knee_index: int = 0,
) -> Tuple[np.ndarray, List[Dict], Dict]:
    """
    Crop knee region from full X-ray image.

    Args:
        image: Full X-ray image
        knee_label_path: Path to knee labels file
        kl_label_path: Optional path to KL labels file
        margin: Margin around knee box (0.15 = 15%)
        knee_index: Which knee to crop (0 = first, 1 = second, etc.)

    Returns:
        Tuple of:
            - Cropped knee image
            - Transformed KL labels (if kl_label_path provided)
            - Metadata dict with crop info
    """
    img_h, img_w = image.shape[:2]

    # Load knee boxes
    knee_boxes = load_yolo_boxes(knee_label_path)

    if not knee_boxes or knee_index >= len(knee_boxes):
        # No knee box found, return original image
        return image, [], {"cropped": False, "reason": "no_knee_box"}

    # Get the specified knee box
    knee_box = knee_boxes[knee_index]

    # Convert to pixel coordinates
    x1, y1, x2, y2 = yolo_to_pixel_box(knee_box, img_w, img_h)

    # Expand to square with margin
    crop_x1, crop_y1, crop_x2, crop_y2 = expand_box_to_square(
        x1, y1, x2, y2, img_w, img_h, margin
    )

    # Crop image
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Transform KL labels if provided
    transformed_labels = []
    if kl_label_path and kl_label_path.exists():
        kl_labels = load_yolo_boxes(kl_label_path)  # Same format as knee boxes
        transformed_labels = transform_labels_to_crop_space(
            kl_labels, crop_x1, crop_y1, crop_x2, crop_y2, img_w, img_h
        )

    metadata = {
        "cropped": True,
        "knee_index": knee_index,
        "total_knees": len(knee_boxes),
        "crop_box": (crop_x1, crop_y1, crop_x2, crop_y2),
        "original_size": (img_w, img_h),
        "crop_size": (crop_x2 - crop_x1, crop_y2 - crop_y1),
    }

    return cropped_image, transformed_labels, metadata
