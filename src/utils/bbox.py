"""
KLGrade Utils Module

Bounding box utilities and helper functions.
"""

from typing import Tuple


def yolo_to_xyxy_norm(
    cx: float, cy: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    """
    Convert YOLO format (normalized center coordinates) to xyxy format (normalized).

    Args:
        cx: Center x (normalized 0-1)
        cy: Center y (normalized 0-1)
        w: Width (normalized 0-1)
        h: Height (normalized 0-1)

    Returns:
        Tuple of (x1, y1, x2, y2) in normalized coordinates
    """
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


__all__ = ["yolo_to_xyxy_norm"]
