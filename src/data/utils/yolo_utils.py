"""
YOLO Utilities - Centralized YOLO box handling functions.

This module consolidates all YOLO box manipulation functions that were
previously duplicated across multiple scripts.
"""

from pathlib import Path
from typing import List, Dict, Tuple


def load_yolo_boxes(label_path: Path) -> List[Dict]:
    """
    Load YOLO format boxes from label file.

    Args:
        label_path: Path to .txt label file

    Returns:
        List of boxes, each dict with keys:
            - class_id (int): Class ID
            - x, y, w, h (float): Normalized coordinates [0,1]

    Format:
        class_id x_center y_center width height
        Example: 2 0.5 0.5 0.3 0.4
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 5:
                try:
                    class_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    boxes.append({"class_id": class_id, "x": x, "y": y, "w": w, "h": h})
                except ValueError:
                    continue

    return boxes


def save_yolo_boxes(boxes: List[Dict], output_path: Path):
    """
    Save boxes in YOLO format.

    Args:
        boxes: List of box dicts with 'class_id', 'x', 'y', 'w', 'h'
        output_path: Path to output .txt file
    """
    lines = []
    for box in boxes:
        line = f"{box['class_id']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}"
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n" if lines else "")


def yolo_to_pixel_box(box: Dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized box (center + size) to pixel coordinates (x1, y1, x2, y2).

    Args:
        box: Dict with 'x', 'y', 'w', 'h' (normalized [0,1])
        img_w, img_h: Image dimensions in pixels

    Returns:
        (x1, y1, x2, y2) in pixels

    Example:
        box = {'x': 0.5, 'y': 0.5, 'w': 0.3, 'h': 0.4}
        img_w, img_h = 1000, 1000
        → (350, 300, 650, 700)
    """
    x_center = box["x"] * img_w
    y_center = box["y"] * img_h
    w = box["w"] * img_w
    h = box["h"] * img_h

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return x1, y1, x2, y2


def pixel_to_yolo_box(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> Dict:
    """
    Convert pixel coordinates to YOLO normalized format.

    Args:
        x1, y1, x2, y2: Pixel coordinates
        img_w, img_h: Image dimensions

    Returns:
        Dict with 'x', 'y', 'w', 'h' (normalized [0,1])
    """
    w = x2 - x1
    h = y2 - y1
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    return {
        "x": x_center / img_w,
        "y": y_center / img_h,
        "w": w / img_w,
        "h": h / img_h,
    }


def expand_box_to_square(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, margin: float = 0.15
) -> Tuple[int, int, int, int]:
    """
    Expand bounding box to square with margin, clamped to image boundaries.

    Process:
        1. Calculate current box dimensions
        2. Take max(width, height) as size
        3. Add margin percentage
        4. Center the square at original box center
        5. Clamp to image bounds

    Args:
        x1, y1, x2, y2: Original box coordinates (pixels)
        img_w, img_h: Image dimensions (pixels)
        margin: Margin fraction (e.g., 0.15 = 15%)

    Returns:
        Square box (x1, y1, x2, y2) clamped to [0, img_w] x [0, img_h]

    Example:
        Box: (200, 300, 300, 400) → w=100, h=100
        Image: 1000x1000
        Margin: 0.15
        → Square size: 100 * 1.15 = 115
        → Expand from center (250, 350)
        → Output: (192, 292, 307, 407)
    """
    box_w = x2 - x1
    box_h = y2 - y1

    # Make square
    size = max(box_w, box_h)
    size_with_margin = int(size * (1 + margin))

    # Calculate center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # New square coordinates
    new_x1 = int(cx - size_with_margin / 2)
    new_y1 = int(cy - size_with_margin / 2)
    new_x2 = int(cx + size_with_margin / 2)
    new_y2 = int(cy + size_with_margin / 2)

    # Clamp to image bounds
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)

    return new_x1, new_y1, new_x2, new_y2
