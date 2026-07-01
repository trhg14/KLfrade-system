"""
Class Filters - Centralized implementation

Filters for removing specific classes (e.g., KL0) and remapping class IDs.
This is the core implementation used by all filtering scripts.
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.data.utils.yolo_utils import load_yolo_boxes, save_yolo_boxes


def remap_class_ids(labels: List[Dict], class_map: Dict[int, int]) -> List[Dict]:
    """
    Remap class IDs according to mapping.

    Args:
        labels: List of label dicts with 'class_id', 'x', 'y', 'w', 'h'
        class_map: Mapping from old class_id to new class_id
                  Use None value to filter out a class

    Returns:
        List of remapped labels (filtered classes removed)

    Example:
        # Remove class 0, shift others down
        class_map = {1: 0, 2: 1, 3: 2, 4: 3}  # class 0 not in map -> filtered
        labels = [{'class_id': 0, ...}, {'class_id': 1, ...}, {'class_id': 2, ...}]
        → [{'class_id': 0, ...}, {'class_id': 1, ...}]  # class 0 removed, 1→0, 2→1
    """
    remapped = []
    for label in labels:
        old_class = label["class_id"]

        # Skip if class not in map (filtered out)
        if old_class not in class_map:
            continue

        new_class = class_map[old_class]

        # Create remapped label
        remapped.append(
            {
                "class_id": new_class,
                "x": label["x"],
                "y": label["y"],
                "w": label["w"],
                "h": label["h"],
            }
        )

    return remapped


def filter_classes(
    input_dir: Path,
    output_dir: Path,
    classes_to_remove: List[int] = None,
    num_classes: int = 5,
) -> Dict:
    """
    Filter dataset by removing specified classes.

    Creates filtered dataset by removing images with only specified class labels
    and remapping remaining class IDs.

    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        classes_to_remove: List of class IDs to remove (default: [0] for KL0)
        num_classes: Original number of classes (5 or 10)
            - 5: Remove class 0 (KL0), remap 1-4 → 0-3 (output: 4 classes)
            - 10: Remove classes 0,1 (KL0-a, KL0-b), remap 2-9 → 0-7 (output: 8 classes)

    Returns:
        Dictionary with filtering statistics:
            - total_images: Total images processed
            - kept_images: Images with non-filtered labels
            - filtered_images: Images with only filtered class labels
            - original_boxes: Total boxes before filtering
            - kept_boxes: Boxes after filtering
            - filtered_boxes: Number of filtered boxes removed
    """
    # Default to removing class 0 (KL0) if not specified
    if classes_to_remove is None:
        if num_classes == 5:
            classes_to_remove = [0]  # Remove KL0
        elif num_classes == 10:
            classes_to_remove = [0, 1]  # Remove KL0-a, KL0-b
        else:
            classes_to_remove = [0]

    # Convert to set for faster lookup
    classes_to_remove_set = set(classes_to_remove)

    img_dir = input_dir / "images"
    label_dir = input_dir / "labels"

    output_img_dir = output_dir / "images"
    output_label_dir = output_dir / "labels"

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Create class mapping (all classes not in remove list get remapped sequentially)
    all_classes = set(range(num_classes))
    remaining_classes = sorted(all_classes - classes_to_remove_set)
    class_map = {old_id: new_id for new_id, old_id in enumerate(remaining_classes)}
    output_classes = len(remaining_classes)

    # Get all images
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    stats = {
        "total_images": len(images),
        "kept_images": 0,
        "filtered_images": 0,
        "original_boxes": 0,
        "kept_boxes": 0,
        "filtered_boxes": 0,
        "classes_removed": list(classes_to_remove),
        "output_classes": output_classes,
    }

    filtered_files = []

    for img_file in images:
        stem = img_file.stem
        label_file = label_dir / f"{stem}.txt"

        # Load boxes
        boxes = load_yolo_boxes(label_file)
        stats["original_boxes"] += len(boxes)

        # Filter out specified boxes and remap
        kept_boxes = []
        for box in boxes:
            if box["class_id"] in classes_to_remove_set:
                stats["filtered_boxes"] += 1
                continue

            # Remap class ID
            new_class_id = class_map.get(box["class_id"])
            if new_class_id is not None:
                kept_boxes.append(
                    {
                        "class_id": new_class_id,
                        "x": box["x"],
                        "y": box["y"],
                        "w": box["w"],
                        "h": box["h"],
                    }
                )
                stats["kept_boxes"] += 1

        # Keep image only if it has non-filtered boxes
        if kept_boxes:
            # Copy image
            shutil.copy2(img_file, output_img_dir / img_file.name)

            # Save remapped labels
            save_yolo_boxes(kept_boxes, output_label_dir / f"{stem}.txt")

            stats["kept_images"] += 1
        else:
            # Image only had filtered boxes
            filtered_files.append(stem)
            stats["filtered_images"] += 1

    # Save stats
    stats_path = output_dir / "filter_classes_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Save filtered files list
    if filtered_files:
        filtered_path = output_dir / "filtered_files.json"
        with open(filtered_path, "w") as f:
            json.dump(filtered_files, f, indent=2)

    return stats
