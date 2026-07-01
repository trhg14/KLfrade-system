"""
YOLO Label Utilities

Functions for YOLO-specific label transformations including bounding box scaling,
class filtering, and remapping.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def scale_bounding_box(
    bbox: List[float],
    original_size: Tuple[int, int],
    new_size: Tuple[int, int] = (640, 640),
) -> List[float]:
    """
    Scale YOLO bounding box coordinates after image resize.

    Args:
        bbox: [x_center, y_center, width, height] in normalized coordinates (0-1)
        original_size: (width, height) of original image
        new_size: (width, height) of resized image

    Returns:
        Scaled bbox [x_center, y_center, width, height]
    """
    x_center, y_center, width, height = bbox

    # Convert from normalized to pixel coordinates (original image)
    x_center_pixel = x_center * original_size[0]
    y_center_pixel = y_center * original_size[1]
    width_pixel = width * original_size[0]
    height_pixel = height * original_size[1]

    # Scale to new image size
    x_scale = new_size[0] / original_size[0]
    y_scale = new_size[1] / original_size[1]

    x_center_scaled = x_center_pixel * x_scale
    y_center_scaled = y_center_pixel * y_scale
    width_scaled = width_pixel * x_scale
    height_scaled = height_pixel * y_scale

    # Convert back to normalized coordinates (new image)
    x_center_new = x_center_scaled / new_size[0]
    y_center_new = y_center_scaled / new_size[1]
    width_new = width_scaled / new_size[0]
    height_new = height_scaled / new_size[1]

    return [x_center_new, y_center_new, width_new, height_new]


def process_labels_after_resize(
    label_path: Union[str, Path],
    original_size: Tuple[int, int],
    new_size: Tuple[int, int] = (640, 640),
) -> List[str]:
    """
    Process all labels in file after image resize.

    Args:
        label_path: Path to YOLO label file
        original_size: (width, height) of original image
        new_size: (width, height) of resized image

    Returns:
        List of scaled label strings
    """
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_labels = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])
        bbox = [float(x) for x in parts[1:5]]

        scaled_bbox = scale_bounding_box(bbox, original_size, new_size)

        new_label = f"{class_id} {' '.join(f'{x:.6f}' for x in scaled_bbox)}\n"
        new_labels.append(new_label)

    return new_labels


def filter_classes(
    label_dir: Union[str, Path],
    output_dir: Union[str, Path],
    classes_to_keep: Optional[List[int]] = None,
    classes_to_exclude: Optional[List[int]] = None,
    remap: bool = True,
) -> Dict[str, int]:
    """
    Filter labels to keep/exclude specific classes, optionally remap IDs.

    Args:
        label_dir: Input label directory
        output_dir: Output label directory
        classes_to_keep: List of class IDs to keep (if None, all kept)
        classes_to_exclude: List of class IDs to exclude (if None, none excluded)
        remap: Whether to remap class IDs to contiguous range [0, N-1]

    Returns:
        Dictionary with filtering statistics
    """
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which classes to keep
    if classes_to_exclude is not None:
        exclude_set = set(classes_to_exclude)
    else:
        exclude_set = set()

    if classes_to_keep is not None:
        keep_set = set(classes_to_keep)
    else:
        keep_set = None

    # Build class mapping if remapping
    if remap:
        if keep_set is not None:
            sorted_classes = sorted(keep_set)
        else:
            # Need to scan all labels to find all classes
            all_classes = set()
            for label_file in label_dir.glob("*.txt"):
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_classes.add(int(parts[0]))
            sorted_classes = sorted(all_classes - exclude_set)

        class_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_classes)}
    else:
        class_mapping = None

    # Process labels
    stats = {
        "total_files": 0,
        "files_with_labels": 0,
        "files_without_labels": 0,
        "total_instances_before": 0,
        "total_instances_after": 0,
    }

    for label_file in label_dir.glob("*.txt"):
        stats["total_files"] += 1

        with open(label_file, "r") as f:
            lines = f.readlines()

        stats["total_instances_before"] += len([l for l in lines if l.strip()])

        # Filter and optionally remap
        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])

            # Check if should keep
            if class_id in exclude_set:
                continue
            if keep_set is not None and class_id not in keep_set:
                continue

            # Remap if needed
            if class_mapping is not None:
                new_class_id = class_mapping[class_id]
            else:
                new_class_id = class_id

            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
            filtered_lines.append(new_line)

        # Save if not empty
        if filtered_lines:
            output_file = output_dir / label_file.name
            with open(output_file, "w") as f:
                f.writelines(filtered_lines)
            stats["files_with_labels"] += 1
            stats["total_instances_after"] += len(filtered_lines)
        else:
            stats["files_without_labels"] += 1

    print(f"\n✅ Filtered labels saved to: {output_dir}")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Files with labels: {stats['files_with_labels']}")
    print(f"   Files without labels: {stats['files_without_labels']}")
    print(f"   Instances before: {stats['total_instances_before']}")
    print(f"   Instances after: {stats['total_instances_after']}")

    if class_mapping:
        print("\nClass mapping:")
        for old_id, new_id in sorted(class_mapping.items()):
            print(f"  {old_id} → {new_id}")

    return stats


def remap_class_ids(
    label_dir: Union[str, Path],
    output_dir: Union[str, Path],
    class_mapping: Dict[int, int],
) -> None:
    """
    Remap class IDs according to provided mapping.

    Args:
        label_dir: Input label directory
        output_dir: Output label directory
        class_mapping: Dictionary mapping old_class_id -> new_class_id
    """
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            old_class_id = int(parts[0])
            new_class_id = class_mapping.get(old_class_id, old_class_id)

            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
            new_lines.append(new_line)

        if new_lines:
            output_file = output_dir / label_file.name
            with open(output_file, "w") as f:
                f.writelines(new_lines)

    print(f"✅ Remapped labels saved to: {output_dir}")
