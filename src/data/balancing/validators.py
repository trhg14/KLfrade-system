"""
Dataset Validators

Functions for validating dataset integrity, checking image-label pairs,
and generating dataset statistics.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set


def check_image_label_pairs(
    image_dir: Union[str, Path],
    label_dir: Union[str, Path],
    move_unmatched: bool = False,
    unmatched_dir: Optional[Union[str, Path]] = None,
    image_extensions: Optional[Set[str]] = None,
) -> Dict[str, any]:
    """
    Check correspondence between images and labels, optionally move unmatched files.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        move_unmatched: Whether to move unmatched files
        unmatched_dir: Directory to move unmatched files (required if move_unmatched=True)
        image_extensions: Set of valid image extensions (default: {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'})

    Returns:
        Dictionary with validation results
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    if image_extensions is None:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    if move_unmatched and unmatched_dir is None:
        raise ValueError("unmatched_dir must be provided when move_unmatched=True")

    if move_unmatched:
        unmatched_dir = Path(unmatched_dir)
        unmatched_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("VALIDATING IMAGE-LABEL PAIRS")
    print("=" * 60)

    # Get image files (with extensions)
    img_files = {}
    for f in image_dir.iterdir():
        if f.suffix.lower() in image_extensions:
            img_files[f.stem] = f.suffix

    # Get label files (without extensions)
    lab_files = {f.stem for f in label_dir.glob("*.txt")}

    # Find mismatches
    img_not_in_lab = set(img_files.keys()) - lab_files
    lab_not_in_img = lab_files - set(img_files.keys())

    stats = {
        "total_images": len(img_files),
        "total_labels": len(lab_files),
        "matched_pairs": len(img_files) - len(img_not_in_lab),
        "images_without_labels": len(img_not_in_lab),
        "labels_without_images": len(lab_not_in_img),
        "unmatched_images": list(img_not_in_lab),
        "unmatched_labels": list(lab_not_in_img),
    }

    print(f"\nImages: {stats['total_images']}")
    print(f"Labels: {stats['total_labels']}")
    print(f"Matched pairs: {stats['matched_pairs']}")
    print(f"Images without labels: {stats['images_without_labels']}")
    print(f"Labels without images: {stats['labels_without_images']}")

    # Move unmatched files if requested
    if move_unmatched:
        print(f"\n📁 Moving unmatched files to: {unmatched_dir}")

        # Move images without labels
        for img_stem in img_not_in_lab:
            src = image_dir / (img_stem + img_files[img_stem])
            dst = unmatched_dir / (img_stem + img_files[img_stem])
            shutil.move(str(src), str(dst))
            print(f"  Moved: {src.name}")

        # Move labels without images
        for lab_stem in lab_not_in_img:
            src = label_dir / (lab_stem + ".txt")
            dst = unmatched_dir / (lab_stem + ".txt")
            shutil.move(str(src), str(dst))
            print(f"  Moved: {src.name}")

    if img_not_in_lab or lab_not_in_img:
        print("\n⚠️  Dataset has unmatched files")
    else:
        print("\n✅ All files matched")

    print("=" * 60)

    return stats


def validate_yolo_labels(
    label_dir: Union[str, Path], num_classes: int, fix_errors: bool = False
) -> Dict[str, any]:
    """
    Validate YOLO label format and class IDs.

    Args:
        label_dir: Directory containing YOLO label files
        num_classes: Expected number of classes
        fix_errors: Whether to attempt fixing errors (remove invalid lines)

    Returns:
        Dictionary with validation results
    """
    label_dir = Path(label_dir)

    print("\n" + "=" * 60)
    print("VALIDATING YOLO LABELS")
    print("=" * 60)

    stats = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "empty_files": 0,
        "total_labels": 0,
        "invalid_labels": 0,
        "class_distribution": {},
        "errors": [],
    }

    for label_file in label_dir.glob("*.txt"):
        stats["total_files"] += 1

        with open(label_file, "r") as f:
            lines = f.readlines()

        if not lines or all(not line.strip() for line in lines):
            stats["empty_files"] += 1
            continue

        valid_lines = []
        file_has_error = False

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            stats["total_labels"] += 1

            # Validate format
            parts = line.split()
            if len(parts) != 5:
                stats["invalid_labels"] += 1
                file_has_error = True
                stats["errors"].append(
                    f"{label_file.name}:{line_num} - Invalid format (expected 5 values, got {len(parts)})"
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Validate class ID
                if class_id < 0 or class_id >= num_classes:
                    stats["invalid_labels"] += 1
                    file_has_error = True
                    stats["errors"].append(
                        f"{label_file.name}:{line_num} - Invalid class_id {class_id} (expected 0-{num_classes-1})"
                    )
                    continue

                # Validate coordinates (should be 0-1)
                if not (
                    0 <= x_center <= 1
                    and 0 <= y_center <= 1
                    and 0 <= width <= 1
                    and 0 <= height <= 1
                ):
                    stats["invalid_labels"] += 1
                    file_has_error = True
                    stats["errors"].append(
                        f"{label_file.name}:{line_num} - Coordinates out of range [0,1]"
                    )
                    continue

                # Count class distribution
                stats["class_distribution"][class_id] = (
                    stats["class_distribution"].get(class_id, 0) + 1
                )
                valid_lines.append(line + "\n")

            except ValueError as e:
                stats["invalid_labels"] += 1
                file_has_error = True
                stats["errors"].append(f"{label_file.name}:{line_num} - {str(e)}")
                continue

        if file_has_error:
            stats["invalid_files"] += 1

            # Fix if requested
            if fix_errors and valid_lines:
                with open(label_file, "w") as f:
                    f.writelines(valid_lines)
                print(
                    f"  Fixed: {label_file.name} (removed {len(lines) - len(valid_lines)} invalid lines)"
                )
        else:
            stats["valid_files"] += 1

    print(f"\nTotal files: {stats['total_files']}")
    print(f"Valid files: {stats['valid_files']}")
    print(f"Invalid files: {stats['invalid_files']}")
    print(f"Empty files: {stats['empty_files']}")
    print(f"Total labels: {stats['total_labels']}")
    print(f"Invalid labels: {stats['invalid_labels']}")

    if stats["errors"]:
        print(f"\n⚠️  Found {len(stats['errors'])} errors:")
        for error in stats["errors"][:10]:  # Show first 10
            print(f"  {error}")
        if len(stats["errors"]) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more")
    else:
        print("\n✅ All labels valid")

    print("\nClass distribution:")
    for class_id in sorted(stats["class_distribution"].keys()):
        count = stats["class_distribution"][class_id]
        print(f"  Class {class_id}: {count} instances")

    print("=" * 60)

    return stats


def get_dataset_stats(
    image_dir: Union[str, Path],
    label_dir: Union[str, Path],
    num_classes: Optional[int] = None,
) -> Dict[str, any]:
    """
    Get comprehensive dataset statistics.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        num_classes: Number of classes (optional)

    Returns:
        Dictionary with dataset statistics
    """
    from .sampler import count_class_distribution

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Count files
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    label_files = list(label_dir.glob("*.txt"))

    # Count class distribution
    class_counts = count_class_distribution(label_dir, num_classes)

    # Calculate statistics
    total_instances = sum(class_counts.values())

    stats = {
        "image_count": len(image_files),
        "label_count": len(label_files),
        "class_distribution": class_counts,
        "total_instances": total_instances,
        "num_classes": len(class_counts),
    }

    print(f"\nImages: {stats['image_count']}")
    print(f"Labels: {stats['label_count']}")
    print(f"Total instances: {stats['total_instances']}")
    print(f"Number of classes: {stats['num_classes']}")

    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"  Class {class_id}: {count:>5} instances ({percentage:>5.2f}%)")

    print("=" * 60)

    return stats
