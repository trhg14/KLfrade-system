"""
Class Distribution Sampling and Balancing

Functions for counting class distributions, calculating balance targets,
and balancing datasets via oversampling with augmentation.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from tqdm import tqdm

from src.data.image_io import read_image, write_image


def count_class_distribution(
    label_dir: Union[str, Path],
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[int, int]:
    """
    Count instances per class in YOLO label directory.

    Args:
        label_dir: Directory containing YOLO label files (.txt)
        num_classes: Number of classes (optional, auto-detected if None)
        class_names: List of class names for display (optional)

    Returns:
        Dictionary mapping class_id to count
    """
    label_dir = Path(label_dir)
    class_counts = defaultdict(int)

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    # Convert to regular dict and ensure all classes present
    if num_classes is not None:
        class_counts = {i: class_counts.get(i, 0) for i in range(num_classes)}
    else:
        class_counts = dict(class_counts)

    return class_counts


def calculate_balance_targets(
    class_counts: Dict[int, int],
    strategy: str = "max",
    custom_target: Optional[int] = None,
) -> Dict[int, int]:
    """
    Calculate target counts for each class based on balancing strategy.

    Args:
        class_counts: Current class distribution
        strategy: Balancing strategy ('max', 'median', 'mean', 'custom')
        custom_target: Custom target count (required if strategy='custom')

    Returns:
        Dictionary mapping class_id to target count
    """
    if strategy == "max":
        target = max(class_counts.values())
    elif strategy == "median":
        sorted_counts = sorted(class_counts.values())
        n = len(sorted_counts)
        target = (
            sorted_counts[n // 2]
            if n % 2
            else (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) // 2
        )
    elif strategy == "mean":
        target = sum(class_counts.values()) // len(class_counts)
    elif strategy == "custom":
        if custom_target is None:
            raise ValueError("custom_target must be provided when strategy='custom'")
        target = custom_target
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {class_id: target for class_id in class_counts.keys()}


def find_images_by_class(
    image_dir: Union[str, Path], label_dir: Union[str, Path], class_id: int
) -> List[Tuple[Path, Path]]:
    """
    Find all (image, label) pairs containing specific class.

    Args:
        image_dir: Directory containing images
        label_dir: Directory containing labels
        class_id: Class ID to search for

    Returns:
        List of (image_path, label_path) tuples
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    images_with_class = []

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
            if any(int(line.split()[0]) == class_id for line in lines if line.strip()):
                # Find corresponding image
                base_name = label_file.stem
                for ext in [".jpg", ".png", ".jpeg"]:
                    img_path = image_dir / f"{base_name}{ext}"
                    if img_path.exists():
                        images_with_class.append((img_path, label_file))
                        break

    return images_with_class


def balance_dataset(
    image_dir: Union[str, Path],
    label_dir: Union[str, Path],
    output_img_dir: Union[str, Path],
    output_label_dir: Union[str, Path],
    strategy: str = "flip",
    target_strategy: str = "max",
    custom_target: Optional[int] = None,
    num_classes: int = 5,
    aux_label_dirs: Optional[List[Union[str, Path]]] = None,
    output_aux_label_dirs: Optional[List[Union[str, Path]]] = None,
    save_report: bool = True,
) -> Dict[str, any]:
    """
    Balance dataset by augmenting minority classes.

    Args:
        image_dir: Input image directory
        label_dir: Input label directory
        output_img_dir: Output image directory
        output_label_dir: Output label directory
        strategy: Augmentation strategy ('flip', 'none')
        target_strategy: Target calculation strategy ('max', 'median', 'mean', 'custom')
        custom_target: Custom target count (if target_strategy='custom')
        num_classes: Number of classes
        aux_label_dirs: List of auxiliary label directories (e.g., knee boxes)
        output_aux_label_dirs: List of output auxiliary label directories
        save_report: Whether to save balance report

    Returns:
        Dictionary with balancing statistics
    """
    from .augmentor import flip_horizontal

    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_img_dir = Path(output_img_dir)
    output_label_dir = Path(output_label_dir)

    # Create output directories
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    if aux_label_dirs and output_aux_label_dirs:
        for out_aux_dir in output_aux_label_dirs:
            Path(out_aux_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DATASET BALANCING")
    print("=" * 60)

    # Step 1: Copy all original data
    print("\n📂 Copying original data...")
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    for img_file in tqdm(image_files, desc="Copying images"):
        shutil.copy2(img_file, output_img_dir / img_file.name)

    for label_file in label_dir.glob("*.txt"):
        shutil.copy2(label_file, output_label_dir / label_file.name)

    # Copy auxiliary labels
    if aux_label_dirs and output_aux_label_dirs:
        for aux_dir, out_aux_dir in zip(aux_label_dirs, output_aux_label_dirs):
            for label_file in Path(aux_dir).glob("*.txt"):
                shutil.copy2(label_file, Path(out_aux_dir) / label_file.name)

    # Step 2: Count current distribution
    print("\n📊 Analyzing class distribution...")
    original_counts = count_class_distribution(label_dir, num_classes)

    print("\nOriginal distribution:")
    for class_id, count in sorted(original_counts.items()):
        print(f"  Class {class_id}: {count:>5} instances")

    # Step 3: Calculate targets
    targets = calculate_balance_targets(original_counts, target_strategy, custom_target)
    print(f"\nTarget strategy: {target_strategy}")
    print(f"Target count: {max(targets.values())} instances per class")

    if strategy == "none":
        print("\n🚫 Skipping augmentation (strategy='none')")
        final_counts = count_class_distribution(output_label_dir, num_classes)
        return {
            "original_counts": original_counts,
            "final_counts": final_counts,
            "targets": targets,
            "augmented": {},
        }

    # Step 4: Balance via augmentation
    print(f"\n🔄 Balancing via {strategy} augmentation...")

    augmented_counts = {i: 0 for i in range(num_classes)}

    for class_id in range(num_classes):
        needed = targets[class_id] - original_counts[class_id]

        if needed <= 0:
            print(f"  Class {class_id}: Already balanced")
            continue

        print(f"  Class {class_id}: Need {needed} more instances")

        # Find images with this class
        images_with_class = find_images_by_class(
            output_img_dir, output_label_dir, class_id
        )

        if not images_with_class:
            print(f"    ⚠️  No images found for class {class_id}")
            continue

        # Augment until balanced
        random.shuffle(images_with_class)
        idx = 0
        augmented = 0

        while (
            augmented < needed and idx < len(images_with_class) * 100
        ):  # Max 100 rounds
            img_path, label_path = images_with_class[idx % len(images_with_class)]
            idx += 1

            # Read image and labels
            image = read_image(img_path)
            if image is None:
                continue
            with open(label_path, "r") as f:
                all_labels = f.readlines()

            # Filter to only this class
            class_labels = [
                line
                for line in all_labels
                if line.strip() and int(line.split()[0]) == class_id
            ]

            if not class_labels:
                continue

            # Apply augmentation
            if strategy == "flip":
                aug_image, aug_labels = flip_horizontal(image, class_labels)
            else:
                continue

            # Save augmented image
            base_name = img_path.stem
            aug_img_name = f"{base_name}_flip_{augmented}{img_path.suffix}"
            aug_img_path = output_img_dir / aug_img_name
            write_image(aug_img_path, aug_image)

            # Save augmented labels
            aug_label_name = f"{base_name}_flip_{augmented}.txt"
            aug_label_path = output_label_dir / aug_label_name
            with open(aug_label_path, "w") as f:
                f.write("".join(aug_labels))

            # Copy auxiliary labels if needed (same augmentation applies)
            if aux_label_dirs and output_aux_label_dirs:
                for aux_dir, out_aux_dir in zip(aux_label_dirs, output_aux_label_dirs):
                    aux_label_file = Path(aux_dir) / (base_name + ".txt")
                    if aux_label_file.exists():
                        with open(aux_label_file, "r") as f:
                            aux_labels_content = f.readlines()

                        if strategy == "flip":
                            _, aug_aux_labels = flip_horizontal(
                                image, aux_labels_content
                            )

                        aug_aux_path = Path(out_aux_dir) / aug_label_name
                        with open(aug_aux_path, "w") as f:
                            f.write("".join(aug_aux_labels))

            augmented += len(class_labels)
            augmented_counts[class_id] += len(class_labels)

        print(f"    ✅ Added {augmented_counts[class_id]} instances")

    # Step 5: Final statistics
    final_counts = count_class_distribution(output_label_dir, num_classes)

    print("\nFinal distribution:")
    for class_id, count in sorted(final_counts.items()):
        print(
            f"  Class {class_id}: {count:>5} instances ({augmented_counts[class_id]:>4} augmented)"
        )

    # Save report
    stats = {
        "original_counts": original_counts,
        "final_counts": final_counts,
        "targets": targets,
        "augmented": augmented_counts,
    }

    if save_report:
        report_path = output_img_dir.parent / "balance_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("DATASET BALANCING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Strategy: {strategy}\n")
            f.write(f"Target strategy: {target_strategy}\n\n")

            f.write("Original Distribution:\n")
            for class_id, count in sorted(original_counts.items()):
                f.write(f"  Class {class_id}: {count:>5} instances\n")

            f.write("\nFinal Distribution:\n")
            for class_id, count in sorted(final_counts.items()):
                aug = augmented_counts[class_id]
                f.write(
                    f"  Class {class_id}: {count:>5} instances ({aug:>4} augmented)\n"
                )

        print(f"\n✅ Report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("BALANCING COMPLETE")
    print("=" * 60)

    return stats
