"""
Empty Labels Filter - Centralized implementation

Filters out cropped images that have no corresponding KL grade labels.
This is the core implementation used by all filtering scripts.
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List


def filter_empty_labels(input_dir: Path) -> Dict:
    """
    Filter crops without labels - core implementation.

    Moves cropped images without KL grade labels to separate folder.

    Args:
        input_dir: Directory containing images/ and labels/

    Returns:
        Dictionary with filtering statistics:
            - total_images: Total number of images processed
            - moved_count: Number of images moved to no-labels folder
            - remaining_count: Number of images with labels
            - no_label_files: List of moved file stems
    """
    img_dir = input_dir / "images"
    label_dir = input_dir / "labels"

    no_label_img_dir = input_dir / "images-no-labels"
    no_label_label_dir = input_dir / "labels-no-labels"

    no_label_img_dir.mkdir(exist_ok=True)
    no_label_label_dir.mkdir(exist_ok=True)

    if not img_dir.exists() or not label_dir.exists():
        raise ValueError(f"Missing directories: {img_dir} or {label_dir}")

    # Get all images
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    moved_count = 0
    no_label_files = []

    for img_file in images:
        stem = img_file.stem
        label_file = label_dir / f"{stem}.txt"

        # Check if label is empty or missing
        has_label = False
        if label_file.exists():
            with open(label_file, "r") as f:
                content = f.read().strip()
                has_label = len(content) > 0

        if not has_label:
            # Move image to no-labels folder
            dest_img_path = no_label_img_dir / img_file.name
            shutil.move(str(img_file), str(dest_img_path))

            # Move empty label file if exists
            if label_file.exists():
                dest_label_path = no_label_label_dir / label_file.name
                shutil.move(str(label_file), str(dest_label_path))

            no_label_files.append(stem)
            moved_count += 1

    # Save log
    if no_label_files:
        log_path = input_dir / "no_label_files.json"
        with open(log_path, "w") as f:
            json.dump(no_label_files, f, indent=2)

    remaining_count = len(images) - moved_count

    return {
        "total_images": len(images),
        "moved_count": moved_count,
        "remaining_count": remaining_count,
        "no_label_files": no_label_files,
    }
