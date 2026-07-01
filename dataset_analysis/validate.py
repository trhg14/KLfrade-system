"""
Dataset Validation Tool

Comprehensive validation for YOLO format datasets:
- Label format validation (correct number of values, valid ranges)
- Bounding box validation (0-1 range, w/h > 0, valid xyxy conversion)
- Image-label pairing check (missing labels, orphaned labels)
- File integrity checks

Usage:
    python tools/check_dataset/validate_dataset.py --img_dir dataset/images --label_dir dataset/labels
"""

import sys
from pathlib import Path
import argparse
import math
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


MIN_WH = 0.001  # Minimum width/height threshold


def is_valid_number(x: float) -> bool:
    """Check if number is valid (not NaN or Inf)."""
    return not (math.isnan(x) or math.isinf(x))


def validate_label_file(label_path: Path, verbose: bool = False) -> dict:
    """
    Validate a single label file.

    Returns:
        dict with validation results: {
            'valid': bool,
            'errors': list of str,
            'warnings': list of str,
            'num_boxes': int
        }
    """
    errors = []
    warnings = []
    num_boxes = 0

    if not label_path.exists():
        return {
            "valid": False,
            "errors": ["File not found"],
            "warnings": [],
            "num_boxes": 0,
        }

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) == 0:
            warnings.append("Empty file")
            return {"valid": True, "errors": [], "warnings": warnings, "num_boxes": 0}

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Check format
            if len(parts) < 5:
                errors.append(
                    f"Line {line_num}: Invalid format (expected 5 values, got {len(parts)})"
                )
                continue

            try:
                class_id = float(parts[0])
                x, y, w, h = map(float, parts[1:5])

                # Validate class_id
                if not (class_id == int(class_id) and class_id >= 0):
                    errors.append(f"Line {line_num}: Invalid class_id ({class_id})")

                # Check for NaN/Inf
                for val, name in zip([x, y, w, h], ["x", "y", "w", "h"]):
                    if not is_valid_number(val):
                        errors.append(f"Line {line_num}: {name} is NaN/Inf ({val})")

                # Check valid ranges
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    errors.append(
                        f"Line {line_num}: Values out of range or w/h <= 0 "
                        f"(x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f})"
                    )

                # Check minimum size
                if w < MIN_WH or h < MIN_WH:
                    warnings.append(
                        f"Line {line_num}: Very small box (w={w:.6f}, h={h:.6f})"
                    )

                # Validate xyxy conversion
                x_center_norm = x
                y_center_norm = y
                w_norm = w
                h_norm = h

                x1 = x_center_norm - w_norm / 2
                y1 = y_center_norm - h_norm / 2
                x2 = x_center_norm + w_norm / 2
                y2 = y_center_norm + h_norm / 2

                # Check xyxy validity
                if not (
                    0.0 <= x1 <= 1.0
                    and 0.0 <= y1 <= 1.0
                    and 0.0 <= x2 <= 1.0
                    and 0.0 <= y2 <= 1.0
                ):
                    errors.append(
                        f"Line {line_num}: xyxy out of range "
                        f"({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f})"
                    )

                if x2 <= x1 or y2 <= y1:
                    errors.append(
                        f"Line {line_num}: Invalid xyxy (x2 <= x1 or y2 <= y1)"
                    )

                num_boxes += 1

            except ValueError as e:
                errors.append(f"Line {line_num}: Cannot parse values - {str(e)}")
            except Exception as e:
                errors.append(f"Line {line_num}: Unexpected error - {str(e)}")

    except Exception as e:
        errors.append(f"File read error: {str(e)}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "num_boxes": num_boxes,
    }


def check_image_label_pairing(img_dir: Path, label_dir: Path) -> dict:
    """
    Check image-label pairing.

    Returns:
        dict with pairing info: {
            'images_without_labels': list,
            'labels_without_images': list,
            'paired': int
        }
    """
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Get image stems
    image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]
    image_stems = {f.stem for f in image_files}

    # Get label stems
    label_files = list(label_dir.glob("*.txt"))
    label_stems = {f.stem for f in label_files}

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)
    paired = len(image_stems & label_stems)

    return {
        "images_without_labels": images_without_labels,
        "labels_without_images": labels_without_images,
        "paired": paired,
        "total_images": len(image_stems),
        "total_labels": len(label_stems),
    }


def validate_dataset(img_dir: Path, label_dir: Path, verbose: bool = False):
    """Run comprehensive dataset validation."""
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    print(f"\nImage dir: {img_dir}")
    print(f"Label dir: {label_dir}")

    # Check pairing
    print("\n" + "=" * 80)
    print("CHECKING IMAGE-LABEL PAIRING")
    print("=" * 80)

    pairing = check_image_label_pairing(img_dir, label_dir)

    print(f"\n✅ Paired: {pairing['paired']} files")
    print(f"📸 Total images: {pairing['total_images']}")
    print(f"🏷️  Total labels: {pairing['total_labels']}")

    if pairing["images_without_labels"]:
        print(f"\n⚠️  Images without labels: {len(pairing['images_without_labels'])}")
        if verbose or len(pairing["images_without_labels"]) <= 10:
            for stem in pairing["images_without_labels"][:10]:
                print(f"    {stem}")
            if len(pairing["images_without_labels"]) > 10:
                print(f"    ... and {len(pairing['images_without_labels']) - 10} more")

    if pairing["labels_without_images"]:
        print(f"\n⚠️  Labels without images: {len(pairing['labels_without_images'])}")
        if verbose or len(pairing["labels_without_images"]) <= 10:
            for stem in pairing["labels_without_images"][:10]:
                print(f"    {stem}")
            if len(pairing["labels_without_images"]) > 10:
                print(f"    ... and {len(pairing['labels_without_images']) - 10} more")

    # Validate labels
    print("\n" + "=" * 80)
    print("VALIDATING LABEL FILES")
    print("=" * 80)

    label_files = list(label_dir.glob("*.txt"))

    all_errors = []
    all_warnings = []
    valid_count = 0
    invalid_count = 0
    total_boxes = 0

    for label_file in tqdm(label_files, desc="Validating"):
        result = validate_label_file(label_file, verbose=verbose)

        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
            for error in result["errors"]:
                all_errors.append(f"{label_file.name}: {error}")

        for warning in result["warnings"]:
            all_warnings.append(f"{label_file.name}: {warning}")

        total_boxes += result["num_boxes"]

    # Print results
    print(f"\n📊 Validation Results:")
    print(f"  Valid files: {valid_count}")
    print(f"  Invalid files: {invalid_count}")
    print(f"  Total boxes: {total_boxes}")

    if all_errors:
        print(f"\n❌ Errors found: {len(all_errors)}")
        for error in all_errors[:20]:
            print(f"    {error}")
        if len(all_errors) > 20:
            print(f"    ... and {len(all_errors) - 20} more errors")
    else:
        print(f"\n✅ No errors found!")

    if all_warnings:
        print(f"\n⚠️  Warnings: {len(all_warnings)}")
        if verbose:
            for warning in all_warnings[:20]:
                print(f"    {warning}")
            if len(all_warnings) > 20:
                print(f"    ... and {len(all_warnings) - 20} more warnings")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_good = (
        invalid_count == 0
        and len(pairing["images_without_labels"]) == 0
        and len(pairing["labels_without_images"]) == 0
    )

    if all_good:
        print("✅ Dataset validation passed!")
        print("   - All labels are valid")
        print("   - All images have corresponding labels")
        print("   - No orphaned labels")
    else:
        print("⚠️  Issues found:")
        if invalid_count > 0:
            print(f"   - {invalid_count} invalid label files")
        if pairing["images_without_labels"]:
            print(f"   - {len(pairing['images_without_labels'])} images without labels")
        if pairing["labels_without_images"]:
            print(f"   - {len(pairing['labels_without_images'])} labels without images")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument("--img_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--label_dir", type=str, required=True, help="Label directory")
    parser.add_argument(
        "--verbose", action="store_true", help="Show all errors/warnings"
    )

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)

    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return

    if not label_dir.exists():
        print(f"❌ Label directory not found: {label_dir}")
        return

    validate_dataset(img_dir, label_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
