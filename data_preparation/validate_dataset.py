"""
Validate Dataset Script

Validates dataset integrity by checking image-label pairs,
YOLO label format, and generating comprehensive statistics.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.balancing import (
    check_image_label_pairs,
    validate_yolo_labels,
    get_dataset_stats,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset integrity and format"
    )

    # Required arguments
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--label-dir", type=str, required=True, help="Label directory")

    # Optional arguments
    parser.add_argument(
        "--num-classes", type=int, default=5, help="Number of classes (default: 5)"
    )
    parser.add_argument(
        "--move-unmatched",
        action="store_true",
        help="Move unmatched files to unmatched directory",
    )
    parser.add_argument(
        "--unmatched-dir",
        type=str,
        default=None,
        help="Directory for unmatched files (default: dataset_dir/unmatched)",
    )
    parser.add_argument(
        "--fix-labels",
        action="store_true",
        help="Attempt to fix invalid labels (remove invalid lines)",
    )
    parser.add_argument(
        "--skip-pairs", action="store_true", help="Skip image-label pair validation"
    )
    parser.add_argument(
        "--skip-format", action="store_true", help="Skip YOLO format validation"
    )
    parser.add_argument(
        "--skip-stats", action="store_true", help="Skip dataset statistics"
    )

    args = parser.parse_args()

    # Convert paths
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)

    # Validate inputs
    if not image_dir.exists():
        print(f"❌ Error: Image directory not found: {image_dir}")
        sys.exit(1)

    if not label_dir.exists():
        print(f"❌ Error: Label directory not found: {label_dir}")
        sys.exit(1)

    # Setup unmatched directory
    if args.move_unmatched and args.unmatched_dir is None:
        args.unmatched_dir = image_dir.parent / "unmatched"

    all_valid = True

    # Step 1: Check image-label pairs
    if not args.skip_pairs:
        pair_stats = check_image_label_pairs(
            image_dir,
            label_dir,
            move_unmatched=args.move_unmatched,
            unmatched_dir=args.unmatched_dir,
        )

        if (
            pair_stats["images_without_labels"] > 0
            or pair_stats["labels_without_images"] > 0
        ):
            all_valid = False

    # Step 2: Validate YOLO label format
    if not args.skip_format:
        format_stats = validate_yolo_labels(
            label_dir, args.num_classes, fix_errors=args.fix_labels
        )

        if format_stats["invalid_labels"] > 0:
            all_valid = False

    # Step 3: Dataset statistics
    if not args.skip_stats:
        get_dataset_stats(image_dir, label_dir, args.num_classes)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if all_valid:
        print("✅ Dataset validation passed")
        print("\nDataset is ready for:")
        print("  - Preprocessing")
        print("  - Balancing")
        print("  - Train/val/test splitting")
        print("  - Training")
        sys.exit(0)
    else:
        print("⚠️  Dataset validation found issues")
        if args.move_unmatched or args.fix_labels:
            print("\nIssues have been addressed. Re-run validation to confirm.")
        else:
            print("\nRecommendations:")
            print("  - Use --move-unmatched to move unmatched files")
            print("  - Use --fix-labels to attempt fixing invalid labels")
        sys.exit(1)


if __name__ == "__main__":
    main()
