#!/usr/bin/env python3
"""
Filter Dataset by Removing Specific Classes

Creates filtered datasets by removing images with specified class labels
and remapping remaining class IDs.

This is a CLI wrapper for the centralized filter implementation.

Usage:
    # Remove KL0 (class 0) from 5-class dataset
    python scripts/data_preparation/filter_kl0.py --input processed/knee --output processed/knee_4_class

    # Remove KL0-a and KL0-b (classes 0,1) from 10-class dataset
    python scripts/data_preparation/filter_kl0.py --input processed/knee_10_class --output processed/knee_8_class --num_classes 10

    # Remove specific classes
    python scripts/data_preparation/filter_kl0.py --input dataset --output dataset_filtered --classes 0 3 4
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.filters import filter_classes


def main():
    parser = argparse.ArgumentParser(description="Filter specific classes from dataset")
    parser.add_argument(
        "--input", type=str, required=True, help="Input dataset directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output dataset directory"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        choices=[5, 10],
        help="Original number of classes (5 or 10, used for default class removal)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Specific class IDs to remove (e.g., --classes 0 1 2). If not specified, defaults based on num_classes.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 80)
    print("FILTERING CLASSES")
    print("=" * 80)
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")

    # Determine classes to remove
    if args.classes is not None:
        classes_to_remove = args.classes
        print(f"Mode: Custom - removing classes {classes_to_remove}")
    else:
        # Default based on num_classes
        if args.num_classes == 5:
            classes_to_remove = None  # Will default to [0]
            print(f"Mode: {args.num_classes}-class dataset")
            print("Removing class 0 (KL0)")
            print("Remapping: 1→0, 2→1, 3→2, 4→3")
            print("Output classes: 4 (0-3)")
        else:  # 10 classes
            classes_to_remove = None  # Will default to [0, 1]
            print(f"Mode: {args.num_classes}-class dataset")
            print("Removing classes 0, 1 (KL0-a, KL0-b)")
            print("Remapping: 2→0, 3→1, ..., 9→7")
            print("Output classes: 8 (0-7)")

    # Call centralized implementation
    stats = filter_classes(
        input_dir,
        output_dir,
        classes_to_remove=classes_to_remove,
        num_classes=args.num_classes,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)
    print(f"\nClasses removed: {stats['classes_removed']}")
    print(f"Output classes: {stats['output_classes']}")
    print(f"\nTotal images: {stats['total_images']}")
    print(
        f"✅ Kept: {stats['kept_images']} ({stats['kept_images']/stats['total_images']*100:.1f}%)"
    )
    print(
        f"🗑️  Filtered: {stats['filtered_images']} ({stats['filtered_images']/stats['total_images']*100:.1f}%)"
    )

    print(f"\nBoxes:")
    print(f"  Original: {stats['original_boxes']}")
    print(
        f"  Kept: {stats['kept_boxes']} ({stats['kept_boxes']/stats['original_boxes']*100:.1f}%)"
    )
    print(
        f"  Filtered: {stats['filtered_boxes']} ({stats['filtered_boxes']/stats['original_boxes']*100:.1f}%)"
    )

    print(f"\n💾 Saved stats: {output_dir}/filter_classes_stats.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
