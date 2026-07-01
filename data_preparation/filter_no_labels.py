#!/usr/bin/env python3
"""
Filter Crops Without Labels

Removes cropped knee images without KL grade labels.
This is a CLI wrapper for the centralized filter implementation.

Usage:
    python scripts/data_preparation/filter_no_labels.py --input processed/knee
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.filters import filter_empty_labels


def main():
    parser = argparse.ArgumentParser(description="Filter crops without labels")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory (e.g., processed/knee)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)

    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 80)
    print("FILTERING CROPS WITHOUT LABELS")
    print("=" * 80)
    print(f"\nInput: {input_dir}\n")

    try:
        # Call centralized implementation
        stats = filter_empty_labels(input_dir)

        # Print summary
        print("\n" + "=" * 80)
        print("FILTERING SUMMARY")
        print("=" * 80)
        print(f"\nTotal crops: {stats['total_images']}")
        print(f"🗂️  Moved to no-labels: {stats['moved_count']}")
        print(f"📊 Remaining with labels: {stats['remaining_count']}")

        if stats["no_label_files"]:
            print(f"\n📝 Saved no-label files list: {input_dir}/no_label_files.json")

        print("=" * 80)

    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
