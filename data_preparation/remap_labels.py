"""
Remap filtered dataset labels from original class IDs to new continuous IDs (0-6).

This script remaps class IDs in the filtered dataset labels from the original
labels_10_class IDs {0,2,4,5,6,7,8} to continuous IDs {0,1,2,3,4,5,6} for easier training.

Original mapping (labels_10_class):
  0: KL0-a, 1: KL0-b (removed), 2: KL1-a, 3: KL1-b (removed),
  4: KL2-a, 5: KL2-b, 6: KL3-a, 7: KL3-b, 8: KL4-a, 9: KL4-b (removed)

New filtered mapping:
  0: KL0-a, 1: KL1-a, 2: KL2-a, 3: KL2-b, 4: KL3-a, 5: KL3-b, 6: KL4-a
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CLASS_REMAP_FILTERED
from tqdm import tqdm


def remap_labels(label_dir: Path, output_dir: Path = None):
    """
    Remap class IDs in label files.

    Args:
        label_dir: Directory containing label files to remap
        output_dir: Output directory (if None, overwrites original files)
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = label_dir

    label_files = list(label_dir.glob("*.txt"))

    remapped_count = 0
    total_boxes = 0

    for label_file in tqdm(label_files, desc="Remapping labels"):
        new_lines = []

        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    old_class_id = int(float(parts[0]))
                    total_boxes += 1

                    # Remap class ID
                    new_class_id = CLASS_REMAP_FILTERED.get(old_class_id)

                    if new_class_id is not None:
                        remapped_count += 1
                        # Replace class ID
                        parts[0] = str(new_class_id)
                        new_lines.append(" ".join(parts))

        # Write remapped labels
        output_file = output_dir / label_file.name
        with open(output_file, "w") as f:
            for line in new_lines:
                f.write(line + "\n")

    return {
        "total_files": len(label_files),
        "total_boxes": total_boxes,
        "remapped_boxes": remapped_count,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remap filtered dataset class IDs")
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing labels to remap",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: overwrite original)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run - show mapping without writing files",
    )

    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    print("=" * 70)
    print("Remap Filtered Dataset Class IDs")
    print("=" * 70)

    print(f"\nLabel directory: {label_dir}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    else:
        print("Output directory: [Overwrite original]")

    print(f"\nClass ID mapping:")
    for old_id, new_id in sorted(CLASS_REMAP_FILTERED.items()):
        if new_id is not None:
            print(f"  {old_id} -> {new_id}")
        else:
            print(f"  {old_id} -> REMOVED")

    if args.dry_run:
        print("\n⚠️  DRY RUN - No files will be modified")
        exit(0)

    print("\n🔄 Remapping labels...")
    stats = remap_labels(label_dir, output_dir)

    print("\n" + "=" * 70)
    print("Remapping Summary")
    print("=" * 70)
    print(f"Files processed: {stats['total_files']}")
    print(f"Total boxes: {stats['total_boxes']}")
    print(f"Remapped boxes: {stats['remapped_boxes']}")

    print("\n✅ Remapping completed successfully!")
    print("=" * 70)


# Example usage:
# python remap_filtered_labels.py --label_dir dataset/dataset_filtered/labels --output_dir dataset/dataset_filtered/labels_remapped
# python remap_filtered_labels.py --label_dir dataset/dataset_filtered/labels  # Overwrite original
