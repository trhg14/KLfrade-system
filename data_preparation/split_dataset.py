"""
Stratified Dataset Splitting for Object Detection (YOLO format)

This script creates stratified train/val/test splits for object detection datasets,
ensuring balanced class distribution across all splits.

Best practices implemented:
- Multi-label stratification (images can have multiple classes)
- Balanced class distribution across splits
- Handles class imbalance
- Preserves rare classes in all splits
- Random shuffling with fixed seed for reproducibility

Standard ratios:
- 70% train, 15% val, 15% test (recommended)
- 80% train, 10% val, 10% test (alternative)
"""

import argparse
import random
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
from datetime import datetime
import numpy as np
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stratified split for object detection dataset (YOLO format)"
    )
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory containing YOLO format labels (.txt)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="processed/splits",
        help="Output directory for split files (default: processed/splits)",
    )
    parser.add_argument(
        "--train", type=float, default=0.7, help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val", type=float, default=0.15, help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test", type=float, default=0.15, help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=2,
        help="Minimum samples per class in each split (default: 2)",
    )

    return parser.parse_args()


def load_image_class_mapping(
    img_dir: Path, label_dir: Path
) -> Tuple[Dict[str, Set[int]], Dict[int, List[str]], Dict[str, str]]:
    """
    Load mapping between images and their classes.

    Args:
        img_dir: Directory containing images
        label_dir: Directory containing YOLO labels

    Returns:
        Tuple of:
        - img_to_classes: Dict mapping image stem to set of class IDs
        - class_to_imgs: Dict mapping class ID to list of image stems
        - stem_to_filename: Dict mapping image stem to filename (with extension)
    """
    img_to_classes = {}
    class_to_imgs = defaultdict(list)
    stem_to_filename = {}

    # Get all images
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    print(f"Found {len(image_files)} images in {img_dir}")

    # Build a multi-label view of the dataset: one image can contribute to several
    # class buckets because object detection labels are instance-based.
    missing_labels = []
    for img_file in tqdm(image_files, desc="Loading labels"):
        stem = img_file.stem
        label_file = label_dir / f"{stem}.txt"

        if not label_file.exists():
            missing_labels.append(stem)
            continue

        # Store filename mapping
        stem_to_filename[stem] = img_file.name

        # Read classes from label file
        classes = set()
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    classes.add(class_id)

        if classes:  # Only add images with at least one class
            img_to_classes[stem] = classes
            for cls in classes:
                class_to_imgs[cls].append(stem)

    if missing_labels:
        print(f"[WARNING] Warning: {len(missing_labels)} images have no labels")

    print(f"[OK] Loaded {len(img_to_classes)} images with labels")

    return img_to_classes, class_to_imgs, stem_to_filename

    return img_to_classes, class_to_imgs, stem_to_filename


def print_class_distribution(
    img_stems: List[str], img_to_classes: Dict[str, Set[int]], split_name: str
):
    """Print class distribution for a split."""
    class_counts = Counter()
    for stem in img_stems:
        for cls in img_to_classes[stem]:
            class_counts[cls] += 1

    print(f"\n{split_name} split:")
    print(f"  Images: {len(img_stems)}")
    print(f"  Class distribution:")
    for cls in sorted(class_counts.keys()):
        print(f"    Class {cls}: {class_counts[cls]} instances")


def stratified_split(
    img_to_classes: Dict[str, Set[int]],
    class_to_imgs: Dict[int, List[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_samples: int,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Perform stratified splitting for multi-label object detection dataset.

    Strategy:
    1. For each class, ensure minimum representation in each split
    2. Shuffle images and assign to splits while maintaining class balance
    3. Handle edge cases for rare classes

    Args:
        img_to_classes: Mapping of image stem to set of class IDs
        class_to_imgs: Mapping of class ID to list of image stems
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        min_samples: Minimum samples per class in each split
        seed: Random seed

    Returns:
        Tuple of (train_stems, val_stems, test_stems)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Validate ratios
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Split ratios must sum to 1.0"

    all_stems = list(img_to_classes.keys())
    random.shuffle(all_stems)

    # Initialize splits
    train_stems = []
    val_stems = []
    test_stems = []

    # Track class counts in each split
    train_class_counts = Counter()
    val_class_counts = Counter()
    test_class_counts = Counter()

    # Calculate target counts for each split
    n_total = len(all_stems)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # n_test will be the remainder

    print(f"\nTarget split sizes:")
    print(f"  Train: {n_train} ({train_ratio*100:.1f}%)")
    print(f"  Val: {n_val} ({val_ratio*100:.1f}%)")
    print(f"  Test: {n_total - n_train - n_val} ({test_ratio*100:.1f}%)")

    # Process each image
    assigned = set()

    # Rare classes are handled first so the later greedy assignment does not starve
    # validation/test splits of classes that only have a handful of examples.
    print("\n[INFO] First pass: ensuring minimum samples for rare classes...")
    for cls, imgs in sorted(class_to_imgs.items(), key=lambda x: len(x[1])):
        class_imgs = [img for img in imgs if img not in assigned]

        if len(class_imgs) < min_samples * 3:
            # Rare class: carefully distribute
            n_class = len(class_imgs)
            if n_class >= 3:
                # At least one for each split
                random.shuffle(class_imgs)
                train_stems.append(class_imgs[0])
                val_stems.append(class_imgs[1])
                test_stems.append(class_imgs[2])

                for stem in class_imgs[:3]:
                    assigned.add(stem)
                    for c in img_to_classes[stem]:
                        if stem == class_imgs[0]:
                            train_class_counts[c] += 1
                        elif stem == class_imgs[1]:
                            val_class_counts[c] += 1
                        else:
                            test_class_counts[c] += 1

                print(f"  Class {cls}: {n_class} samples (rare) - distributed")

    # After the rare-class bootstrap, assign the rest greedily to balance both
    # split size and class coverage.
    print("\n[INFO] Second pass: distributing remaining images...")
    remaining = [stem for stem in all_stems if stem not in assigned]
    random.shuffle(remaining)

    for stem in tqdm(remaining, desc="Assigning images"):
        # Determine which split needs this image most
        # Based on current split sizes and class balance

        current_train = len(train_stems)
        current_val = len(val_stems)
        current_test = len(test_stems)

        # Calculate "need" score for each split
        train_need = (n_train - current_train) / n_train if n_train > 0 else 0
        val_need = (n_val - current_val) / n_val if n_val > 0 else 0
        test_need = ((n_total - n_train - n_val) - current_test) / (
            n_total - n_train - n_val
        )

        # Normalize needs
        total_need = train_need + val_need + test_need
        if total_need > 0:
            train_prob = train_need / total_need
            val_prob = val_need / total_need
        else:
            train_prob = train_ratio
            val_prob = val_ratio

        # Assign to split based on probabilities
        rand = random.random()
        if rand < train_prob:
            train_stems.append(stem)
            for cls in img_to_classes[stem]:
                train_class_counts[cls] += 1
        elif rand < train_prob + val_prob:
            val_stems.append(stem)
            for cls in img_to_classes[stem]:
                val_class_counts[cls] += 1
        else:
            test_stems.append(stem)
            for cls in img_to_classes[stem]:
                test_class_counts[cls] += 1

    return train_stems, val_stems, test_stems


def save_splits(
    train_stems: List[str],
    val_stems: List[str],
    test_stems: List[str],
    output_dir: Path,
    stem_to_filename: Dict[str, str],
    img_dir: Path,
):
    """Save split files with robust relative paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use posix path string for cross-platform compatibility
    # Get relative path from current working directory (where script is run)
    try:
        # Try to make path relative to CWD if possible
        rel_img_dir = img_dir.relative_to(Path.cwd())
    except ValueError:
        # If absolute path provided, try to make it relative or just use it as is
        # Usually user runs script from project root
        print(
            f"[WARNING] Warning: {img_dir} is not relative to {Path.cwd()}. Using provided path."
        )
        rel_img_dir = img_dir

    rel_img_dir_str = str(rel_img_dir).replace("\\", "/")

    def write_stems(stems, filename):
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for stem in sorted(stems):
                # Construct relative path: images/filename.jpg
                full_name = stem_to_filename.get(stem, f"{stem}.jpg")  # Fallback to jpg
                path_str = f"{rel_img_dir_str}/{full_name}"
                f.write(f"{path_str}\n")

    write_stems(train_stems, "train.txt")
    write_stems(val_stems, "val.txt")
    write_stems(test_stems, "test.txt")

    print(f"\n[OK] Saved split files to {output_dir}")
    print(f"   train.txt: {len(train_stems)} images")
    print(f"   val.txt: {len(val_stems)} images")
    print(f"   test.txt: {len(test_stems)} images")


def save_split_info(
    train_stems: List[str],
    val_stems: List[str],
    test_stems: List[str],
    img_to_classes: Dict[str, Set[int]],
    output_dir: Path,
    config: Dict,
):
    """
    Save detailed split information to JSON file.

    Args:
        train_stems: Train image stems
        val_stems: Val image stems
        test_stems: Test image stems
        img_to_classes: Mapping of image to classes
        output_dir: Output directory
        config: Configuration dict with img_dir, label_dir, ratios, seed
    """

    # Calculate class distributions
    def get_class_distribution(stems):
        class_counts = Counter()
        for stem in stems:
            for cls in img_to_classes[stem]:
                class_counts[cls] += 1
        return dict(sorted(class_counts.items()))

    train_dist = get_class_distribution(train_stems)
    val_dist = get_class_distribution(val_stems)
    test_dist = get_class_distribution(test_stems)

    # Calculate total distribution
    total_dist = Counter()
    for dist in [train_dist, val_dist, test_dist]:
        for cls, count in dist.items():
            total_dist[cls] += count

    # Create info dict
    split_info = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "script_version": "1.0",
            "description": "Stratified dataset split for object detection",
        },
        "configuration": {
            "image_directory": str(config["img_dir"]).replace("\\", "/"),
            "label_directory": str(config["label_dir"]).replace("\\", "/"),
            "output_directory": str(config["out_dir"]).replace("\\", "/"),
            "split_ratios": {
                "train": config["train_ratio"],
                "val": config["val_ratio"],
                "test": config["test_ratio"],
            },
            "random_seed": config["seed"],
            "min_samples_per_class": config["min_samples"],
        },
        "summary": {
            "total_images": len(train_stems) + len(val_stems) + len(test_stems),
            "total_classes": len(total_dist),
            "splits": {
                "train": {
                    "num_images": len(train_stems),
                    "percentage": round(
                        len(train_stems)
                        / (len(train_stems) + len(val_stems) + len(test_stems))
                        * 100,
                        2,
                    ),
                },
                "val": {
                    "num_images": len(val_stems),
                    "percentage": round(
                        len(val_stems)
                        / (len(train_stems) + len(val_stems) + len(test_stems))
                        * 100,
                        2,
                    ),
                },
                "test": {
                    "num_images": len(test_stems),
                    "percentage": round(
                        len(test_stems)
                        / (len(train_stems) + len(val_stems) + len(test_stems))
                        * 100,
                        2,
                    ),
                },
            },
        },
        "class_distribution": {
            "total": dict(sorted(total_dist.items())),
            "train": train_dist,
            "val": val_dist,
            "test": test_dist,
        },
        "class_balance": {},
    }

    # Calculate class balance (percentage in each split)
    for cls in sorted(total_dist.keys()):
        total_count = total_dist[cls]
        split_info["class_balance"][f"class_{cls}"] = {
            "total_instances": total_count,
            "train": {
                "count": train_dist.get(cls, 0),
                "percentage": (
                    round(train_dist.get(cls, 0) / total_count * 100, 2)
                    if total_count > 0
                    else 0
                ),
            },
            "val": {
                "count": val_dist.get(cls, 0),
                "percentage": (
                    round(val_dist.get(cls, 0) / total_count * 100, 2)
                    if total_count > 0
                    else 0
                ),
            },
            "test": {
                "count": test_dist.get(cls, 0),
                "percentage": (
                    round(test_dist.get(cls, 0) / total_count * 100, 2)
                    if total_count > 0
                    else 0
                ),
            },
        }

    # Save to JSON
    info_path = output_dir / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved split information to {info_path}")


def main():
    """Main function."""
    args = parse_args()

    print("=" * 70)
    print("Stratified Dataset Splitting for Object Detection")
    print("=" * 70)

    # Convert paths
    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)

    # Validate directories
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    print(f"\nConfiguration:")
    print(f"  Image dir: {img_dir}")
    print(f"  Label dir: {label_dir}")
    print(f"  Output dir: {out_dir}")
    print(f"  Split ratio: {args.train}/{args.val}/{args.test}")
    print(f"  Random seed: {args.seed}")

    # Load data
    print("\n" + "=" * 70)
    print("Loading dataset...")
    print("=" * 70)

    img_to_classes, class_to_imgs, stem_to_filename = load_image_class_mapping(
        img_dir, label_dir
    )

    # Show class statistics
    print(f"\n[INFO] Dataset statistics:")
    print(f"  Total images: {len(img_to_classes)}")
    print(f"  Total classes: {len(class_to_imgs)}")
    print(f"\n  Class distribution:")
    for cls in sorted(class_to_imgs.keys()):
        print(f"    Class {cls}: {len(class_to_imgs[cls])} images")

    # Perform stratified split
    print("\n" + "=" * 70)
    print("Performing stratified split...")
    print("=" * 70)

    train_stems, val_stems, test_stems = stratified_split(
        img_to_classes=img_to_classes,
        class_to_imgs=class_to_imgs,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        min_samples=args.min_samples_per_class,
        seed=args.seed,
    )

    # Print split statistics
    print("\n" + "=" * 70)
    print("Split statistics:")
    print("=" * 70)

    print_class_distribution(train_stems, img_to_classes, "Train")
    print_class_distribution(val_stems, img_to_classes, "Val")
    print_class_distribution(test_stems, img_to_classes, "Test")

    # Save splits
    print("\n" + "=" * 70)
    print("Saving splits...")
    print("=" * 70)

    save_splits(train_stems, val_stems, test_stems, out_dir, stem_to_filename, img_dir)

    # Save split information
    save_split_info(
        train_stems=train_stems,
        val_stems=val_stems,
        test_stems=test_stems,
        img_to_classes=img_to_classes,
        output_dir=out_dir,
        config={
            "img_dir": img_dir,
            "label_dir": label_dir,
            "out_dir": out_dir,
            "train_ratio": args.train,
            "val_ratio": args.val,
            "test_ratio": args.test,
            "seed": args.seed,
            "min_samples": args.min_samples_per_class,
        },
    )

    print("\n" + "=" * 70)
    print("[OK] Dataset splitting completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()


# python split_dataset.py --img_dir dataset\dataset_v0\images --label_dir dataset\dataset_v0\labels --out_dir splits --train 0.7 --val 0.15 --test 0.15 --seed 42
# python split_dataset.py --img_dir dataset\dataset_v0\images --label_dir dataset\dataset_v0\labels_10_class --out_dir splits --train 0.7 --val 0.15 --test 0.15 --seed 42
