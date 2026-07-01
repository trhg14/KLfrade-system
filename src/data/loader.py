from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from collections import defaultdict
from src.data.dataset_utils import get_image_files, parse_yolo_label

def load_dataset_mapping(
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
    image_files = get_image_files(img_dir)
    print(f"Found {len(image_files)} images in {img_dir}")

    # Parse labels
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
        classes = parse_yolo_label(label_file)

        if classes:  # Only add images with at least one class
            img_to_classes[stem] = classes
            for cls in classes:
                class_to_imgs[cls].append(stem)

    if missing_labels:
        print(f"⚠️  Warning: {len(missing_labels)} images have no labels")

    print(f"✅ Loaded {len(img_to_classes)} images with labels")

    return img_to_classes, class_to_imgs, stem_to_filename
