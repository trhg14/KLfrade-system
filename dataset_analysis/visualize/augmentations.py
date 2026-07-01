"""
Augmentation Visualization Tool

Visualizes the effects of data augmentation on sample images
to verify correctness before training.

Generates before/after comparison images showing:
- Original image with bboxes
- Augmented image with transformed bboxes

Usage:
    python tools/check_dataset/visualize_augmentations.py 
        --img_dir processed/knee/images 
        --label_dir processed/knee/labels 
        --num_samples 20 
        --save_dir analysis/augmentation_checks
"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets import (
    get_conservative_train_transform,
    get_moderate_train_transform,
)
from src.datasets import YoloDataset
from src.config import CLASSES


def draw_boxes(img, boxes, class_labels, class_names, color=(0, 255, 0)):
    """Draw bounding boxes on image"""
    img_draw = img.copy()
    if len(img_draw.shape) == 2:  # Grayscale
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)

    h, w = img_draw.shape[:2]

    for box, cls in zip(boxes, class_labels):
        # YOLO format: center_x, center_y, width, height
        cx, cy, bw, bh = box

        # Convert to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # Draw box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = class_names.get(int(cls), f"Class {cls}")
        cv2.putText(
            img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return img_draw


def visualize_augmentations(
    img_dir,
    label_dir,
    num_samples=20,
    save_dir="analysis/augmentation_checks",
    augmentation_type="conservative",
):
    """
    Visualize original vs augmented samples.

    Args:
        img_dir: Directory containing images
        label_dir: Directory containing YOLO labels
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualization results
        augmentation_type: 'conservative' or 'moderate'
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Select augmentation
    if augmentation_type == "conservative":
        transform = get_conservative_train_transform()
    else:
        transform = get_moderate_train_transform()

    # Load dataset without transform
    dataset_raw = YoloDataset(img_dir=img_dir, label_dir=label_dir, transform=None)

    print(f"\n{'='*80}")
    print(f"AUGMENTATION VISUALIZATION ({augmentation_type.upper()})")
    print(f"{'='*80}")
    print(f"\nDataset: {img_dir}")
    print(f"Num samples: {num_samples}")
    print(f"Output: {save_dir}\n")

    for idx in tqdm(
        range(min(num_samples, len(dataset_raw))), desc="Generating visualizations"
    ):
        # Get original sample - YoloDataset returns tuple: (image, boxes, labels)
        try:
            img_raw, boxes_raw, labels_raw = dataset_raw[idx]

            # Convert from tensor if needed
            if isinstance(img_raw, torch.Tensor):
                img_raw = (img_raw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if isinstance(boxes_raw, torch.Tensor):
                boxes_raw = boxes_raw.tolist()
            if isinstance(labels_raw, torch.Tensor):
                labels_raw = labels_raw.tolist()
        except Exception as e:
            print(f"\nWarning: Failed to load sample {idx}: {e}")
            continue

        # Apply augmentation
        try:
            result = transform(image=img_raw, bboxes=boxes_raw, class_labels=labels_raw)

            # Convert tensor back to numpy for visualization
            img_aug = result["image"].permute(1, 2, 0).numpy()
            # Denormalize
            img_aug = (img_aug * 0.5 + 0.5) * 255  # Reverse normalization
            img_aug = img_aug.astype(np.uint8)

            boxes_aug = result["bboxes"]
            labels_aug = result["class_labels"]
        except Exception as e:
            print(f"\nWarning: Failed to augment sample {idx}: {e}")
            continue

        # Draw boxes on both images
        img_raw_draw = draw_boxes(
            img_raw, boxes_raw, labels_raw, CLASSES, color=(0, 255, 0)
        )
        img_aug_draw = draw_boxes(
            img_aug, boxes_aug, labels_aug, CLASSES, color=(0, 255, 0)
        )

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Original
        axes[0].imshow(cv2.cvtColor(img_raw_draw, cv2.COLOR_BGR2RGB))
        axes[0].set_title(
            f"Original (Sample {idx})\n{len(boxes_raw)} boxes", fontsize=12
        )
        axes[0].axis("off")

        # Augmented
        axes[1].imshow(cv2.cvtColor(img_aug_draw, cv2.COLOR_BGR2RGB))
        axes[1].set_title(
            f"Augmented ({augmentation_type})\n{len(boxes_aug)} boxes", fontsize=12
        )
        axes[1].axis("off")

        # Save
        plt.tight_layout()
        plt.savefig(save_dir / f"aug_{idx:03d}.png", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"\n✅ Saved {num_samples} augmentation samples to {save_dir}")
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize data augmentation effects")
    parser.add_argument(
        "--img_dir", type=str, default="processed/knee/images", help="Image directory"
    )
    parser.add_argument(
        "--label_dir", type=str, default="processed/knee/labels", help="Label directory"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="analysis/augmentation_checks",
        help="Output directory",
    )
    parser.add_argument(
        "--aug_type",
        type=str,
        default="conservative",
        choices=["conservative", "moderate"],
        help="Augmentation type",
    )

    args = parser.parse_args()

    visualize_augmentations(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        num_samples=args.num_samples,
        save_dir=args.save_dir,
        augmentation_type=args.aug_type,
    )


if __name__ == "__main__":
    main()
