"""
Knee Region Cropping Pipeline

Cắt vùng đầu gối từ X-ray toàn thân và transform KL grade labels.

Chiến lược:
1. Sử dụng knee detection box làm anchor
2. Mở rộng knee box để bao quanh các KL labels gần đó (trong 0.75x knee size)
3. Expand thành hình vuông + margin 15%
4. Transform tọa độ KL labels về crop space
5. Chỉ drop labels hoàn toàn nằm ngoài crop region

Kết quả:
- Multi-knee images → multiple crops (file_knee0.jpg, file_knee1.jpg)
- Label retention ~100% (chỉ drop labels xa)
- Square crops ready for training

Usage:
    python scripts/data_preparation/crop_knee_regions.py \
        --dataset_dir dataset/dataset_v0 \
        --output_dir processed/knee \
        --margin 0.15 \
        --min_size 300
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import json
from typing import List, Dict, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.image_io import read_image, write_image, get_image_size


def load_yolo_boxes(label_path: Path) -> List[Dict]:
    """
    Đọc YOLO format labels từ file.

    Args:
        label_path: Đường dẫn đến file .txt chứa YOLO labels

    Returns:
        List[Dict]: Danh sách boxes, mỗi box có keys:
            - class_id (int): Class ID
            - x, y, w, h (float): Tọa độ normalized [0,1]

    Format YOLO:
        class_id x_center y_center width height
        Example: 2 0.5 0.5 0.3 0.4
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    class_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    boxes.append({"class_id": class_id, "x": x, "y": y, "w": w, "h": h})
                except ValueError:
                    continue
    return boxes


def yolo_to_xyxy(box: Dict, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format (normalized center + size) sang pixel coordinates (x1,y1,x2,y2).

    Args:
        box: Dict với keys 'x', 'y', 'w', 'h' (normalized [0,1])
        img_w, img_h: Kích thước ảnh (pixels)

    Returns:
        Tuple[int, int, int, int]: (x1, y1, x2, y2) in pixels

    Example:
        box = {'x': 0.5, 'y': 0.5, 'w': 0.3, 'h': 0.4}
        img_w, img_h = 1000, 1000
        → (350, 300, 650, 700)  # x1, y1, x2, y2
    """
    x_center = box["x"] * img_w
    y_center = box["y"] * img_h
    w = box["w"] * img_w
    h = box["h"] * img_h

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return x1, y1, x2, y2


def expand_to_square(x1, y1, x2, y2, img_w, img_h, margin=0.1):
    """
    Mở rộng box thành hình vuông với margin, clamp vào image boundary.

    Quy trình:
    1. Tính kích thước hiện tại: box_w, box_h
    2. Chọn max(box_w, box_h) làm size
    3. Thêm margin: size * (1 + margin)
    4. Expand từ center thành square
    5. Clamp về [0, img_w] và [0, img_h]

    Args:
        x1, y1, x2, y2: Box coordinates (pixels)
        img_w, img_h: Image dimensions (pixels)
        margin: Margin fraction to add (0.15 = 15%)

    Returns:
        Tuple[int, int, int, int]: Square box (x1, y1, x2, y2) clamped to image

    Example:
        Input box: (100, 200, 400, 500)  # 300×300
        Margin: 0.15
        → Square size: 300 * 1.15 = 345
        → Expand từ center (250, 350)
        → Output: (77, 177, 422, 522) clamped to image bounds
    """
    # Current box dimensions
    box_w = x2 - x1
    box_h = y2 - y1

    # Make square (use larger dimension)
    size = max(box_w, box_h)

    # Add margin
    size_with_margin = int(size * (1 + margin))

    # Calculate center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # New square coordinates
    new_x1 = int(cx - size_with_margin / 2)
    new_y1 = int(cy - size_with_margin / 2)
    new_x2 = int(cx + size_with_margin / 2)
    new_y2 = int(cy + size_with_margin / 2)

    # Clamp to image bounds
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)

    return new_x1, new_y1, new_x2, new_y2


def transform_labels_to_crop(
    kl_labels, crop_x1, crop_y1, crop_x2, crop_y2, img_w, img_h
):
    """
    Transform KL labels từ tọa độ full image sang crop coordinates.

    Chiến lược:
    - Chỉ drop labels **hoàn toàn** ngoài crop (center cách xa > box size)
    - Labels gần edge được giữ lại vì knee box đã expand để bao chúng
    - Clamp box boundaries về crop region để tránh out-of-range

    Args:
        kl_labels: List[Dict] - KL labels in YOLO format (relative to full image)
        crop_x1, crop_y1, crop_x2, crop_y2: Crop region (pixels)
        img_w, img_h: Full image dimensions (pixels)

    Returns:
        Tuple:
            - transformed_boxes: List[Dict] - Labels in crop space (YOLO format)
            - dropped_boxes: List[Dict] - Dropped labels with reasons

    Transform Steps:
    1. Check if label completely outside crop → drop
    2. Transform center: (x_full - crop_x1, y_full - crop_y1)
    3. Clamp box boundaries to [0, crop_w] × [0, crop_h]
    4. Recalculate center from clamped boundaries
    5. Normalize to [0, 1]
    6. Guard against floating point negative zeros
    """
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w <= 0 or crop_h <= 0:
        return [], []

    transformed = []
    dropped = []

    for box in kl_labels:
        # Convert to pixel coordinates in full image
        x_center = box["x"] * img_w
        y_center = box["y"] * img_h
        width = box["w"] * img_w
        height = box["h"] * img_h

        # Check if box center is within crop region
        # Note: Knee box was already expanded to include nearby labels,
        # so we only drop labels that are truly far outside
        if not (
            crop_x1 - width <= x_center <= crop_x2 + width
            and crop_y1 - height <= y_center <= crop_y2 + height
        ):
            # Box is completely outside crop region
            dropped.append(
                {
                    "box": box,
                    "reason": "completely_outside_crop",
                    "center": (x_center, y_center),
                    "crop": (crop_x1, crop_y1, crop_x2, crop_y2),
                }
            )
            continue

        # Transform to crop coordinates
        new_x_center = x_center - crop_x1
        new_y_center = y_center - crop_y1

        # Keep width/height same
        new_width = width
        new_height = height

        # Calculate box boundaries in crop space (before normalization)
        box_x1 = new_x_center - new_width / 2
        box_y1 = new_y_center - new_height / 2
        box_x2 = new_x_center + new_width / 2
        box_y2 = new_y_center + new_height / 2

        # Clamp box boundaries to crop region [0, crop_w] and [0, crop_h]
        box_x1 = max(0, box_x1)
        box_y1 = max(0, box_y1)
        box_x2 = min(crop_w, box_x2)
        box_y2 = min(crop_h, box_y2)

        # Recalculate center and dimensions after clamping
        clamped_width = box_x2 - box_x1
        clamped_height = box_y2 - box_y1
        clamped_x_center = (box_x1 + box_x2) / 2
        clamped_y_center = (box_y1 + box_y2) / 2

        # Normalize to crop dimensions
        new_x = clamped_x_center / crop_w
        new_y = clamped_y_center / crop_h
        new_w = clamped_width / crop_w
        new_h = clamped_height / crop_h

        # Final safety clamp to [0, 1]
        new_x = np.clip(new_x, 0.0, 1.0)
        new_y = np.clip(new_y, 0.0, 1.0)
        new_w = np.clip(new_w, 0.01, 1.0)  # Min 1% to avoid tiny boxes
        new_h = np.clip(new_h, 0.01, 1.0)

        # Guard against negative zeros from floating point precision
        if new_x < 0.0001:
            new_x = 0.0
        if new_y < 0.0001:
            new_y = 0.0

        # Keep ALL boxes with center in crop (no size threshold)
        transformed.append(
            {
                "class_id": box["class_id"],
                "x": new_x,
                "y": new_y,
                "w": new_w,
                "h": new_h,
                "original_size": (width, height),  # Track original size
            }
        )

    return transformed, dropped


def save_yolo_labels(boxes, output_path: Path):
    """Save boxes in YOLO format."""
    if not boxes:
        # Create empty file
        output_path.write_text("")
        return

    lines = []
    for box in boxes:
        line = f"{box['class_id']} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}"
        lines.append(line)

    output_path.write_text("\n".join(lines) + "\n")


def crop_knee_regions(
    dataset_dir: Path, output_dir: Path, margin: float = 0.15, min_size: int = 300
):
    """
    Crop knee regions from full X-ray images.

    Args:
        dataset_dir: Path to dataset_v0
        output_dir: Path to output directory (processed/knee)
        margin: Margin to add around knee box (fraction, e.g., 0.15 = 15%)
        min_size: Minimum crop size in pixels
    """
    print("=" * 80)
    print("KNEE REGION CROPPING")
    print("=" * 80)
    print(f"\nDataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Margin: {margin * 100:.0f}%")
    print(f"Min size: {min_size}px")

    # Setup paths
    img_dir = dataset_dir / "images"
    knee_label_dir = dataset_dir / "labels-knee"
    kl_label_dir = dataset_dir / "labels"

    output_img_dir = output_dir / "images"
    output_label_dir = output_dir / "labels"
    output_knee_dir = output_dir / "labels-knee"

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    output_knee_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    print(f"\nFound {len(images)} images")

    # Statistics
    stats = {
        "total_images": len(images),
        "cropped": 0,
        "skipped_no_knee_box": 0,
        "skipped_too_small": 0,
        "total_kl_boxes_original": 0,
        "total_kl_boxes_transformed": 0,
        "total_kl_boxes_dropped": 0,
    }

    # Detailed logging
    skipped_files = []
    dropped_labels_log = []
    expanded_files_log = []

    # Process each image
    for img_path in tqdm(images, desc="Cropping"):
        stem = img_path.stem

        # Load knee detection boxes
        knee_label_path = knee_label_dir / f"{stem}.txt"
        knee_boxes = load_yolo_boxes(knee_label_path)

        if not knee_boxes:
            stats["skipped_no_knee_box"] += 1
            # Try to get image size if possible, otherwise None
            try:
                img_w_temp, img_h_temp = get_image_size(img_path)
            except (ValueError, ImportError):
                img_w_temp, img_h_temp = None, None
            skipped_files.append(
                {
                    "file": stem,
                    "reason": "no_knee_box",
                    "image_size": (img_w_temp, img_h_temp),
                }
            )
            continue

        # Load image once for all knee boxes
        try:
            img = read_image(img_path)
        except (ValueError, ImportError):
            img = None
        if img is None:
            stats["skipped_no_knee_box"] += 1
            skipped_files.append(
                {
                    "file": stem,
                    "reason": "image_load_failed",
                    "image_path": str(img_path),
                }
            )
            continue

        img_h, img_w = img.shape[:2]

        # Load KL labels once for all knees
        kl_label_path = kl_label_dir / f"{stem}.txt"
        kl_boxes = load_yolo_boxes(kl_label_path)

        # In full-leg images, multiple knees can share one source frame. Assign each
        # lesion label to the nearest knee first so every crop keeps only its local
        # targets before any bbox expansion happens.
        knee_assignments = {i: [] for i in range(len(knee_boxes))}
        if knee_boxes and kl_boxes:
            # Get knee centers
            knee_centers = []
            for kb in knee_boxes:
                k_cx = kb["x"] * img_w
                k_cy = kb["y"] * img_h
                knee_centers.append((k_cx, k_cy))

            # Assign each KL label to closest knee
            for kl_box in kl_boxes:
                kl_cx = kl_box["x"] * img_w
                kl_cy = kl_box["y"] * img_h

                min_dist = float("inf")
                best_knee_idx = -1

                for idx, (k_cx, k_cy) in enumerate(knee_centers):
                    dist = ((kl_cx - k_cx) ** 2 + (kl_cy - k_cy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_knee_idx = idx

                if best_knee_idx != -1:
                    knee_assignments[best_knee_idx].append(kl_box)

        # Process EACH knee box from this image
        for knee_idx, knee_box in enumerate(knee_boxes):
            # Convert knee box to pixel coordinates
            kx1, ky1, kx2, ky2 = yolo_to_xyxy(knee_box, img_w, img_h)

            # Get center and size of knee box for proximity check
            knee_cx = (kx1 + kx2) / 2
            knee_cy = (ky1 + ky2) / 2
            knee_w = kx2 - kx1
            knee_h = ky2 - ky1
            knee_size = max(knee_w, knee_h)

            # Store original coordinates to check for expansion later
            orig_kx1, orig_ky1, orig_kx2, orig_ky2 = kx1, ky1, kx2, ky2

            # Expand only with the labels already assigned to this knee; this avoids
            # leaking lesions from the contralateral knee into the crop.
            assigned_labels = knee_assignments[knee_idx]
            if assigned_labels:
                for kl_box in assigned_labels:
                    kl_w = kl_box["w"] * img_w
                    kl_h = kl_box["h"] * img_h
                    kl_x_center = kl_box["x"] * img_w
                    kl_y_center = kl_box["y"] * img_h

                    kl_x1 = kl_x_center - kl_w / 2
                    kl_y1 = kl_y_center - kl_h / 2
                    kl_x2 = kl_x_center + kl_w / 2
                    kl_y2 = kl_y_center + kl_h / 2

                    # Expand knee box to include this assigned KL box
                    # Add 10% padding relative to the KL label size
                    kl_size = max(kl_w, kl_h)
                    padding = kl_size * 0.1

                    kx1 = min(kx1, kl_x1 - padding)
                    ky1 = min(ky1, kl_y1 - padding)
                    kx2 = max(kx2, kl_x2 + padding)
                    ky2 = max(ky2, kl_y2 + padding)

            # Check if expansion happened (with 1px tolerance for float precision)
            tolerance = 1.0
            if (
                kx1 < orig_kx1 - tolerance
                or ky1 < orig_ky1 - tolerance
                or kx2 > orig_kx2 + tolerance
                or ky2 > orig_ky2 + tolerance
            ):
                expanded_files_log.append(
                    {
                        "file": f"{stem}_knee{knee_idx}",
                        "original_box": (orig_kx1, orig_ky1, orig_kx2, orig_ky2),
                        "expanded_box": (kx1, ky1, kx2, ky2),
                    }
                )

            # Expand to square with margin
            crop_x1, crop_y1, crop_x2, crop_y2 = expand_to_square(
                kx1, ky1, kx2, ky2, img_w, img_h, margin=margin
            )

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            # Check minimum size
            if crop_w < min_size or crop_h < min_size:
                stats["skipped_too_small"] += 1
                skipped_files.append(
                    {
                        "file": f"{stem}_knee{knee_idx}",
                        "reason": "crop_too_small",
                        "crop_size": (crop_w, crop_h),
                        "min_size": min_size,
                    }
                )
                continue

            # Crop image
            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

            # Count original boxes only once per image
            if knee_idx == 0:
                stats["total_kl_boxes_original"] += len(kl_boxes)

            # Transform KL labels to this crop
            transformed_boxes, dropped_boxes = transform_labels_to_crop(
                assigned_labels, crop_x1, crop_y1, crop_x2, crop_y2, img_w, img_h
            )
            stats["total_kl_boxes_transformed"] += len(transformed_boxes)
            stats["total_kl_boxes_dropped"] += len(dropped_boxes)

            # Log dropped labels
            if dropped_boxes:
                for drop in dropped_boxes:
                    dropped_labels_log.append(
                        {
                            "file": f"{stem}_knee{knee_idx}",
                            "class_id": drop["box"]["class_id"],
                            "reason": drop["reason"],
                            "details": str(drop),
                        }
                    )

            # Generate output filename with index if multiple knees
            if len(knee_boxes) == 1:
                # Single knee: keep original name
                output_name = img_path.name
                output_stem = stem
            else:
                # Multiple knees: add index
                output_name = f"{stem}_knee{knee_idx}{img_path.suffix}"
                output_stem = f"{stem}_knee{knee_idx}"

            # ⭐ FILTER: Only save if there are labels
            if not transformed_boxes:
                stats["skipped_no_labels"] = stats.get("skipped_no_labels", 0) + 1
                skipped_files.append(
                    {
                        "file": output_stem,
                        "reason": "no_labels_after_transform",
                        "original_kl_boxes": len(assigned_labels),
                        "transformed_boxes": 0,
                    }
                )
                continue

            # Save cropped image
            output_img_path = output_img_dir / output_name
            write_image(output_img_path, cropped)

            # Save transformed labels
            output_label_path = output_label_dir / f"{output_stem}.txt"
            save_yolo_labels(transformed_boxes, output_label_path)

            # Save knee box (full crop, since entire image is knee region)
            output_knee_path = output_knee_dir / f"{output_stem}.txt"
            knee_full_box = {"class_id": 0, "x": 0.5, "y": 0.5, "w": 1.0, "h": 1.0}
            save_yolo_labels([knee_full_box], output_knee_path)

            stats["cropped"] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("CROPPING SUMMARY")
    print("=" * 80)
    print(f"\nTotal images: {stats['total_images']}")
    print(f"[OK] Successfully cropped: {stats['cropped']}")
    print(f"[SKIP] No knee box: {stats['skipped_no_knee_box']}")
    print(f"[SKIP] Too small: {stats['skipped_too_small']}")
    print(f"[SKIP] No labels: {stats.get('skipped_no_labels', 0)}")
    print(f"\nKL Labels:")
    print(f"  Original boxes: {stats['total_kl_boxes_original']}")
    print(f"  Transformed boxes: {stats['total_kl_boxes_transformed']}")
    print(f"  Dropped boxes: {stats['total_kl_boxes_dropped']}")
    # Avoid division by zero if no original boxes
    if stats["total_kl_boxes_original"] > 0:
        print(
            f"  Retention rate: {stats['total_kl_boxes_transformed']/stats['total_kl_boxes_original']*100:.1f}%"
        )
    else:
        print(f"  Retention rate: N/A (no original KL boxes found)")

    # Save stats
    stats_path = output_dir / "crop_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to: {stats_path}")

    # Save skipped files log
    if skipped_files:
        skip_log_path = output_dir / "skipped_files.json"
        with open(skip_log_path, "w") as f:
            json.dump(skipped_files, f, indent=2)
        print(
            f"📝 Saved skipped files log ({len(skipped_files)} files): {skip_log_path}"
        )

    # Save dropped labels log
    if dropped_labels_log:
        drop_log_path = output_dir / "dropped_labels.json"
        with open(drop_log_path, "w") as f:
            json.dump(dropped_labels_log, f, indent=2)
        print(
            f"📝 Saved dropped labels log ({len(dropped_labels_log)} labels): {drop_log_path}"
        )

    # Save expanded files log
    if expanded_files_log:
        expanded_log_path = output_dir / "expanded_files.json"
        with open(expanded_log_path, "w") as f:
            json.dump(expanded_files_log, f, indent=2)
        print(
            f"📝 Saved expanded files log ({len(expanded_files_log)} files - labels outside original knee box): {expanded_log_path}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Crop knee regions from X-ray images")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to dataset_v0"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for cropped images",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="Margin around knee box (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=300,
        help="Minimum crop size in pixels (default: 300)",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return

    crop_knee_regions(dataset_dir, output_dir, args.margin, args.min_size)


if __name__ == "__main__":
    main()
