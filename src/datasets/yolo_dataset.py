"""
YOLO Dataset Loader for Object Detection

This module provides a flexible dataset loader for YOLO format labels, optimized for
knee osteoarthritis (OA) detection using the Kellgren-Lawrence (KL) grading system.

Features:
- Train/Val/Test splits from text files
- Support for both original labels (5 classes) and labels_10_class (10 classes)
- Memory caching for images and labels (optional)
- Robust error handling and validation
- Albumentations-based augmentation with bbox support
- Automatic filtering of images without labels
- Multiple output formats: YOLO (normalized), Pascal VOC (absolute)
- Compatible with YOLO11 (Ultralytics) and PyTorch DataLoader

Author: KLGrade Project
Compatible with: YOLO11, PyTorch 2.0+, Albumentations
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class YoloDataset(Dataset):
    """
    YOLO format dataset loader for object detection tasks.

    This class loads images and YOLO format labels (normalized bounding boxes) and
    supports various preprocessing, augmentation, and output formats.

    Args:
        img_dir (str): Directory containing images (.jpg, .png, .bmp)
        label_dir (str): Directory containing YOLO format label files (.txt)
        transform (Optional[A.Compose]): Albumentations transform pipeline
        split_file (Optional[str]): Path to split file (train.txt/val.txt) containing image stems
        use_labels_10_class (bool): Use labels_10_class directory (10 classes) instead of labels (5 classes)
        filter_no_label (bool): Skip images without corresponding label files
        cache_images (bool): Cache loaded images in memory for faster access
        cache_labels (bool): Cache parsed labels in memory
        img_size (Optional[Tuple[int, int]]): Resize images to (height, width) before transform
        bbox_format (str): Output bbox format - 'yolo' (cx,cy,w,h normalized) or 'pascal_voc' (x1,y1,x2,y2)
        return_dict (bool): Return dict instead of tuple for better compatibility

    Returns:
        If return_dict=False: (image_tensor, boxes_tensor, labels_tensor)
        If return_dict=True: {'image': tensor, 'boxes': tensor, 'labels': tensor, 'image_id': str}

    Example:
        >>> from dataset import YoloDataset, get_default_train_transform
        >>> transform = get_default_train_transform(img_size=(640, 640))
        >>> dataset = YoloDataset(
        ...     img_dir="processed/knee/images",
        ...     label_dir="processed/knee/labels",
        ...     transform=transform,
        ...     split_file="splits/train.txt",
        ...     bbox_format='pascal_voc',
        ...     return_dict=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample['image'].shape, sample['boxes'].shape)
    """

    def __init__(
        self,
        img_dir,
        label_dir,
        transform=None,
        split_file: Optional[str] = None,
        use_labels_10_class: bool = False,
        filter_no_label: bool = True,
        cache_images: bool = False,
        cache_labels: bool = True,
        img_size: Optional[Tuple[int, int]] = None,
        bbox_format: str = "pascal_voc",
        return_dict: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.use_labels_10_class = use_labels_10_class
        self.filter_no_label = filter_no_label
        self.cache_images = cache_images
        self.cache_labels = cache_labels
        self.img_size = img_size
        self.bbox_format = bbox_format
        self.return_dict = return_dict

        # Nếu dùng labels_10_class, thay đổi label_dir
        if use_labels_10_class:
            label_dir_parent = self.label_dir.parent
            self.label_dir = label_dir_parent / "labels_10_class"
            if not self.label_dir.exists():
                warnings.warn(
                    f"labels_10_class directory not found: {self.label_dir}, falling back to labels"
                )
                self.label_dir = Path(label_dir)

        # Load danh sách ảnh
        # Split files contain stems, not full filenames, so resolve the concrete
        # image extension once here and reuse that mapping later.
        self.image_files = self._load_image_list(split_file)

        # Cache
        self.image_cache = {} if cache_images else None
        self.label_cache = {} if cache_labels else None

        # Transform
        self.transform = transform

        # Validate dataset
        self._validate_dataset()

    def _load_image_list(self, split_file: Optional[str] = None) -> List[str]:
        """Load danh sách ảnh từ split_file hoặc từ img_dir"""
        if split_file and os.path.exists(split_file):
            # Load từ split file (chứa stem, không có extension)
            with open(split_file, "r", encoding="utf-8") as f:
                stems = [line.strip() for line in f if line.strip()]

            # Tìm ảnh tương ứng với stems
            image_files = []
            img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

            for stem in stems:
                # Tìm file ảnh với stem này
                found = False
                for ext in img_extensions:
                    img_path = self.img_dir / f"{stem}{ext}"
                    if img_path.exists():
                        image_files.append(img_path.name)
                        found = True
                        break

                if not found:
                    warnings.warn(f"Image not found for stem: {stem}")

            return sorted(image_files)
        else:
            # Load tất cả ảnh từ img_dir (tương thích với code cũ)
            img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            image_files = [
                f.name
                for f in self.img_dir.iterdir()
                if f.suffix.lower() in img_extensions
            ]
            return sorted(image_files)

    def _validate_dataset(self):
        """Validate dataset và filter ảnh không có label nếu cần"""
        valid_files = []
        missing_labels = []

        for img_file in self.image_files:
            label_path = self._get_label_path(img_file)
            if label_path.exists():
                valid_files.append(img_file)
            else:
                missing_labels.append(img_file)
                if not self.filter_no_label:
                    valid_files.append(img_file)

        if missing_labels and self.filter_no_label:
            print(f"⚠️  Filtered out {len(missing_labels)} images without labels")

        self.image_files = valid_files
        if len(self.image_files) > 0:
            print(f"✅ Loaded {len(self.image_files)} images from {self.img_dir}")

    def _get_label_path(self, img_file: str) -> Path:
        """Lấy đường dẫn file label tương ứng với ảnh"""
        stem = Path(img_file).stem
        return self.label_dir / f"{stem}.txt"

    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load ảnh từ disk hoặc cache"""
        if self.cache_images and img_path.name in self.image_cache:
            return self.image_cache[img_path.name]

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize nếu cần
        if self.img_size:
            h, w = self.img_size
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.cache_images:
            self.image_cache[img_path.name] = image

        return image

    def _load_labels(self, label_path: Path) -> Tuple[List[List[float]], List[int]]:
        """
        Load labels from YOLO format file with validation.

        Returns:
            Tuple of (boxes, labels) where boxes are in YOLO format [cx, cy, w, h] (normalized)
        """
        if self.cache_labels and str(label_path) in self.label_cache:
            return self.label_cache[str(label_path)]

        boxes = []
        labels = []

        if not label_path.exists():
            return boxes, labels

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        warnings.warn(
                            f"{label_path.name}:{line_num} - Invalid format (need 5 values)"
                        )
                        continue

                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Validate bbox coordinates (YOLO format should be 0-1)
                    if not (
                        0 <= x_center <= 1
                        and 0 <= y_center <= 1
                        and 0 <= width <= 1
                        and 0 <= height <= 1
                    ):
                        warnings.warn(
                            f"{label_path.name}:{line_num} - Bbox out of range: "
                            f"[{x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f}]"
                        )
                        # Clip to valid range
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                    boxes.append([x_center, y_center, width, height])
                    labels.append(class_id)
        except Exception as e:
            warnings.warn(f"Error loading label {label_path}: {e}")

        if self.cache_labels:
            self.label_cache[str(label_path)] = (boxes, labels)

        return boxes, labels

    def _yolo_to_pascal_voc(
        self, boxes: List[List[float]], img_h: int, img_w: int
    ) -> List[List[float]]:
        """Chuyển đổi từ YOLO format (normalized) sang Pascal VOC format (x1, y1, x2, y2)"""
        pascal_boxes = []
        for box in boxes:
            x_center, y_center, width, height = box
            x1 = (x_center - width / 2) * img_w
            y1 = (y_center - height / 2) * img_h
            x2 = (x_center + width / 2) * img_w
            y2 = (y_center + height / 2) * img_h
            pascal_boxes.append([x1, y1, x2, y2])
        return pascal_boxes

    def _pascal_voc_to_yolo(
        self, boxes: List[List[float]], img_h: int, img_w: int
    ) -> List[List[float]]:
        """Chuyển đổi từ Pascal VOC format (x1, y1, x2, y2) sang YOLO format (normalized)"""
        yolo_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            yolo_boxes.append([x_center, y_center, width, height])
        return yolo_boxes

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        Returns:
            Nếu return_dict=False (mặc định): (image_tensor, boxes_tensor, labels_tensor)
            Nếu return_dict=True: dict với keys: 'image', 'boxes', 'labels', 'image_id'
        """
        img_file = self.image_files[idx]
        img_path = self.img_dir / img_file
        label_path = self._get_label_path(img_file)

        # Load ảnh
        image = self._load_image(img_path)
        img_h, img_w = image.shape[:2]

        # Load labels
        boxes_yolo, labels = self._load_labels(label_path)

        # Chuyển sang Pascal VOC nếu cần (để augment)
        # Albumentations works most naturally with corner-format boxes, so labels are
        # converted to Pascal VOC before augmentation and optionally converted back.
        if boxes_yolo:
            boxes = self._yolo_to_pascal_voc(boxes_yolo, img_h, img_w)
        else:
            boxes = []

        # Augmentation
        if self.transform:
            try:
                transformed = self.transform(
                    image=image, bboxes=boxes, class_labels=labels
                )
                image = transformed["image"]
                boxes = transformed["bboxes"]
                labels = transformed["class_labels"]

                # Cập nhật kích thước ảnh sau transform
                if isinstance(image, torch.Tensor):
                    img_h, img_w = image.shape[1], image.shape[2]
                else:
                    img_h, img_w = image.shape[:2]
            except Exception as e:
                warnings.warn(f"Augmentation error for {img_file}: {e}")

        # Chuyển về YOLO format nếu cần
        if self.bbox_format == "yolo" and boxes:
            boxes = self._pascal_voc_to_yolo(boxes, img_h, img_w)

        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            # Normalize và convert
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image_tensor = image

        # Convert boxes và labels
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            # Empty boxes
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        # Return format
        # Support both the old tuple contract and the newer dict contract so this
        # loader can be reused by training, debugging, and visualization scripts.
        if self.return_dict:
            return {
                "image": image_tensor,
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "image_id": img_file,
            }
        else:
            # Tương thích ngược với code cũ
            return image_tensor, boxes_tensor, labels_tensor


# === Hàm visualize ===
def visualize_sample(image_tensor, boxes, labels, class_names=None):
    """
    Hiển thị ảnh và bbox sau augment
    image_tensor: torch.Tensor (C,H,W)
    boxes: list hoặc tensor [[x1,y1,x2,y2], ...] hoặc YOLO format
    labels: list hoặc tensor
    class_names: danh sách tên lớp (tuỳ chọn)
    """
    image = image_tensor.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    # Convert boxes nếu cần
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    img_h, img_w = image.shape[:2]

    for box, label in zip(boxes, labels):
        # Kiểm tra format: nếu có giá trị > 1 thì là Pascal VOC, ngược lại là YOLO
        if len(box) == 4:
            if max(box) > 1.0:
                # Pascal VOC format
                x1, y1, x2, y2 = box
            else:
                # YOLO format (normalized)
                x_center, y_center, width, height = box
                x1 = (x_center - width / 2) * img_w
                y1 = (y_center - height / 2) * img_h
                x2 = (x_center + width / 2) * img_w
                y2 = (y_center + height / 2) * img_h
        else:
            continue

        width_box = x2 - x1
        height_box = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            width_box,
            height_box,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        if class_names:
            lbl = class_names[int(label)]
        else:
            lbl = str(int(label))
        ax.text(
            x1,
            y1 - 5,
            lbl,
            color="yellow",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    plt.axis("off")
    plt.show()


def visualize_dataset_sample(
    dataset: YoloDataset,
    idx: int,
    class_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize một sample từ dataset

    Args:
        dataset: YoloDataset instance
        idx: Index của sample
        class_names: Dict mapping class_id -> class_name
        save_path: Đường dẫn lưu ảnh (nếu None thì hiển thị)
    """
    sample = dataset[idx]

    # Xử lý cả tuple và dict format
    if isinstance(sample, dict):
        image = sample["image"]
        boxes = sample["boxes"]
        labels = sample["labels"]
        image_id = sample.get("image_id", f"sample_{idx}")
    else:
        image, boxes, labels = sample
        image_id = f"sample_{idx}"

    # Convert image tensor to numpy
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).numpy()
        # Denormalize nếu đã normalize
        if image_np.max() <= 1.0:
            # Có thể đã normalize, thử denormalize với ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
    else:
        image_np = image

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_np)
    ax.set_title(f"Image: {image_id}\nBoxes: {len(boxes)}")

    # Draw boxes
    img_h, img_w = image_np.shape[:2]

    # Convert boxes nếu cần
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    for box, label in zip(boxes, labels):
        if len(box) != 4:
            continue

        # Kiểm tra format
        if dataset.bbox_format == "yolo" or max(box) <= 1.0:
            # YOLO format: normalized (x_center, y_center, width, height)
            x_center, y_center, width, height = box
            x1 = (x_center - width / 2) * img_w
            y1 = (y_center - height / 2) * img_h
            x2 = (x_center + width / 2) * img_w
            y2 = (y_center + height / 2) * img_h
        else:
            # Pascal VOC format: (x1, y1, x2, y2)
            x1, y1, x2, y2 = box

        width_box = x2 - x1
        height_box = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            width_box,
            height_box,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Label text
        if class_names:
            label_text = class_names[int(label)]
        else:
            label_text = f"Class {int(label)}"

        ax.text(
            x1,
            y1 - 5,
            label_text,
            color="yellow",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def get_default_train_transform(
    img_size: Tuple[int, int] = (640, 640), use_augmentation: bool = True
) -> A.Compose:
    """
    Tạo transform mặc định cho training

    Args:
        img_size: (height, width)
        use_augmentation: Có dùng augmentation không
    """
    h, w = img_size

    if use_augmentation:
        transform = A.Compose(
            [
                A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7,
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.15, p=0.5
                ),
                A.GaussNoise(var_limit=(5, 15), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.01,
                clip=True,
            ),
        )
    else:
        transform = A.Compose(
            [
                A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.01,
                clip=True,
            ),
        )

    return transform


def get_default_val_transform(img_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """
    Tạo transform mặc định cho validation (không augmentation)

    Args:
        img_size: (height, width)
    """
    h, w = img_size

    transform = A.Compose(
        [
            A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.01,
            clip=True,
        ),
    )

    return transform


if __name__ == "__main__":
    # Example usage
    from src.config import CLASSES, CLASSES_10_CLASS

    # Tạo train dataset với các tính năng mới
    train_transform = get_default_train_transform(img_size=(640, 640))
    train_dataset = YoloDataset(
        img_dir="processed/knee/images",
        label_dir="processed/knee/labels",
        transform=train_transform,
        split_file="splits/train.txt",
        use_labels_10_class=False,
        filter_no_label=True,
        cache_images=False,
        cache_labels=True,
        img_size=None,  # Resize trong transform
        bbox_format="pascal_voc",
        return_dict=False,  # Tương thích với code cũ
    )

    # Tạo val dataset
    val_transform = get_default_val_transform(img_size=(640, 640))
    val_dataset = YoloDataset(
        img_dir="processed/knee/images",
        label_dir="processed/knee/labels",
        transform=val_transform,
        split_file="splits/val.txt",
        use_labels_10_class=False,
        filter_no_label=True,
        cache_images=False,
        cache_labels=True,
        img_size=None,
        bbox_format="pascal_voc",
        return_dict=False,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Test một sample (tương thích với code cũ)
    image, boxes, labels = train_dataset[0]
    print(f"\nImage shape: {image.shape}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test với dict format
    train_dataset_dict = YoloDataset(
        img_dir="processed/knee/images",
        label_dir="processed/knee/labels",
        transform=train_transform,
        split_file="splits/train.txt",
        return_dict=True,
    )
    sample_dict = train_dataset_dict[0]
    print(f"\nDict format keys: {sample_dict.keys()}")
    print(f"Image ID: {sample_dict['image_id']}")
