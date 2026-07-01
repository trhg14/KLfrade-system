"""
Shared dataset visualization utilities.

These are COMMON visualization functions that can be reused across
different tools and scripts for dataset analysis.

NOT model-specific - works with any YOLO-format dataset.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional


def draw_yolo_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    class_names: List[str],
    colors: Optional[List] = None,
) -> np.ndarray:
    """
    Draw YOLO format bounding boxes on image.

    Args:
        image: Image array (H, W, 3)
        boxes: List of [class_id, x_center, y_center, width, height] (normalized 0-1)
        class_names: List of class names
        colors: Optional list of colors, one per class

    Returns:
        Image with boxes drawn
    """
    img = image.copy()
    h, w = img.shape[:2]

    if colors is None:
        # Default colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        colors = (colors[:, :3] * 255).astype(int)

    for box in boxes:
        class_id = int(box[0])
        x_center, y_center, box_w, box_h = box[1:5]

        # Convert to pixel coordinates
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # Draw box
        color = tuple(int(c) for c in colors[class_id])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"Class {class_id}"
        )
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def plot_class_distribution(
    class_counts: dict, save_path: Path, title: str = "Class Distribution"
):
    """
    Plot class distribution bar chart.

    Args:
        class_counts: Dict mapping class_name -> count
        save_path: Path to save plot
        title: Plot title
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(classes)), counts, edgecolor="black", linewidth=1.5)

    # Color bars
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add count labels
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({count/total*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_image_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Create a grid of images.

    Args:
        images: List of images
        titles: Optional titles for each image
        grid_size: (rows, cols), auto-calculated if None
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_images = len(images)

    if grid_size is None:
        # Auto-calculate grid
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (ax, img) in enumerate(zip(axes, images)):
        if img.shape[-1] == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")

        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        ax.axis("off")

    # Hide unused axes
    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def plot_box_size_distribution(
    boxes_data: List[Tuple[float, float]],
    save_path: Path,
    title: str = "Bounding Box Size Distribution",
):
    """
    Plot distribution of bounding box sizes.

    Args:
        boxes_data: List of (width, height) tuples (normalized 0-1)
        save_path: Path to save plot
        title: Plot title
    """
    widths = [w for w, h in boxes_data]
    heights = [h for w, h in boxes_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Width distribution
    ax1.hist(widths, bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Width (normalized)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Width Distribution", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Height distribution
    ax2.hist(heights, bins=50, edgecolor="black", alpha=0.7, color="coral")
    ax2.set_xlabel("Height (normalized)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Height Distribution", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compare_distributions(
    data_dict: dict,
    save_path: Path,
    title: str = "Distribution Comparison",
    ylabel: str = "Count",
):
    """
    Compare distributions across multiple datasets/variants.

    Args:
        data_dict: Dict mapping variant_name -> class_counts_dict
        save_path: Path to save plot
        title: Plot title
        ylabel: Y-axis label
    """
    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame(data_dict).fillna(0)

    # Plot
    ax = df.plot(kind="bar", figsize=(12, 6), edgecolor="black", linewidth=1.5)

    plt.xlabel("Class", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(title="Dataset Variant", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
