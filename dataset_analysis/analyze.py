"""
Comprehensive Dataset Analysis Tool

Unified script to analyze YOLO dataset:
- Image statistics (count, dimensions, formats)
- Label statistics for all label directories
- Class distribution and balance
- Bounding box statistics
- Data quality checks
- Visualization samples
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from collections import Counter, defaultdict
import json
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CLASSES, CLASSES_10_CLASS


def analyze_images(img_dir: Path):
    """Analyze images in directory."""
    print("=" * 80)
    print("IMAGE ANALYSIS")
    print("=" * 80)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

    if not images:
        print("❌ No images found!")
        return {}

    print(f"\n📊 Found {len(images)} images")

    # Analyze dimensions and formats
    dimensions = []
    formats = Counter()
    sizes_mb = []

    for img_path in tqdm(images[:100], desc="Sampling images"):  # Sample first 100
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            dimensions.append((h, w))
            formats[img_path.suffix.lower()] += 1
            sizes_mb.append(img_path.stat().st_size / (1024 * 1024))

    stats = {
        "total_images": len(images),
        "formats": dict(formats),
        "dimensions": {
            "unique": len(set(dimensions)),
            "most_common": Counter(dimensions).most_common(5),
            "sample": dimensions[:10],
        },
        "file_sizes_mb": {
            "min": min(sizes_mb) if sizes_mb else 0,
            "max": max(sizes_mb) if sizes_mb else 0,
            "mean": np.mean(sizes_mb) if sizes_mb else 0,
        },
    }

    print(f"\n📐 Dimensions:")
    print(f"  Unique sizes: {stats['dimensions']['unique']}")
    print(f"  Most common:")
    for (h, w), count in stats["dimensions"]["most_common"]:
        print(f"    {w}x{h}: {count} images")

    print(f"\n📁 Formats: {stats['formats']}")
    print(
        f"\n💾 File sizes: {stats['file_sizes_mb']['min']:.2f} - {stats['file_sizes_mb']['max']:.2f} MB (avg: {stats['file_sizes_mb']['mean']:.2f} MB)"
    )

    return stats


def analyze_labels(label_dir: Path, label_name: str):
    """Analyze labels in directory."""
    print("\n" + "=" * 80)
    print(f"LABEL ANALYSIS: {label_name}")
    print("=" * 80)

    if not label_dir.exists():
        print(f"❌ Directory not found: {label_dir}")
        return {}

    label_files = list(label_dir.glob("*.txt"))

    if not label_files:
        print("❌ No label files found!")
        return {}

    print(f"\n📊 Found {len(label_files)} label files")

    # Statistics
    class_counts = Counter()
    bbox_stats = defaultdict(list)  # width, height per class
    images_per_class = defaultdict(set)
    boxes_per_image = []
    empty_files = []
    invalid_files = []

    for label_file in tqdm(label_files, desc="Analyzing labels"):
        try:
            with open(label_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                empty_files.append(label_file.name)
                boxes_per_image.append(0)
                continue

            image_boxes = 0
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        class_id = int(float(parts[0]))
                        x_center, y_center, width, height = map(float, parts[1:5])

                        class_counts[class_id] += 1
                        bbox_stats[class_id].append((width, height))
                        images_per_class[class_id].add(label_file.stem)
                        image_boxes += 1
                    except ValueError:
                        invalid_files.append(f"{label_file.name}: {line}")

            boxes_per_image.append(image_boxes)

        except Exception as e:
            invalid_files.append(f"{label_file.name}: {str(e)}")

    # Compile stats
    stats = {
        "total_files": len(label_files),
        "empty_files": len(empty_files),
        "invalid_files": len(invalid_files),
        "total_boxes": sum(class_counts.values()),
        "classes": dict(class_counts),
        "images_per_class": {k: len(v) for k, v in images_per_class.items()},
        "boxes_per_image": {
            "min": min(boxes_per_image) if boxes_per_image else 0,
            "max": max(boxes_per_image) if boxes_per_image else 0,
            "mean": np.mean(boxes_per_image) if boxes_per_image else 0,
            "median": np.median(boxes_per_image) if boxes_per_image else 0,
        },
        "bbox_stats": bbox_stats,  # For bbox analysis visualization
    }

    # Print summary
    print(f"\n📦 Total bounding boxes: {stats['total_boxes']}")
    print(f"\n📊 Class Distribution:")
    total = stats["total_boxes"]
    for class_id in sorted(stats["classes"].keys()):
        count = stats["classes"][class_id]
        images = stats["images_per_class"][class_id]
        percentage = (count / total * 100) if total > 0 else 0
        print(
            f"  Class {class_id}: {count:>6} boxes ({percentage:>5.2f}%) in {images:>4} images"
        )

    print(f"\n📦 Boxes per image:")
    print(f"  Min: {stats['boxes_per_image']['min']}")
    print(f"  Max: {stats['boxes_per_image']['max']}")
    print(f"  Mean: {stats['boxes_per_image']['mean']:.2f}")
    print(f"  Median: {stats['boxes_per_image']['median']:.2f}")

    # Bbox size statistics
    print(f"\n📏 Bounding Box Sizes (normalized):")
    for class_id in sorted(bbox_stats.keys()):
        sizes = bbox_stats[class_id]
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"  Class {class_id}:")
        print(
            f"    Width:  {np.min(widths):.4f} - {np.max(widths):.4f} (mean: {np.mean(widths):.4f})"
        )
        print(
            f"    Height: {np.min(heights):.4f} - {np.max(heights):.4f} (mean: {np.mean(heights):.4f})"
        )

    if empty_files:
        print(f"\n⚠️  Found {len(empty_files)} empty label files")
        if len(empty_files) <= 10:
            for f in empty_files:
                print(f"    {f}")

    if invalid_files:
        print(f"\n❌ Found {len(invalid_files)} invalid entries")
        for entry in invalid_files[:10]:
            print(f"    {entry}")

    return stats


def visualize_class_distribution(all_label_stats, output_dir: Path):
    """Create visualization of class distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare class distributions across label directories
    fig, axes = plt.subplots(
        len(all_label_stats), 1, figsize=(12, 4 * len(all_label_stats))
    )
    if len(all_label_stats) == 1:
        axes = [axes]

    for idx, (label_name, stats) in enumerate(all_label_stats.items()):
        classes = sorted(stats.get("classes", {}).keys())
        counts = [stats["classes"][c] for c in classes]

        axes[idx].bar(classes, counts, color="skyblue", edgecolor="black")
        axes[idx].set_title(
            f"Class Distribution - {label_name}", fontsize=14, fontweight="bold"
        )
        axes[idx].set_xlabel("Class ID")
        axes[idx].set_ylabel("Number of Boxes")
        axes[idx].grid(axis="y", alpha=0.3)

        # Add counts on top of bars
        for i, (c, count) in enumerate(zip(classes, counts)):
            axes[idx].text(c, count, str(count), ha="center", va="bottom")

    plt.tight_layout()
    viz_path = output_dir / "class_distribution.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Saved class distribution: {viz_path}")
    plt.close()


def visualize_bbox_analysis(all_label_stats, output_dir: Path, label_dir: Path):
    """Create comprehensive bounding box analysis visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get main label stats (usually 'labels')
    main_stats = None
    for name in ["labels", "labels_10_class", "labels_4_class"]:
        if name in all_label_stats and all_label_stats[name].get("bbox_stats"):
            main_stats = all_label_stats[name]
            break

    if not main_stats or "bbox_stats" not in main_stats:
        print("\n⚠️  No bbox stats available for visualization")
        return

    bbox_stats = main_stats["bbox_stats"]

    # Collect all bbox data
    all_widths = []
    all_heights = []
    all_areas = []
    all_aspects = []
    all_x_centers = []
    all_y_centers = []
    boxes_per_image_list = []

    # Read label files to get centers and boxes per image
    if label_dir.exists():
        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                boxes_per_image_list.append(len(lines))
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        all_widths.append(width)
                        all_heights.append(height)
                        all_areas.append(width * height)
                        all_aspects.append(width / height if height > 0 else 0)
                        all_x_centers.append(x_center)
                        all_y_centers.append(y_center)
            except:
                continue

    # Create 8-panel visualization like bbox_analysis.png
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Width Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_widths, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.set_title("Width Distribution", fontweight="bold")
    ax1.set_xlabel("Normalized Width")
    ax1.set_ylabel("Count")
    ax1.axvline(
        np.mean(all_widths),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_widths):.3f}",
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Height Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_heights, bins=50, color="coral", edgecolor="black", alpha=0.7)
    ax2.set_title("Height Distribution", fontweight="bold")
    ax2.set_xlabel("Normalized Height")
    ax2.set_ylabel("Count")
    ax2.axvline(
        np.mean(all_heights),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_heights):.3f}",
    )
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Area Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(all_areas, bins=50, color="mediumseagreen", edgecolor="black", alpha=0.7)
    ax3.set_title("Area Distribution", fontweight="bold")
    ax3.set_xlabel("Normalized Area (Width × Height)")
    ax3.set_ylabel("Count")
    ax3.axvline(
        np.mean(all_areas),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_areas):.3f}",
    )
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Aspect Ratio Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(all_aspects, bins=50, color="mediumpurple", edgecolor="black", alpha=0.7)
    ax4.set_title(
        "Aspect Ratio Distribution (for Anchor Box Optimization)", fontweight="bold"
    )
    ax4.set_xlabel("Aspect Ratio (Width/Height)")
    ax4.set_ylabel("Count")
    ax4.axvline(
        np.mean(all_aspects),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(all_aspects):.3f}",
    )
    ax4.axvline(
        np.median(all_aspects),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(all_aspects):.3f}",
    )
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. PTH (Objects per Image) - changed name to match image
    ax5 = fig.add_subplot(gs[1, 1])
    single_count = sum(1 for x in boxes_per_image_list if x == 1)
    multi_count = sum(1 for x in boxes_per_image_list if x > 1)
    categories = ["Single", "Multiple"]
    counts = [single_count, multi_count]
    colors = ["gold", "orange"]
    bars = ax5.bar(categories, counts, color=colors, edgecolor="black")
    ax5.set_title("Object Bbox Categories\n(% of Image Type)", fontweight="bold")
    ax5.set_ylabel("Number of Images")
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax5.grid(axis="y", alpha=0.3)

    # 6. Width vs Height Scatter (colored by Area)
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(
        all_widths,
        all_heights,
        c=all_areas,
        cmap="viridis",
        alpha=0.6,
        s=20,
        edgecolors="black",
        linewidth=0.5,
    )
    ax6.set_title("Width vs Height (colored by Area)", fontweight="bold")
    ax6.set_xlabel("Normalized Width")
    ax6.set_ylabel("Normalized Height")
    # Add diagonal line (square boxes)
    max_val = max(max(all_widths), max(all_heights))
    ax6.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Square (1:1)")
    ax6.legend()
    ax6.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label("Area", rotation=270, labelpad=15)

    # 7. Object Center Heatmap
    ax7 = fig.add_subplot(gs[2, :])
    heatmap, xedges, yedges = np.histogram2d(all_x_centers, all_y_centers, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax7.imshow(heatmap.T, extent=extent, origin="lower", cmap="hot", aspect="auto")
    ax7.set_title("Object Center Heatmap", fontweight="bold", fontsize=14)
    ax7.set_xlabel("X Center (normalized)")
    ax7.set_ylabel("Y Center (normalized)")
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.grid(True, alpha=0.3, color="white", linestyle="--")
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label("Density", rotation=270, labelpad=15)

    fig.suptitle("Bounding Box Analysis", fontsize=16, fontweight="bold", y=0.995)

    viz_path = output_dir / "bbox_analysis.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    print(f"📊 Saved bbox analysis: {viz_path}")
    plt.close()


def save_analysis_report(img_stats, all_label_stats, output_dir: Path):
    """Save comprehensive analysis reports - both summary and detailed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report = {"images": img_stats, "labels": all_label_stats}

    json_path = output_dir / "analysis_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved JSON report: {json_path}")

    # Summary Markdown report
    md_path = output_dir / "ANALYSIS_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Analysis Report\n\n")

        f.write("## Image Statistics\n\n")
        f.write(f"- **Total Images**: {img_stats.get('total_images', 0)}\n")
        f.write(f"- **Formats**: {img_stats.get('formats', {})}\n")
        f.write(
            f"- **Unique Dimensions**: {img_stats.get('dimensions', {}).get('unique', 0)}\n\n"
        )

        f.write("## Label Statistics\n\n")
        for label_name, stats in all_label_stats.items():
            f.write(f"### {label_name}\n\n")
            f.write(f"- **Total Files**: {stats.get('total_files', 0)}\n")
            f.write(f"- **Total Boxes**: {stats.get('total_boxes', 0)}\n")
            f.write(f"- **Empty Files**: {stats.get('empty_files', 0)}\n")
            f.write(f"- **Invalid Files**: {stats.get('invalid_files', 0)}\n\n")

            f.write("#### Class Distribution\n\n")
            f.write("| Class | Boxes | Images | Percentage |\n")
            f.write("|-------|-------|--------|------------|\n")
            total = stats.get("total_boxes", 1)
            for class_id in sorted(stats.get("classes", {}).keys()):
                count = stats["classes"][class_id]
                images = stats["images_per_class"][class_id]
                pct = (count / total * 100) if total > 0 else 0
                f.write(f"| {class_id} | {count} | {images} | {pct:.2f}% |\n")
            f.write("\n")

    print(f"📄 Saved Summary report: {md_path}")

    # DETAILED Markdown report
    detailed_path = output_dir / "DETAILED_ANALYSIS_REPORT.md"
    with open(detailed_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Dataset Analysis Report\n\n")
        f.write("*Auto-generated by comprehensive_analysis.py*\n\n")
        f.write("---\n\n")

        # Image Statistics
        f.write("## 📸 Image Statistics\n\n")
        f.write(f"- **Total Images**: {img_stats.get('total_images', 0):,}\n")
        formats_str = ", ".join(
            f"`{k}` ({v}%)" for k, v in img_stats.get("formats", {}).items()
        )
        f.write(f"- **Formats**: {formats_str}\n")
        f.write(
            f"- **Unique Dimensions**: {img_stats.get('dimensions', {}).get('unique', 0)} different sizes\n\n"
        )

        f.write("### Image Dimensions\n\n")
        f.write("**Most Common Sizes:**\n\n")
        for idx, ((h, w), count) in enumerate(
            img_stats.get("dimensions", {}).get("most_common", [])
        ):
            marker = " (most common)" if idx == 0 else ""
            f.write(f"- `{w}x{h}`: {count} images{marker}\n")

        f.write(f"\n### File Sizes\n\n")
        sizes = img_stats.get("file_sizes_mb", {})
        f.write(f"- **Min**: {sizes.get('min', 0):.2f} MB\n")
        f.write(f"- **Max**: {sizes.get('max', 0):.2f} MB\n")
        f.write(f"- **Average**: {sizes.get('mean', 0):.2f} MB\n\n")

        f.write("---\n\n")

        # Label Statistics
        f.write("## 🏷️ Label Statistics\n\n")

        for label_name, stats in all_label_stats.items():
            f.write(f"### 📂 {label_name}\n\n")
            f.write(f"- **Total Files**: {stats.get('total_files', 0):,}\n")
            f.write(f"- **Total Bounding Boxes**: {stats.get('total_boxes', 0):,}\n")
            f.write(f"- **Empty Files**: {stats.get('empty_files', 0)}\n")
            f.write(f"- **Invalid Files**: {stats.get('invalid_files', 0)}\n\n")

            # Class Distribution
            f.write("#### Class Distribution\n\n")
            f.write("| Class ID | Boxes | Images | Percentage |\n")
            f.write("|----------|-------|--------|------------|\n")
            total = stats.get("total_boxes", 1)
            for class_id in sorted(stats.get("classes", {}).keys()):
                count = stats["classes"][class_id]
                images = stats["images_per_class"][class_id]
                pct = (count / total * 100) if total > 0 else 0
                highlight = " ⭐" if pct > 30 else ""
                f.write(
                    f"| {class_id} | {count:,} | {images:,} | **{pct:.2f}%**{highlight} |\n"
                )
            f.write("\n")

            # Boxes per Image
            f.write("#### Boxes per Image\n\n")
            bpi = stats.get("boxes_per_image", {})
            f.write(f"- **Min**: {bpi.get('min', 0)}\n")
            f.write(f"- **Max**: {bpi.get('max', 0)}\n")
            f.write(f"- **Mean**: {bpi.get('mean', 0):.2f}\n")
            f.write(f"- **Median**: {bpi.get('median', 0):.2f}\n\n")

            f.write("---\n\n")

        # Key Insights
        f.write("## 💡 Key Insights\n\n")

        if "labels" in all_label_stats:
            main_stats = all_label_stats["labels"]
            total = main_stats.get("total_boxes", 0)
            classes = main_stats.get("classes", {})

            if classes:
                most_common_class = max(classes.items(), key=lambda x: x[1])
                least_common_class = min(classes.items(), key=lambda x: x[1])

                f.write("### Class Balance (labels directory)\n\n")
                f.write(
                    f"- **Most frequent class**: Class {most_common_class[0]} with {most_common_class[1]:,} boxes ({most_common_class[1]/total*100:.2f}%)\n"
                )
                f.write(
                    f"- **Least frequent class**: Class {least_common_class[0]} with {least_common_class[1]:,} boxes ({least_common_class[1]/total*100:.2f}%)\n"
                )
                imbalance_ratio = most_common_class[1] / least_common_class[1]
                f.write(f"- **Imbalance ratio**: {imbalance_ratio:.2f}x\n\n")

                if imbalance_ratio > 5:
                    f.write(
                        f"⚠️ **Warning**: Significant class imbalance detected (>{imbalance_ratio:.1f}x difference). Consider:\n"
                    )
                    f.write("- Data augmentation for minority classes\n")
                    f.write("- Class weighting during training\n")
                    f.write("- Sampling strategies (oversampling/undersampling)\n")
                    f.write("- Focal loss or similar techniques\n\n")

        f.write("### Dataset Quality\n\n")
        all_empty = sum(s.get("empty_files", 0) for s in all_label_stats.values())
        all_invalid = sum(s.get("invalid_files", 0) for s in all_label_stats.values())

        if all_empty == 0 and all_invalid == 0:
            f.write("✅ **No data quality issues detected**\n")
            f.write("- No empty label files\n")
            f.write("- No invalid label files\n")
            f.write("- All bounding boxes within valid range\n\n")
        else:
            if all_empty > 0:
                f.write(f"⚠️ **Empty label files**: {all_empty}\n")
            if all_invalid > 0:
                f.write(f"❌ **Invalid label files**: {all_invalid}\n")
            f.write("\n")

        # Label type comparison
        if len(all_label_stats) > 1:
            f.write("### Label Types Comparison\n\n")
            f.write("| Label Type | Files | Boxes | Boxes/Image | Purpose |\n")
            f.write("|------------|-------|-------|-------------|---------||\n")
            purposes = {
                "labels": "KL grading (0-4)",
                "labels-knee": "Knee detection",
                "labels-knee-2box": "Left/Right knee",
            }
            for name, stats in all_label_stats.items():
                files = stats.get("total_files", 0)
                boxes = stats.get("total_boxes", 0)
                bpi = stats.get("boxes_per_image", {}).get("mean", 0)
                purpose = purposes.get(name, "Unknown")
                highlight = "**" if name == "labels" else ""
                f.write(
                    f"| {highlight}{name}{highlight} | {files:,} | {boxes:,} | {bpi:.2f} | {purpose} |\n"
                )
            f.write("\n")

        # Recommendations
        f.write("### Recommendations\n\n")
        f.write(
            "1. **Use `labels/` for main training** - Contains KL grade annotations (0-4)\n"
        )
        if "labels-knee" in all_label_stats:
            f.write("2. **Consider 2-stage approach**:\n")
            f.write("   - Stage 1: Use `labels-knee/` for knee detection\n")
            f.write(
                "   - Stage 2: Use `labels/` for KL grading within detected region\n"
            )

        if "labels" in all_label_stats and all_label_stats["labels"].get("classes"):
            classes = all_label_stats["labels"]["classes"]
            most_common = max(classes.items(), key=lambda x: x[1])
            least_common = min(classes.items(), key=lambda x: x[1])
            if most_common[1] / least_common[1] > 5:
                f.write("3. **Address class imbalance**:\n")
                for class_id, count in sorted(classes.items()):
                    total_boxes = all_label_stats["labels"]["total_boxes"]
                    pct = count / total_boxes * 100
                    if pct < 10:
                        f.write(
                            f"   - Class {class_id}: Very few samples ({count}) - augment heavily\n"
                        )
                    elif pct > 40:
                        f.write(
                            f"   - Class {class_id}: Dominant class ({pct:.1f}%) - may need undersampling\n"
                        )
                f.write("\n")

        f.write("4. **Data Split Strategy**:\n")
        f.write("   - Stratified split to maintain class distribution\n")
        f.write("   - Ensure minority classes present in all splits\n")
        f.write("   - Recommended: 70% train, 15% val, 15% test\n\n")

        f.write("---\n\n")
        f.write("## 📊 Visualizations\n\n")
        f.write(
            "See `class_distribution.png` for visual comparison of class distributions.\n\n"
        )
        f.write("---\n\n")
        f.write("*Report generated by comprehensive_analysis.py*\n")

    print(f"📄 Saved Detailed report: {detailed_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive dataset analysis")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--output", type=str, default="dataset_analysis", help="Output directory"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: {dataset_dir}")
    print(f"Output: {output_dir}")

    # Analyze images
    img_dir = dataset_dir / "images"
    img_stats = analyze_images(img_dir)

    # Analyze all label directories
    all_label_stats = {}
    for label_dir_name in ["labels", "labels-knee", "labels-knee-2box"]:
        label_dir = dataset_dir / label_dir_name
        if label_dir.exists():
            stats = analyze_labels(label_dir, label_dir_name)
            if stats:
                all_label_stats[label_dir_name] = stats

    # Visualizations
    if all_label_stats:
        visualize_class_distribution(all_label_stats, output_dir)

        # Add bbox analysis visualization
        for label_dir_name in ["labels", "labels_10_class", "labels_4_class"]:
            label_dir = dataset_dir / label_dir_name
            if label_dir.exists():
                visualize_bbox_analysis(all_label_stats, output_dir, label_dir)
                break

    # Save reports
    save_analysis_report(img_stats, all_label_stats, output_dir)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
