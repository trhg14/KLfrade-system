#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import random


def find_top_b_samples(labels_dir: Path, top_k: int = 6):
    """
    Find label files that contain class 'b' boxes (cls % 2 == 1),
    compute max ratio (bw/bh) among b boxes, then return top_k by that ratio.
    """
    files_with_b = []

    for lf in labels_dir.glob("*.txt"):
        with open(lf, "r", encoding="utf-8") as f:
            boxes_ratio = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    ratio = (bw / bh) if bh > 0 else 0.0
                    boxes_ratio.append((cls, ratio))

        b_boxes = [(c, r) for c, r in boxes_ratio if c % 2 == 1]
        if b_boxes:
            files_with_b.append(
                (lf, len(boxes_ratio), len(b_boxes), max(r for _, r in b_boxes))
            )

    files_with_b.sort(key=lambda x: x[3], reverse=True)
    return files_with_b[:top_k], files_with_b


def find_random_samples(labels_dir: Path, top_k: int = 6):
    """
    Randomly select label files and compute stats for consistency with draw_samples.
    """
    all_files = list(labels_dir.glob("*.txt"))
    if not all_files:
        return [], []

    selected_files = random.sample(all_files, min(len(all_files), top_k))
    samples = []

    for lf in selected_files:
        with open(lf, "r", encoding="utf-8") as f:
            boxes_ratio = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    ratio = (bw / bh) if bh > 0 else 0.0
                    boxes_ratio.append((cls, ratio))

        b_boxes = [(c, r) for c, r in boxes_ratio if c % 2 == 1]
        b_cnt = len(b_boxes)
        max_ratio = max(r for _, r in b_boxes) if b_boxes else 0.0

        samples.append((lf, len(boxes_ratio), b_cnt, max_ratio))

    return samples, all_files


def resolve_image_path(images_dir: Path, stem: str):
    jpg = images_dir / f"{stem}.jpg"
    png = images_dir / f"{stem}.png"
    if jpg.exists():
        return jpg
    if png.exists():
        return png
    return None


def load_yolo_boxes(label_file: Path):
    """
    Return list of (cls, x, y, bw, bh) from YOLO label.
    """
    out = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x, y, bw, bh = map(float, parts[1:5])
                out.append((cls, x, y, bw, bh))
    return out


def load_knee_boxes(knee_label_file: Path):
    """
    Return list of (x, y, bw, bh) from YOLO knee label file (class ignored).
    """
    out = []
    if not knee_label_file.exists():
        return out
    with open(knee_label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y, bw, bh = map(float, parts[1:5])
                out.append((x, y, bw, bh))
    return out


def draw_samples(
    samples,
    images_dir: Path,
    knee_labels_dir: Path,
    out_path: Path,
    title: str,
    fig_cols: int = 3,
):
    if not samples:
        print("No samples found.")
        return

    fig_rows = (len(samples) + fig_cols - 1) // fig_cols
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(20, 7 * fig_rows))
    if fig_rows == 1:
        axes = [axes] if fig_cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    # Render each sample
    for idx, (label_file, total, b_cnt, max_ratio) in enumerate(samples):
        img_path = resolve_image_path(images_dir, label_file.stem)
        if img_path is None:
            print(f"[WARN] Image not found for: {label_file.stem}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes = load_yolo_boxes(label_file)

        knee_file = knee_labels_dir / f"{label_file.stem}.txt"
        knee_boxes = load_knee_boxes(knee_file)

        ax = axes[idx]
        ax.imshow(img)
        t_str = (
            f"{label_file.stem[:30]}...\nmax_ratio={max_ratio:.1f} | {b_cnt}b/{total}"
        )
        ax.set_title(t_str, fontsize=10, weight="bold")
        ax.axis("off")

        # Draw knee (lime dashed)
        for x, y, bw, bh in knee_boxes:
            x_pix, y_pix = x * w, y * h
            w_pix, h_pix = bw * w, bh * h
            x1, y1 = x_pix - w_pix / 2, y_pix - h_pix / 2
            rect = patches.Rectangle(
                (x1, y1),
                w_pix,
                h_pix,
                linewidth=2.5,
                edgecolor="lime",
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

        # Draw boxes + label above
        for cls, x, y, bw, bh in boxes:
            x_pix, y_pix = x * w, y * h
            w_pix, h_pix = bw * w, bh * h
            x1, y1 = x_pix - w_pix / 2, y_pix - h_pix / 2

            kl_grade = cls // 2
            suffix = "b" if cls % 2 else "a"
            color = "cyan" if suffix == "b" else "red"
            linewidth = 3 if suffix == "b" else 1.5

            rect = patches.Rectangle(
                (x1, y1),
                w_pix,
                h_pix,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            ratio = (bw / bh) if bh > 0 else 0.0
            label = f"{kl_grade}{suffix}"
            if suffix == "b":
                label += f"\n{ratio:.1f}"

            ax.text(
                x_pix,
                y1 - 8,
                label,
                fontsize=9,
                color="white",
                weight="bold",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9),
            )

    # Hide unused axes
    for j in range(len(samples), len(axes)):
        axes[j].axis("off")

    legend_elements = [
        patches.Patch(
            facecolor="none",
            edgecolor="lime",
            linestyle="--",
            linewidth=2.5,
            label="Knee",
        ),
        patches.Patch(facecolor="red", label="a (Osteophyte)"),
        patches.Patch(facecolor="cyan", label="b (Joint Space)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.99),
    )

    plt.suptitle(
        title,
        fontsize=15,
        weight="bold",
        y=0.97,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO samples (either random or with specific class 'b' criteria)."
    )
    parser.add_argument(
        "--labels_dir", type=str, default="datasets/dataset_v0/labels_10_class_test"
    )
    parser.add_argument("--images_dir", type=str, default="datasets/dataset_v0/images")
    parser.add_argument(
        "--knee_labels_dir", type=str, default="datasets/dataset_v0/labels-knee"
    )
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument(
        "--out", type=str, default="datasets/dataset_v0/samples_viz.png"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, pick random samples instead of top class 'b' samples.",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    knee_labels_dir = Path(args.knee_labels_dir)
    out_path = Path(args.out)

    if args.random:
        print(f"Selecting {args.top_k} random samples...")
        samples, _ = find_random_samples(labels_dir, top_k=args.top_k)
        title = "Random Samples Visualization"
    else:
        print(
            f"Selecting top {args.top_k} samples with Class 'b' (sorted by aspect ratio)..."
        )
        samples, _ = find_top_b_samples(labels_dir, top_k=args.top_k)
        title = 'Top Samples with Wide Boxes (Class "b") - Labels Above Boxes'

    if not samples:
        print("No samples found matching criteria.")
        return

    print("Selected files:")
    for lf, total, b_cnt, max_ratio in samples:
        print(f"  {lf.stem[:35]}: {b_cnt}b/{total} boxes, max_ratio={max_ratio:.2f}")

    draw_samples(samples, images_dir, knee_labels_dir, out_path, title)


if __name__ == "__main__":
    main()
