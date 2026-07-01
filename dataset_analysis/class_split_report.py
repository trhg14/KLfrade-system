"""
Phân tách class YOLO labels thành nhánh con (a/b) dựa trên đặc điểm hình học.

Quy tắc:
- "a" = gai xương (w/h < 1.2 hoặc area < 0.01)
- "b" = khe khớp (w/h > 2.0 hoặc area > 0.03)
- 2 box cùng class khác loại: box lớn → "b", box nhỏ → "a"
- 2 box cùng loại: giữ nguyên

Output: class_id (0-9) x y w h (format YOLO chuẩn)
Mapping: "0a"→0, "0b"→1, "1a"→2, "1b"→3, "2a"→4, "2b"→5, "3a"→6, "3b"→7, "4a"→8, "4b"→9
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Gán nhãn a/b: a=gai xương, b=khe khớp"
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=None,
        default="processed/knee/labels",
        help="Thư mục chứa file label YOLO gốc (*.txt)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Chỉ xử lý các class cụ thể (ví dụ: --classes 2 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Số lượng ví dụ hiển thị cho mỗi class trong báo cáo",
    )
    parser.add_argument(
        "--max-split", type=int, default=2, help="Số nhánh tối đa (mặc định 2: a/b)"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default="processed/knee/labels_10_class",
        help="Thư mục lưu file label mới với tag a/b",
    )
    parser.add_argument(
        "--knee-labels-dir",
        type=Path,
        default=None,
        help="Thư mục chứa knee bounding boxes (để xử lý ảnh toàn chân)",
    )
    return parser.parse_args()


def iter_label_files(labels_dir: Path) -> Iterable[Path]:
    """Lặp qua file .txt trong thư mục (đã sắp xếp)."""
    for path in sorted(labels_dir.glob("*.txt")):
        if path.is_file():
            yield path


def load_boxes(label_path: Path) -> List[Tuple[int, List[float]]]:
    """Đọc file label YOLO: class_id x y w h (normalized)."""
    boxes = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:5]))
            boxes.append((cls, coords))
    return boxes


def load_knee_boxes(knee_label_path: Path) -> List[Tuple[float, float]]:
    """
    Đọc knee bounding boxes và trả về list các tâm knee.

    Args:
        knee_label_path: Path to knee label file

    Returns:
        List of (center_x, center_y) for each knee
    """
    knee_centers = []
    if not knee_label_path.exists():
        return knee_centers

    with knee_label_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            # coords: x, y, w, h (normalized)
            x, y = float(parts[1]), float(parts[2])
            knee_centers.append((x, y))

    return knee_centers


def assign_to_nearest_knee(
    kl_box_coords: List[float], knee_centers: List[Tuple[float, float]]
) -> int:
    """
    Gán KL box vào knee gần nhất (theo khoảng cách Euclidean).

    Args:
        kl_box_coords: [x, y, w, h] của KL box
        knee_centers: List of (center_x, center_y) for each knee

    Returns:
        Index của knee gần nhất (0, 1, ...)
    """
    import math

    if not knee_centers:
        return 0  # Default to knee 0 if no knee info

    kl_x, kl_y = kl_box_coords[0], kl_box_coords[1]

    min_dist = float("inf")
    nearest_knee = 0

    for knee_idx, (knee_x, knee_y) in enumerate(knee_centers):
        dist = math.sqrt((kl_x - knee_x) ** 2 + (kl_y - knee_y) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest_knee = knee_idx

    return nearest_knee


def classify_box(w: float, h: float, area: float) -> str:
    """
    Phân loại box: "a" (gai xương) hoặc "b" (khe khớp).

    Logic ưu tiên:
    1. ratio > 2.0 → "b" (khe khớp - rất rõ ràng)
    2. ratio < 1.2 → "a" (gai xương - rất rõ ràng)
    3. area > 0.03 → "b" (box lớn)
    4. area < 0.01 → "a" (box nhỏ)
    5. Còn lại → None (không rõ ràng)
    """
    if h <= 0:
        return "a"

    ratio = w / h

    # PRIORITIZE ratio - most reliable indicator
    # Ratio > 2.0 → definitely joint space, even if small area
    if ratio > 2.0:
        return "b"

    # Ratio < 1.2 → definitely osteophyte
    if ratio < 1.2:
        return "a"

    # For moderate ratios (1.2 - 2.0), use area
    if area > 0.03:
        return "b"

    if area < 0.01:
        return "a"

    # Unclear → return None
    return None


def is_at_edge(x: float, y: float, w: float, h: float, threshold: float = 0.01) -> bool:
    """
    Kiểm tra box có nằm ở viền ảnh không.

    Args:
        x, y: Tọa độ center (normalized)
        w, h: Width, height (normalized)
        threshold: Ngưỡng khoảng cách từ viền (mặc định 0.05 = 5%)

    Returns:
        True nếu box gần viền ảnh
    """
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2

    # Kiểm tra gần viền trái, phải, trên, dưới
    return (
        x_min < threshold
        or x_max > 1.0 - threshold
        or y_min < threshold
        or y_max > 1.0 - threshold
    )


def assign_relative_group(boxes: List[dict]):
    """
    Xử lý box không rõ ràng: so sánh width và kiểm tra vị trí viền.

    Logic:
    - Nếu cả 2 box đã có suffix khác nhau → giữ nguyên
    - Nếu cả 2 box có cùng suffix 'a' hoặc 'b' → CHỈ reassign nếu chênh lệch width rõ rệt (>3x)
    - Nếu có ít nhất 1 box None → reassign dựa trên width
    """
    if len(boxes) != 2:
        return

    b1, b2 = boxes

    # If both boxes have different suffixes and both are not None, keep them
    if (
        b1["suffix"] is not None
        and b2["suffix"] is not None
        and b1["suffix"] != b2["suffix"]
    ):
        return  # Already correctly classified

    # Lấy thông tin box
    x1, y1, w1, h1 = b1["coords"]
    x2, y2, w2, h2 = b2["coords"]

    # If both have same non-None suffix (e.g., both 'a'), only reassign if width difference is huge
    if (
        b1["suffix"] is not None
        and b2["suffix"] is not None
        and b1["suffix"] == b2["suffix"]
    ):
        # Both classified the same way - only override if width difference is very significant
        width_ratio = max(w1, w2) / min(w1, w2) if min(w1, w2) > 0 else 1
        if width_ratio < 3.0:  # Not significant enough difference
            return  # Keep both as same type (e.g., both 'a' for two small osteophytes)

    # If at least one is None OR width difference is huge, reassign based on width

    # Kiểm tra box nào có width lớn hơn và không ở viền
    at_edge1 = is_at_edge(x1, y1, w1, h1)
    at_edge2 = is_at_edge(x2, y2, w2, h2)

    # Box có width lớn hơn và không ở viền → "b"
    if w1 > w2 and not at_edge1:
        b1["suffix"] = "b"
        b2["suffix"] = "a"
    elif w2 > w1 and not at_edge2:
        b2["suffix"] = "b"
        b1["suffix"] = "a"
    else:
        # Nếu không thỏa điều kiện trên, xét width đơn giản
        if w1 > w2:
            b1["suffix"] = "b"
            b2["suffix"] = "a"
        else:
            b1["suffix"] = "a"
            b2["suffix"] = "b"


def convert_tag_to_yolo_class(tag: str) -> int:
    """
    Chuyển đổi tag (ví dụ "2a", "3b") sang class_id YOLO (0-9).

    Mapping theo CLASSES_10_CLASS:
    - "0a" -> 0, "0b" -> 1
    - "1a" -> 2, "1b" -> 3
    - "2a" -> 4, "2b" -> 5
    - "3a" -> 6, "3b" -> 7
    - "4a" -> 8, "4b" -> 9

    Args:
        tag: Tag dạng "{class}{suffix}" (ví dụ: "2a", "3b")

    Returns:
        class_id (int) từ 0-9
    """
    import re

    match = re.match(r"^(\d+)([a-z]?)$", tag)
    if match:
        base_class = int(match.group(1))
        suffix = match.group(2) or ""

        # Tính class_id: base_class * 2 + (0 nếu "a", 1 nếu "b")
        if suffix == "a":
            return base_class * 2
        elif suffix == "b":
            return base_class * 2 + 1
        else:
            # Nếu không có suffix, giữ nguyên (nhưng không nên xảy ra)
            return base_class

    # Fallback: thử parse số nguyên
    try:
        return int(tag)
    except ValueError:
        raise ValueError(f"Không parse được tag: {tag}")


def build_reports(labels_dir, allowed_classes, max_split, knee_labels_dir=None):
    """
    Xử lý labels và tạo output.

    Args:
        labels_dir: Thư mục chứa KL labels
        allowed_classes: Chỉ xử lý các class này
        max_split: Số nhánh tối đa
        knee_labels_dir: Thư mục chứa knee boxes (optional, cho ảnh toàn chân)

    Returns:
        (class_entries, warnings, exported_lines)
    """
    class_entries = defaultdict(list)
    warnings = []
    exported_lines = defaultdict(list)

    for label_file in iter_label_files(labels_dir):
        boxes_loaded = load_boxes(label_file)

        # Full-leg images may contain two knees, so the same class can appear twice
        # in one file. Knee boxes let us resolve those cases by side first.
        knee_centers = []
        if knee_labels_dir:
            knee_label_path = knee_labels_dir / label_file.name
            knee_centers = load_knee_boxes(knee_label_path)

        class_map = defaultdict(list)
        for cls, coords in boxes_loaded:
            if allowed_classes and cls not in allowed_classes:
                continue
            class_map[cls].append(coords)

        for cls, coords_list in class_map.items():
            entries = []
            for coords in coords_list:
                x, y, w, h = coords
                area = w * h
                suffix = classify_box(w, h, area)

                entry = {
                    "cls": cls,
                    "coords": coords,
                    "suffix": suffix,
                    "area": area,
                    "ratio": w / h if h > 0 else 0,
                    "file": label_file.name,
                }

                # Assign to knee if knee info is available
                if knee_centers:
                    knee_idx = assign_to_nearest_knee(coords, knee_centers)
                    entry["knee_idx"] = knee_idx

                entries.append(entry)

            # Resolve ambiguity per knee instead of per whole image when side-specific
            # knee boxes are available.
            if knee_centers and len(knee_centers) > 1:
                # Group entries by knee_idx
                knee_groups = defaultdict(list)
                for ent in entries:
                    knee_idx = ent.get("knee_idx", 0)
                    knee_groups[knee_idx].append(ent)

                # Process each knee group separately
                for knee_idx, knee_entries in knee_groups.items():
                    if len(knee_entries) == 2:
                        assign_relative_group(knee_entries)
            else:
                # Original logic: only if exactly 2 boxes in same class
                if len(entries) == 2:
                    assign_relative_group(entries)

            # Đảm bảo tất cả box đều có suffix (fallback nếu vẫn None)
            for ent in entries:
                if ent["suffix"] is None:
                    # Fallback: dựa trên ratio
                    ent["suffix"] = "a" if ent["ratio"] < 1.5 else "b"
                    # Cảnh báo box không rõ ràng
                    warnings.append(
                        f"{label_file.name}: class {cls} box không rõ ràng "
                        f"(ratio={ent['ratio']:.2f}, area={ent['area']:.3f}) "
                        f"-> fallback {ent['suffix']}"
                    )

                # Tạo tag dạng "2a", "3b"
                tag = f"{ent['cls']}{ent['suffix']}"

                # Chuyển đổi sang class_id YOLO chuẩn (0-9)
                yolo_class_id = convert_tag_to_yolo_class(tag)

                class_entries[cls].append(ent)
                # Output format YOLO chuẩn: class_id (số nguyên) x y w h
                out_line = f"{yolo_class_id} " + " ".join(
                    f"{v:.6f}" for v in ent["coords"]
                )
                exported_lines[label_file.name].append(out_line)

    return class_entries, warnings, exported_lines


def print_report(class_entries, warnings, limit):
    """In báo cáo thống kê và cảnh báo."""
    for cls in sorted(class_entries):
        print(f"\nClass {cls}")
        for e in class_entries[cls][:limit]:
            print(
                f"  {e['file']} -> {e['cls']}{e['suffix']} "
                f"(ratio={e['ratio']:.2f}, area={e['area']:.3f})  "
                f"{e['coords']}"
            )

    if warnings:
        print("\n=== CANH BAO ===")
        for w in warnings:
            print("  *", w)


def main():
    args = parse_args()

    class_entries, warnings, exported = build_reports(
        args.labels_dir, args.classes, args.max_split, args.knee_labels_dir
    )

    print_report(class_entries, warnings, args.limit)

    if args.save_dir:
        args.save_dir.mkdir(exist_ok=True, parents=True)
        written = 0
        for fname, lines in exported.items():
            if lines:
                with (args.save_dir / fname).open("w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                written += 1
        print(f"\nDa luu {written} file moi vao {args.save_dir}")


if __name__ == "__main__":
    main()

# python check_dataset\class_split_report.py --labels-dir dataset\dataset_v0\labels --save-dir dataset\dataset_v0\labels_10_class
