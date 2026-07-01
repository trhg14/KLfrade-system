"""
Visualize YOLO labels on images and save overlays to a folder.

Features:
- Vẽ bounding boxes từ file label YOLO lên ảnh tương ứng
- Hỗ trợ format label chuẩn (class_id) và format mở rộng (class_id + suffix như "3a", "3b")
- Bỏ qua dòng không hợp lệ (empty/malformed)
- Hỗ trợ overlay nhiều thư mục label cùng lúc với màu khác nhau
- Tùy chọn copy ảnh không có label vào thư mục output

Format label được hỗ trợ:
- Chuẩn YOLO: "class_id x y w h" (ví dụ: "3 0.5 0.5 0.1 0.1")
- Mở rộng: "class_id{suffix} x y w h" (ví dụ: "3a 0.5 0.5 0.1 0.1", "3b 0.6 0.6 0.15 0.08")
  → Hữu ích cho label đã được phân tách bởi class_split_report.py

Usage (PowerShell examples):

  # Visualize knee labels trên ảnh knee crops
  python tools/check_dataset/visualize_samples.py `
    --img_dir processed/knee/images `
    --label_dir processed/knee/labels `
    --out_dir check_vis/knee

  # Visualize label mới (đã phân tách a/b) từ labels_10_class
  python tools/check_dataset/visualize_samples.py `
    --img_dir processed/knee/images `
    --label_dir processed/knee/labels_10_class `
    --out_dir check_vis/label_10_class

  # Visualize với nhiều thư mục label (so sánh)
  python tools/check_dataset/visualize_samples.py `
    --img_dir dataset/dataset_v0/images `
    --label_dir dataset/dataset_v0/labels `
    --extra_label_dir dataset/dataset_v0/labels-knee `
    --out_dir check_vis/kl_full
"""

import os
import argparse
import cv2


def parse_color(color_str: str):
    try:
        r, g, b = [int(x) for x in color_str.split(",")]
        return (b, g, r)  # OpenCV uses BGR
    except Exception:
        return (255, 0, 0)


def normalize_label_token(token: str) -> str:
    """
    Chuẩn hóa token label để hiển thị.

    Hỗ trợ cả format chuẩn (số nguyên) và format mở rộng (có suffix như "3a", "3b").
    - Nếu là số: chuyển về số nguyên nếu có thể
    - Nếu không phải số: giữ nguyên (ví dụ: "3a", "3b")

    Args:
        token: Token đầu tiên trong dòng label (class_id hoặc class_id{suffix})

    Returns:
        String đã chuẩn hóa để hiển thị
    """
    try:
        val = float(token)
        if val.is_integer():
            return str(int(val))
        return f"{val:g}"
    except ValueError:
        return token  # Giữ nguyên nếu không phải số (ví dụ: "3a", "3b")


def parse_yolo_line(parts):
    """
    Parse một dòng label YOLO thành các thành phần.

    Format hỗ trợ:
    - Chuẩn: "class_id x y w h"
    - Mở rộng: "class_id{suffix} x y w h" (ví dụ: "3a 0.5 0.5 0.1 0.1")

    Args:
        parts: List các phần tử đã split từ dòng label

    Returns:
        Tuple (label_token, x, y, w, h) nếu hợp lệ, None nếu không hợp lệ
        - label_token: class_id hoặc class_id{suffix} (đã normalize)
        - x, y, w, h: Tọa độ YOLO normalized (0.0 - 1.0)
    """
    if len(parts) < 5:
        return None
    label_token = normalize_label_token(parts[0])
    try:
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None
    # Validate: tọa độ phải trong [0, 1], w và h phải > 0
    if not (
        (0.0 <= x <= 1.0)
        and (0.0 <= y <= 1.0)
        and (0.0 < w <= 1.0)
        and (0.0 < h <= 1.0)
    ):
        return None
    return label_token, x, y, w, h


def draw_box(img, label, x, y, w, h, color=(255, 0, 0), thickness=2, font_scale=0.5):
    H, W = img.shape[:2]
    cx, cy = x * W, y * H
    bw, bh = w * W, h * H
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    # clamp
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 2, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize YOLO labels on images and save overlays"
    )
    p.add_argument(
        "--img_dir", required=True, help="Directory with images (.jpg/.png/.jpeg)"
    )
    p.add_argument(
        "--label_dir",
        required=True,
        help="Directory with YOLO label .txt files (primary)",
    )
    p.add_argument("--out_dir", required=True, help="Directory to save visualizations")
    p.add_argument(
        "--color", default="255,0,0", help="RGB color for boxes, e.g., '255,0,0'"
    )
    p.add_argument("--thickness", type=int, default=2)
    p.add_argument("--font_scale", type=float, default=0.5)
    p.add_argument(
        "--copy_missing",
        action="store_true",
        help="Also save images without labels into out_dir",
    )
    p.add_argument(
        "--extra_label_dir",
        action="append",
        default=None,
        help="Additional label directory (repeatable) to overlay (e.g., knee labels)",
    )
    p.add_argument(
        "--extra_color",
        action="append",
        default=None,
        help="RGB color for each extra_label_dir (repeatable), e.g., '0,255,0'",
    )
    return p.parse_args()


def main():
    args = parse_args()
    color = parse_color(args.color)
    os.makedirs(args.out_dir, exist_ok=True)

    img_files = [
        f
        for f in os.listdir(args.img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    img_files.sort()

    n_with = 0
    n_without = 0

    # Prepare extra label dirs and colors
    extra_dirs = args.extra_label_dir or []
    extra_colors = []
    if args.extra_color:
        for c in args.extra_color:
            extra_colors.append(parse_color(c))
    # default colors for extras if not provided
    default_palette = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    while len(extra_colors) < len(extra_dirs):
        extra_colors.append(
            default_palette[min(len(extra_colors), len(default_palette) - 1)]
        )

    for fn in img_files:
        base = os.path.splitext(fn)[0]
        img_path = os.path.join(args.img_dir, fn)
        lab_path = os.path.join(args.label_dir, base + ".txt")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        had_label = False
        # draw primary label dir
        if os.path.exists(lab_path):
            try:
                with open(lab_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parsed = parse_yolo_line(line.split())
                        if parsed is None:
                            continue
                        label_token, x, y, w, h = parsed
                        draw_box(
                            img,
                            label_token,
                            x,
                            y,
                            w,
                            h,
                            color=color,
                            thickness=args.thickness,
                            font_scale=args.font_scale,
                        )
                        had_label = True
            except Exception:
                pass

        # draw extras
        for ed_idx, ed in enumerate(extra_dirs):
            elab_path = os.path.join(ed, base + ".txt")
            if not os.path.exists(elab_path):
                continue
            try:
                with open(elab_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parsed = parse_yolo_line(line.split())
                        if parsed is None:
                            continue
                        label_token, x, y, w, h = parsed
                        draw_box(
                            img,
                            label_token,
                            x,
                            y,
                            w,
                            h,
                            color=extra_colors[ed_idx],
                            thickness=args.thickness,
                            font_scale=args.font_scale,
                        )
                        had_label = True
            except Exception:
                pass

        out_path = os.path.join(args.out_dir, fn)
        if had_label or args.copy_missing:
            cv2.imwrite(out_path, img)
        if had_label:
            n_with += 1
        else:
            n_without += 1

    print(f"Saved visualizations to: {args.out_dir}")
    print(f"Images with labels: {n_with} | without labels: {n_without}")


if __name__ == "__main__":
    main()
