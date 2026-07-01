"""
Generate pseudo knee bounding boxes (labels-knee/) from 5-class KL lesion labels.

When knee boxes are not provided in the raw export, cluster lesion centers by
horizontal position (left vs right knee) and build a union box per cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


KNEE_CLASS_ID = 0


def load_yolo_boxes(label_path: Path) -> List[Dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        boxes.append(
            {
                "class_id": int(float(parts[0])),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            }
        )
    return boxes


def _box_to_xyxy(box: Dict) -> Tuple[float, float, float, float]:
    x1 = box["x"] - box["w"] / 2
    y1 = box["y"] - box["h"] / 2
    x2 = box["x"] + box["w"] / 2
    y2 = box["y"] + box["h"] / 2
    return x1, y1, x2, y2


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float) -> Dict:
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    w = max(x2 - x1, 0.01)
    h = max(y2 - y1, 0.01)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return {"class_id": KNEE_CLASS_ID, "x": cx, "y": cy, "w": w, "h": h}


def cluster_kl_boxes(kl_boxes: List[Dict]) -> List[List[Dict]]:
    if not kl_boxes:
        return []
    if len(kl_boxes) == 1:
        return [kl_boxes]

    xs = sorted(b["x"] for b in kl_boxes)
    median_x = xs[len(xs) // 2]
    left = [b for b in kl_boxes if b["x"] < median_x]
    right = [b for b in kl_boxes if b["x"] >= median_x]

    if not left or not right:
        return [kl_boxes]
    return [left, right]


def group_to_knee_box(group: List[Dict], padding: float = 0.08) -> Dict:
    x1 = min(_box_to_xyxy(b)[0] for b in group)
    y1 = min(_box_to_xyxy(b)[1] for b in group)
    x2 = max(_box_to_xyxy(b)[2] for b in group)
    y2 = max(_box_to_xyxy(b)[3] for b in group)

    w = x2 - x1
    h = y2 - y1
    x1 -= w * padding
    y1 -= h * padding
    x2 += w * padding
    y2 += h * padding
    return _xyxy_to_yolo(x1, y1, x2, y2)


def save_yolo_labels(boxes: List[Dict], path: Path) -> None:
    if not boxes:
        path.write_text("", encoding="utf-8")
        return
    lines = [
        f"{b['class_id']} {b['x']:.6f} {b['y']:.6f} {b['w']:.6f} {b['h']:.6f}"
        for b in boxes
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_knee_labels(dataset_dir: Path, padding: float = 0.08) -> dict:
    kl_dir = dataset_dir / "labels"
    knee_dir = dataset_dir / "labels-knee"
    knee_dir.mkdir(parents=True, exist_ok=True)

    stats = {"files": 0, "knee_boxes": 0, "empty": 0}

    for label_path in sorted(kl_dir.glob("*.txt")):
        kl_boxes = load_yolo_boxes(label_path)
        groups = cluster_kl_boxes(kl_boxes)
        knee_boxes = [group_to_knee_box(g, padding=padding) for g in groups]
        out_path = knee_dir / label_path.name
        save_yolo_labels(knee_boxes, out_path)
        stats["files"] += 1
        stats["knee_boxes"] += len(knee_boxes)
        if not knee_boxes:
            stats["empty"] += 1

    report = dataset_dir / "knee_labels_generated_report.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/dataset_v0"),
    )
    parser.add_argument("--padding", type=float, default=0.08)
    args = parser.parse_args()

    stats = generate_knee_labels(args.dataset_dir, padding=args.padding)
    print("Knee labels generated:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
