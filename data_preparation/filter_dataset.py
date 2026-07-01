#!/usr/bin/env python3
"""
Filter dataset by class distribution threshold.
Refactored to use src.data modules.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Ensure project root in python path
sys.path.append(os.getcwd())

from src.data.filter import DatasetFilter
from src.data.dataset_utils import parse_yolo_label

def load_class_distribution(label_dir: Path):
    from collections import Counter
    counts = Counter()
    for f in label_dir.glob("*.txt"):
        classes = parse_yolo_label(f)
        for c in classes:
            counts[c] += 1
    return dict(counts)

def main():
    parser = argparse.ArgumentParser(description="Filter dataset by rare classes")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--output_img_dir", type=str, required=True)
    parser.add_argument("--output_label_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--analysis_json", type=str)
    parser.add_argument("--remove_rare_from_labels", action="store_true")
    
    args = parser.parse_args()
    
    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_img = Path(args.output_img_dir)
    out_lbl = Path(args.output_label_dir)
    
    # Load Distribution
    if args.analysis_json:
        with open(args.analysis_json, "r") as f:
            analysis = json.load(f)
            class_counts = {int(k): v for k, v in analysis["class_distribution"]["class_counts"].items()}
    else:
        # We need counts of instances (all boxes), not just images. 
        # Re-implementing simplified counter here or using src.data logic if expanded.
        # For now, sticking to simple count.
        from collections import Counter
        class_counts = Counter()
        for f in label_dir.glob("*.txt"):
             # Original script counted boxes, parse_yolo_label returns unique set per image?
             # Actually parse_yolo_label returns set. We need total count.
             with open(f, 'r') as fl:
                 for line in fl:
                     parts = line.strip().split()
                     if len(parts) >= 5:
                         class_counts[int(float(parts[0]))] += 1
        class_counts = dict(class_counts)

    # Filter
    processor = DatasetFilter(args.threshold)
    rare, valid = processor.identify_rare_classes(class_counts)
    
    if not rare:
        print("No rare classes found.")
        return
        
    stats = processor.filter_dataset(
        img_dir, label_dir, out_img, out_lbl, rare, args.remove_rare_from_labels
    )
    
    print("Filtering Complete!")
    print(stats)

if __name__ == "__main__":
    main()
