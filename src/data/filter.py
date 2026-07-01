import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from src.data.utils import parse_yolo_label

class DatasetFilter:
    def __init__(self, threshold_percent: float):
        self.threshold = threshold_percent
        
    def identify_rare_classes(self, class_counts: Dict[int, int]) -> Tuple[Set[int], Set[int]]:
        """Identify rare and valid classes."""
        total = sum(class_counts.values())
        rare = set()
        valid = set()
        
        print(f"\n📊 Class Analysis (Threshold: {self.threshold}%)")
        for cls, count in sorted(class_counts.items()):
            pct = (count / total) * 100
            if pct < self.threshold:
                rare.add(cls)
                print(f"❌ Class {cls}: {count} ({pct:.2f}%) - FILTERED")
            else:
                valid.add(cls)
                print(f"✅ Class {cls}: {count} ({pct:.2f}%) - KEPT")
        return rare, valid

    def filter_dataset(
        self,
        img_dir: Path,
        label_dir: Path,
        out_img_dir: Path,
        out_label_dir: Path,
        rare_classes: Set[int],
        remove_rare_boxes: bool = False
    ) -> Dict:
        """Filter dataset and copy to new location."""
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        
        label_files = list(label_dir.glob("*.txt"))
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        stats = {
            "total_images": 0, "copied_images": 0, "filtered_images": 0,
            "total_boxes_original": 0, "total_boxes_filtered": 0, "boxes_removed": 0
        }
        
        for label_file in tqdm(label_files, desc="Filtering"):
            stats["total_images"] += 1
            stem = label_file.stem
            
            # Find image
            img_file = None
            for ext in img_extensions:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    img_file = cand
                    break
            
            if not img_file: 
                continue
                
            # Process labels
            valid_lines = []
            original_box_count = 0
            
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) >= 5:
                        original_box_count += 1
                        cls_id = int(float(parts[0]))
                        
                        if cls_id not in rare_classes:
                            valid_lines.append(line)
                        else:
                            stats["boxes_removed"] += 1
                            
            stats["total_boxes_original"] += original_box_count
            
            should_copy = False
            if remove_rare_boxes:
                # Keep image if valid boxes remain
                if valid_lines:
                    should_copy = True
            else:
                # Keep image only if ALL boxes were valid
                if len(valid_lines) == original_box_count:
                    should_copy = True
                    
            if should_copy:
                shutil.copy2(img_file, out_img_dir / img_file.name)
                with open(out_label_dir / label_file.name, 'w') as f:
                    for l in valid_lines:
                        f.write(l + "\n")
                stats["copied_images"] += 1
                stats["total_boxes_filtered"] += len(valid_lines)
            else:
                stats["filtered_images"] += 1
                
        return stats
