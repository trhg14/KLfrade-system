from pathlib import Path
from typing import List, Set

def get_image_files(img_dir: Path) -> List[Path]:
    """Get all image files in a directory."""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return [f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions]

def parse_yolo_label(label_file: Path) -> Set[int]:
    """Parse YOLO format label file and return set of class IDs."""
    classes = set()
    if not label_file.exists():
        return classes
        
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    class_id = int(float(parts[0]))
                    classes.add(class_id)
                except ValueError:
                    continue
    return classes

def write_split_file(stems: List[str], output_path: Path, stem_to_filename: dict, rel_img_dir_str: str):
    """Write list of image paths to a split file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for stem in sorted(stems):
            # Construct relative path: images/filename.jpg
            full_name = stem_to_filename.get(stem, f"{stem}.jpg") # Fallback to jpg
            path_str = f"{rel_img_dir_str}/{full_name}"
            f.write(f"{path_str}\n")

def create_yolo_dataset_yaml(output_path: Path, class_names: dict, path: str, train: str, val: str, test: str = None):
    """Create a YOLO dataset YAML file."""
    content = f"""# YOLO Dataset Config (generated)
path: {path}
train: {train}
val: {val}
"""
    if test:
        content += f"test: {test}\n"
    
    content += "\nnames:\n"
    # class_names can be list or dict
    if isinstance(class_names, dict):
        for k, v in sorted(class_names.items()):
            content += f"  {k}: {v}\n"
    elif isinstance(class_names, list):
        for i, name in enumerate(class_names):
            content += f"  {i}: {name}\n"
            
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
