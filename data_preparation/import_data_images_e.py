"""
Import Data_Images_E archive into datasets/dataset_v0.

Expected zip layout (outer archive):
  Data_Images_E-.../Data_Images_E/Labels_E/*.txt
  Data_Images_E-.../Data_Images_E/Images_E.zip  (nested, contains Images_E/*.jpg)

Output:
  datasets/dataset_v0/images/{stem}.jpg
  datasets/dataset_v0/labels/{stem}.txt
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path


def _find_prefix(names: list[str], needle: str) -> str:
    for name in names:
        if needle in name:
            return name.split(needle)[0] + needle
    raise FileNotFoundError(f"Cannot find '{needle}' in archive")


def import_archive(
    zip_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images_written": 0,
        "labels_written": 0,
        "images_skipped": 0,
        "labels_skipped": 0,
    }

    with zipfile.ZipFile(zip_path) as outer:
        names = outer.namelist()
        labels_prefix = _find_prefix(names, "Labels_E/")
        nested_zip_name = next(n for n in names if n.endswith("Images_E.zip"))

        label_entries = [
            n
            for n in names
            if n.startswith(labels_prefix) and n.endswith(".txt") and not n.endswith("/")
        ]

        for entry in label_entries:
            stem = Path(entry).stem
            out_label = labels_dir / f"{stem}.txt"
            if out_label.exists() and not overwrite:
                stats["labels_skipped"] += 1
                continue
            out_label.write_bytes(outer.read(entry))
            stats["labels_written"] += 1

        nested_data = outer.read(nested_zip_name)
        with zipfile.ZipFile(io.BytesIO(nested_data)) as inner:
            for entry in inner.namelist():
                if not entry.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                stem = Path(entry).stem
                suffix = Path(entry).suffix.lower()
                out_image = images_dir / f"{stem}{suffix}"
                if out_image.exists() and not overwrite:
                    stats["images_skipped"] += 1
                    continue
                out_image.write_bytes(inner.read(entry))
                stats["images_written"] += 1

    # Keep only image/label pairs that exist on both sides
    image_stems = {p.stem for p in images_dir.iterdir()}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    common = image_stems & label_stems
    stats["paired_samples"] = len(common)
    stats["images_without_labels"] = len(image_stems - label_stems)
    stats["labels_without_images"] = len(label_stems - image_stems)

    report_path = output_dir / "import_report.json"
    report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Data_Images_E zip to dataset_v0")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("Data_Images_E-20250426T014033Z-001-20260521T121722Z-3-001.zip"),
        help="Path to outer Data_Images_E zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/dataset_v0"),
        help="Output dataset directory",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.zip.exists():
        raise FileNotFoundError(f"Zip not found: {args.zip}")

    stats = import_archive(args.zip, args.output, overwrite=args.overwrite)
    print("Import complete:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
