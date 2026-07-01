from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO


def build_ultralytics_yaml(cfg: dict, workspace: Path) -> Path:
    """D?ng data.yaml chu?n Ultralytics t? split txt files."""
    workspace.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).parent.parent.parent
    split_dir = root / cfg["split_dir"]

    data = {
        "path": str(root),
        "train": str(split_dir / "train.txt"),
        "val":   str(split_dir / "val.txt"),
        "test":  str(split_dir / "test.txt"),
        "nc":    cfg["nc"],
        "names": cfg["names"],
    }

    out_yaml = workspace / f"{cfg['dataset_name']}.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"Generated data yaml: {out_yaml}")
    return out_yaml


def train(cfg: dict, dry_run: bool = False) -> None:
    root = Path(__file__).parent.parent.parent
    workspace = root / "artifacts/yolo11/workspace" / cfg["dataset_name"]
    data_yaml = build_ultralytics_yaml(cfg, workspace)

    if dry_run:
        print("=== DRY RUN ===")
        print(f"  dataset : {cfg['dataset_name']}")
        print(f"  model   : {cfg['model']}")
        print(f"  epochs  : {cfg['epochs']}")
        print(f"  imgsz   : {cfg['imgsz']}")
        print(f"  batch   : {cfg['batch']}")
        print(f"  data    : {data_yaml}")
        print("Dry run OK - no training performed.")
        return

    model = YOLO(cfg["model"])
    model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device", "0"),
        project=cfg.get("project", "runs/detect"),
        name=cfg.get("name", cfg["dataset_name"]),
        exist_ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO11 Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Check pipeline without training")
    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
