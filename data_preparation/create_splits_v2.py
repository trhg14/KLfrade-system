import os
import random
from pathlib import Path


def create_splits(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    img_dir = Path(dataset_dir) / "images"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = [
        f.name for f in img_dir.iterdir() if f.suffix in {".jpg", ".png", ".jpeg"}
    ]
    random.shuffle(images)

    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    print(f"Total: {n}")
    print(f"Train: {len(train_imgs)}")
    print(f"Val: {len(val_imgs)}")
    print(f"Test: {len(test_imgs)}")

    def write_split(split_name, img_names):
        with open(output_dir / f"{split_name}.txt", "w") as f:
            for name in img_names:
                # Write path relative to project root (assuming dataset_dir is relative to root)
                # Or just dataset relative path
                path = Path(dataset_dir) / "images" / name
                f.write(f"{path}\n")

    write_split("train", train_imgs)
    write_split("val", val_imgs)
    write_split("test", test_imgs)
    print(f"Splits saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    create_splits(dataset_dir=args.dataset_dir, output_dir=args.output_dir)
