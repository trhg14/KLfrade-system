import sys
import argparse
from pathlib import Path

project_root = (
    Path(__file__).resolve().parents[2]
)  
sys.path.append(str(project_root))

from src.data.balancing import balance_dataset, get_dataset_stats


def main():
    parser = argparse.ArgumentParser(
        description="Balance dataset by oversampling minority classes"
    )

    # Required arguments
    parser.add_argument(
        "--input-images", type=str, required=True, help="Input image directory"
    )
    parser.add_argument(
        "--input-labels", type=str, required=True, help="Input label directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for balanced dataset",
    )

    parser.add_argument(
        "--num-classes", type=int, default=5, help="Number of classes (default: 5)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="flip",
        choices=["flip", "none"],
        help="Augmentation strategy (default: flip)",
    )
    parser.add_argument(
        "--target-strategy",
        type=str,
        default="max",
        choices=["max", "median", "mean", "custom"],
        help="Target calculation strategy (default: max)",
    )
    parser.add_argument(
        "--custom-target",
        type=int,
        default=None,
        help="Custom target count (required if target-strategy=custom)",
    )
    parser.add_argument(
        "--aux-labels",
        type=str,
        nargs="*",
        default=None,
        help="Auxiliary label directories to sync (e.g., knee boxes)",
    )
    parser.add_argument(
        "--skip-stats", action="store_true", help="Skip initial dataset statistics"
    )

    args = parser.parse_args()

    input_images = Path(args.input_images)
    input_labels = Path(args.input_labels)
    output_dir = Path(args.output_dir)

    if not input_images.exists():
        print(f"❌ Error: Image directory not found: {input_images}")
        sys.exit(1)

    if not input_labels.exists():
        print(f"❌ Error: Label directory not found: {input_labels}")
        sys.exit(1)

    output_images = output_dir / "images"
    output_labels = output_dir / "labels"

    aux_label_dirs = None
    output_aux_label_dirs = None

    if args.aux_labels:
        aux_label_dirs = []
        output_aux_label_dirs = []

        for aux_dir_str in args.aux_labels:
            aux_dir = Path(aux_dir_str)
            if not aux_dir.exists():
                print(f"⚠️  Warning: Auxiliary label directory not found: {aux_dir}")
                continue

            aux_label_dirs.append(aux_dir)

            aux_out_name = aux_dir.name
            aux_out_path = output_dir / aux_out_name

            if aux_out_path == output_labels:
                new_name = f"{aux_out_name}_aux"
                print(
                    f"⚠️  Warning: Aux dir '{aux_out_name}' collides with primary output. Renaming to '{new_name}'"
                )
                aux_out_path = output_dir / new_name

            output_aux_label_dirs.append(aux_out_path)

    if not args.skip_stats:
        print("\n" + "=" * 60)
        print("ORIGINAL DATASET STATISTICS")
        print("=" * 60)
        get_dataset_stats(input_images, input_labels, args.num_classes)

    stats = balance_dataset(
        image_dir=input_images,
        label_dir=input_labels,
        output_img_dir=output_images,
        output_label_dir=output_labels,
        strategy=args.strategy,
        target_strategy=args.target_strategy,
        custom_target=args.custom_target,
        num_classes=args.num_classes,
        aux_label_dirs=aux_label_dirs,
        output_aux_label_dirs=output_aux_label_dirs,
        save_report=True,
    )

    print("\n" + "=" * 60)
    print("BALANCED DATASET STATISTICS")
    print("=" * 60)
    get_dataset_stats(output_images, output_labels, args.num_classes)

    print(f"\n✅ Balanced dataset saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Preprocess balanced dataset")
    print(f"  2. Create train/val/test splits")
    print(f"  3. Train model on balanced data")


if __name__ == "__main__":
    main()
