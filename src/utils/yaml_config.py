"""
YAML Configuration Utilities

Shared utilities for creating and managing YOLO YAML config files.
"""

import yaml
from pathlib import Path
from typing import Union, List, Dict, Optional


def create_yolo_config(
    dataset_dir: Union[str, Path],
    nc: int,
    names: Union[List[str], Dict[int, str]],
    output_path: Union[str, Path],
    train_split: str = "train.txt",
    val_split: str = "val.txt",
    test_split: Optional[str] = "test.txt",
) -> Path:
    """
    Create a YOLO dataset configuration YAML file.

    Args:
        dataset_dir: Directory containing the dataset
        nc: Number of classes
        names: Class names (list or dict)
        output_path: Path to save the config YAML
        train_split: Name of train split file (default: train.txt)
        val_split: Name of val split file (default: val.txt)
        test_split: Name of test split file (default: test.txt), None to omit

    Returns:
        Path: Absolute path to created config file
    """
    dataset_dir = Path(dataset_dir).absolute()
    output_path = Path(output_path)

    # Convert names to list if dict
    if isinstance(names, dict):
        names_list = list(names.values())
    else:
        names_list = names

    # Validate
    if len(names_list) != nc:
        raise ValueError(f"Number of names ({len(names_list)}) doesn't match nc ({nc})")

    # Check for split files
    train_txt = dataset_dir / train_split
    val_txt = dataset_dir / val_split

    if not val_txt.exists():
        print(f"⚠️  Warning: val.txt not found at {val_txt}")

    # Build config
    config = {
        "path": str(
            dataset_dir.parent.parent.parent
        ),  # Root path (often overridden by absolute paths below)
        "train": (
            str(train_txt) if train_txt.exists() else str(val_txt)
        ),  # Fallback to val if train missing
        "val": str(val_txt),
        "nc": nc,
        "names": names_list,
    }

    # Add test split if provided
    if test_split:
        test_txt = dataset_dir / test_split
        if test_txt.exists():
            config["test"] = str(test_txt)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created YAML config: {output_path}")
    return output_path.absolute()


def load_yolo_config(config_path: Union[str, Path]) -> Dict:
    """
    Load and parse a YOLO config YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Parsed configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def update_yolo_config(
    config_path: Union[str, Path],
    updates: Dict,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Update an existing YOLO config file.

    Args:
        config_path: Path to existing config
        updates: Dictionary of updates to apply
        output_path: Path to save updated config (default: overwrite original)

    Returns:
        Path: Path to updated config
    """
    config = load_yolo_config(config_path)
    config.update(updates)

    output_path = Path(output_path) if output_path else Path(config_path)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path.absolute()


def validate_yolo_config(config_path: Union[str, Path]) -> bool:
    """
    Validate a YOLO config file.

    Args:
        config_path: Path to config file

    Returns:
        bool: True if valid, raises exception otherwise
    """
    config = load_yolo_config(config_path)

    # Check required fields
    required = ["nc", "names"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    # Check names count matches nc
    if len(config["names"]) != config["nc"]:
        raise ValueError(
            f"Number of names ({len(config['names'])}) doesn't match nc ({config['nc']})"
        )

    # Check split files exist (if paths are absolute)
    for split in ["train", "val", "test"]:
        if split in config:
            split_path = Path(config[split])
            if split_path.is_absolute() and not split_path.exists():
                print(f"⚠️  Warning: {split} split not found: {split_path}")

    return True
