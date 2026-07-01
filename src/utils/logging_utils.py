"""
Logging utilities for training scripts.

Provides functions for managing log directories and saving training configurations.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def get_next_log_dir(base_dir: str = "log", module_name: str = None) -> Path:
    """
    Get next available log directory with run number.
    Creates log/run_001, log/run_002 (no module name)
    or log/run_kiocmil_001, log/run_kiocmil_002 (with module name)

    Args:
        base_dir: Base directory for logs (default: 'log')
        module_name: Name of module/script (e.g., 'kiocmil', 'detr')

    Returns:
        Path to next available log directory

    Example:
        >>> log_dir = get_next_log_dir("log", "kiocmil")
        >>> print(log_dir)  # log/run_kiocmil_001
    """
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    # Determine prefix based on module name
    if module_name:
        prefix = f"run_{module_name}_"
    else:
        prefix = "run_"

    # Find existing run directories with this prefix
    existing_runs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            try:
                # Extract number from end of directory name
                parts = item.name.split("_")
                run_num = int(parts[-1])
                existing_runs.append(run_num)
            except (ValueError, IndexError):
                continue

    # Get next run number
    next_run = 1 if not existing_runs else max(existing_runs) + 1

    # Create new log directory
    log_dir = base_path / f"{prefix}{next_run:03d}"
    log_dir.mkdir(exist_ok=True)

    return log_dir


def save_training_config(
    log_dir: Path, args: Any, additional_info: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save training configuration to log directory.

    Args:
        log_dir: Directory to save config to
        args: Training arguments (from argparse)
        additional_info: Additional information to save (optional)

    Returns:
        Path to saved config file

    Example:
        >>> log_dir = get_next_log_dir()
        >>> config_path = save_training_config(log_dir, args, {"device": "cuda:0"})
    """
    config_file = log_dir / "config.json"

    # Build config dictionary
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }

    # Add additional info if provided
    if additional_info:
        config.update(additional_info)

    # Save to JSON
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config_file


def setup_training_logging(
    base_dir: str = "log",
    module_name: str = None,
    args: Any = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Setup logging for a training run.
    Creates versioned log directory and saves configuration.

    Args:
        base_dir: Base directory for logs (default: 'log')
        module_name: Name of module/script (e.g., 'kiocmil', 'detr')
        args: Training arguments to save (optional)
        additional_info: Additional information to save (optional)

    Returns:
        Path to created log directory

    Example:
        >>> log_dir = setup_training_logging("log", "kiocmil", args, {"model": "resnet18"})
        >>> print(f"Logging to: {log_dir}")  # log/run_kiocmil_001
    """
    # Create versioned log directory
    log_dir = get_next_log_dir(base_dir, module_name)

    # Save configuration if args provided
    if args is not None:
        save_training_config(log_dir, args, additional_info)

    return log_dir
