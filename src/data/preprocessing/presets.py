"""
Preprocessing Presets.

Provides pre-configured pipelines for common use cases.
"""

from typing import Tuple
from .pipeline import PreprocessingPipeline
from .core import resize_image, gaussian_blur, apply_clahe


def get_basic_pipeline(
    target_size: Tuple[int, int] = (640, 640)
) -> PreprocessingPipeline:
    """
    Basic preprocessing: chỉ resize, không có blur hay CLAHE.

    Use case: Baseline experiments, minimal preprocessing

    Args:
        target_size: Target image size (width, height)

    Returns:
        PreprocessingPipeline instance
    """
    return PreprocessingPipeline(
        [
            lambda img: resize_image(img, target_size),
        ]
    )


def get_v0_pipeline(target_size: Tuple[int, int] = (640, 640)) -> PreprocessingPipeline:
    """
    v0/v2 pipeline: Gaussian Blur + CLAHE 2.0.

    This is the default preprocessing used in:
    - reproduce_dataset_v0.py
    - reproduce_dataset_v2.py (if existed)

    Args:
        target_size: Target image size (width, height)

    Returns:
        PreprocessingPipeline instance
    """
    return PreprocessingPipeline(
        [
            lambda img: resize_image(img, target_size),
            lambda img: gaussian_blur(img, kernel_size=(5, 5)),
            lambda img: apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)),
        ]
    )


def get_v3_legacy_pipeline(
    target_size: Tuple[int, int] = (640, 640)
) -> PreprocessingPipeline:
    """
    v3 legacy pipeline: No Blur + CLAHE 4.0.

    Designed to match legacy dataset preprocessing:
    - Sharper images (no blur)
    - Higher contrast (CLAHE 4.0)

    Used in: reproduce_dataset_v3_legacy.py

    Args:
        target_size: Target image size (width, height)

    Returns:
        PreprocessingPipeline instance
    """
    return PreprocessingPipeline(
        [
            lambda img: resize_image(img, target_size),
            lambda img: apply_clahe(img, clip_limit=4.0, tile_grid_size=(8, 8)),
        ]
    )


def get_notebook_pipeline(
    target_size: Tuple[int, int] = (640, 640)
) -> PreprocessingPipeline:
    """
    Notebook preprocessing pipeline (from Yolo_Detection_XuongKhop_v0.ipynb).

    Identical to v0 pipeline:
    - Resize
    - Gaussian Blur (5x5)
    - CLAHE (clip_limit=2.0, tile_grid_size=(8,8))

    This preset is provided for clarity when reproducing notebook experiments.

    Args:
        target_size: Target image size (width, height)

    Returns:
        PreprocessingPipeline instance
    """
    # Notebook preprocessing is identical to v0
    return get_v0_pipeline(target_size)


def get_custom_pipeline(
    use_blur: bool = True,
    clahe_clip: float = 2.0,
    target_size: Tuple[int, int] = (640, 640),
) -> PreprocessingPipeline:
    """
    Custom pipeline with configurable parameters.

    Allows flexible combination of preprocessing steps.

    Args:
        use_blur: Whether to apply Gaussian blur
        clahe_clip: CLAHE clip limit (0 to disable CLAHE)
        target_size: Target image size (width, height)

    Returns:
        PreprocessingPipeline instance
    """
    steps = [lambda img: resize_image(img, target_size)]

    if use_blur:
        steps.append(lambda img: gaussian_blur(img, kernel_size=(5, 5)))

    if clahe_clip > 0:
        steps.append(
            lambda img: apply_clahe(img, clip_limit=clahe_clip, tile_grid_size=(8, 8))
        )

    return PreprocessingPipeline(steps)


# Preset aliases for convenience
BASIC = get_basic_pipeline
V0 = get_v0_pipeline
V3_LEGACY = get_v3_legacy_pipeline
NOTEBOOK = get_notebook_pipeline
CUSTOM = get_custom_pipeline
