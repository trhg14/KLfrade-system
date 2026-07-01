"""
Modular Preprocessing Package.

Provides flexible, composable preprocessing operations and pipelines.

Usage:
    # Use presets
    from src.data.preprocessing.presets import get_v0_pipeline, get_basic_pipeline

    # Basic preprocessing
    pipeline = get_basic_pipeline(target_size=(640, 640))
    processed_img, _ = pipeline(raw_img)

    # v0 preprocessing (Blur + CLAHE 2.0)
    pipeline = get_v0_pipeline()
    processed_img, _ = pipeline(raw_img)

    # Custom pipeline
    from src.data.preprocessing.pipeline import PreprocessingPipeline
    from src.data.preprocessing.core import resize_image, apply_clahe

    pipeline = PreprocessingPipeline([
        lambda img: resize_image(img, (512, 512)),
        lambda img: apply_clahe(img, clip_limit=3.0),
    ])
"""

from .pipeline import PreprocessingPipeline
from .presets import (
    get_basic_pipeline,
    get_v0_pipeline,
    get_v3_legacy_pipeline,
    get_notebook_pipeline,
    get_custom_pipeline,
    BASIC,
    V0,
    V3_LEGACY,
    NOTEBOOK,
    CUSTOM,
)

# Import core operations for convenience
from .core import (
    load_image,
    save_image,
    resize_image,
    gaussian_blur,
    apply_clahe,
)

__version__ = "1.0.0"

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    # Presets
    "get_basic_pipeline",
    "get_v0_pipeline",
    "get_v3_legacy_pipeline",
    "get_notebook_pipeline",
    "get_custom_pipeline",
    "BASIC",
    "V0",
    "V3_LEGACY",
    "NOTEBOOK",
    "CUSTOM",
    # Core operations (commonly used)
    "load_image",
    "save_image",
    "resize_image",
    "gaussian_blur",
    "apply_clahe",
]
