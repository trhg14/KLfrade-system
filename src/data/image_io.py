"""Image I/O with OpenCV when available, otherwise Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    Image = None  # type: ignore
    _HAS_PIL = False


def read_image(path: Path) -> Optional[np.ndarray]:
    path = Path(path)
    if _HAS_CV2:
        img = cv2.imread(str(path))
        return img
    if _HAS_PIL:
        return np.array(Image.open(path).convert("RGB"))
    raise ImportError("Install opencv-python or Pillow to load images")


def write_image(path: Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_CV2:
        cv2.imwrite(str(path), image)
        return
    if _HAS_PIL:
        Image.fromarray(image.astype(np.uint8)).save(path)
        return
    raise ImportError("Install opencv-python or Pillow to save images")


def get_image_size(path: Path) -> Tuple[int, int]:
    """Return (width, height)."""
    if _HAS_CV2:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        h, w = img.shape[:2]
        return w, h
    if _HAS_PIL:
        with Image.open(path) as im:
            return im.size
    raise ImportError("Install opencv-python or Pillow to read image size")
