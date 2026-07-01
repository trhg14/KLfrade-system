"""
Preprocessing Pipeline Composer.

Provides a flexible way to chain preprocessing operations.
"""

import numpy as np
from typing import List, Callable, Optional, Tuple


class PreprocessingPipeline:
    """
    Composable preprocessing pipeline.

    Allows chaining multiple preprocessing steps together.

    Example:
        >>> from preprocessing.core import resize_image, gaussian_blur, apply_clahe
        >>>
        >>> pipeline = PreprocessingPipeline([
        ...     lambda img: resize_image(img, (640, 640)),
        ...     lambda img: gaussian_blur(img, kernel_size=(5, 5)),
        ...     lambda img: apply_clahe(img, clip_limit=2.0),
        ... ])
        >>>
        >>> processed_img = pipeline(raw_img)
    """

    def __init__(self, steps: List[Callable]):
        """
        Initialize pipeline with preprocessing steps.

        Args:
            steps: List of callable preprocessing functions
        """
        self.steps = steps

    def __call__(
        self, image: np.ndarray, labels: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Apply all preprocessing steps to image.

        Args:
            image: Input image
            labels: Optional YOLO format labels

        Returns:
            Tuple of (processed_image, processed_labels)
        """
        current_image = image
        current_labels = labels

        for step in self.steps:
            # Check if step returns tuple (image, labels) or just image
            result = (
                step(current_image, current_labels)
                if self._takes_labels(step)
                else step(current_image)
            )

            if isinstance(result, tuple):
                current_image, current_labels = result
            else:
                current_image = result

        return current_image, current_labels

    def _takes_labels(self, func: Callable) -> bool:
        """Check if function accepts labels parameter."""
        # Simple heuristic: check if function signature has 2 parameters
        import inspect

        sig = inspect.signature(func)
        return len(sig.parameters) >= 2

    def add_step(self, step: Callable) -> "PreprocessingPipeline":
        """
        Add a preprocessing step to the pipeline.

        Args:
            step: Callable preprocessing function

        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self

    def insert_step(self, index: int, step: Callable) -> "PreprocessingPipeline":
        """
        Insert a preprocessing step at specific position.

        Args:
            index: Position to insert
            step: Callable preprocessing function

        Returns:
            Self for chaining
        """
        self.steps.insert(index, step)
        return self

    def remove_step(self, index: int) -> "PreprocessingPipeline":
        """
        Remove a preprocessing step.

        Args:
            index: Index of step to remove

        Returns:
            Self for chaining
        """
        self.steps.pop(index)
        return self

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(steps={len(self.steps)})"
