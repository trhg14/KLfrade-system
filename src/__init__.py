"""
KLGrade Source Module

Main package for KLGrade object detection.
"""

from . import config

__all__ = ["config", "datasets", "utils"]


def __getattr__(name: str):
    if name == "datasets":
        from . import datasets as _datasets

        return _datasets
    if name == "utils":
        from . import utils as _utils

        return _utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
