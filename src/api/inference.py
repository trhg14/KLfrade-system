import torch
from ultralytics import YOLO
from src.config import CLASSES_8_CLASS


class YOLOModel:
    def __init__(self, model_path: str):
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.names = self._normalize_names(self.model.names)
        self.num_classes = len(self.names)
        print(f"Model loaded. Detected {self.num_classes} classes.")

        # Determine class mapping from config
        self.class_mapping = self._get_class_mapping()

    @staticmethod
    def _normalize_names(names):
        """Ultralytics may return list or dict depending on export/runtime."""
        if isinstance(names, dict):
            return names
        return {idx: name for idx, name in enumerate(names)}

    def _get_class_mapping(self):
        """Match number of classes to config definitions."""
        if self.num_classes == 8:
            print("Using 8-class mapping (CLASSES_8_CLASS)")
            return CLASSES_8_CLASS
        else:
            print(
                f"Warning: No matching config for {self.num_classes} classes. Using model internal names."
            )
            return self.names

    def predict(self, image, conf=0.25, **kwargs):
        """Run inference on an image."""
        results = self.model.predict(image, conf=conf, verbose=False, **kwargs)
        return results