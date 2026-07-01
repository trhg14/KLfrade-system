from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.api.inference import YOLOModel


class YOLODetectionService:
    """Small service wrapper for YOLO detection inference."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path).resolve()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLOModel(str(self.model_path))

    def get_model_info(self) -> dict[str, Any]:
        class_names = [
            self.model.class_mapping[class_id]
            for class_id in sorted(self.model.class_mapping.keys())
        ]
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "num_classes": self.model.num_classes,
            "class_names": class_names,
            "class_mapping": self.model.class_mapping,
            "task": "detect",
        }

    def predict(
        self,
        image_rgb: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 10,
    ) -> dict[str, Any]:
        results = self.model.predict(
            image_rgb,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self.device,
        )

        if not results:
            return {
                "image_width": int(image_rgb.shape[1]),
                "image_height": int(image_rgb.shape[0]),
                "predictions": [],
                "annotated_image_bgr": None,
            }

        result = results[0]
        predictions = []

        for box in result.boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].cpu().tolist()]
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = self.model.class_mapping.get(class_id, str(class_id))

            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                }
            )

        predictions.sort(
            key=lambda item: (
                item["bbox"]["x1"],
                item["bbox"]["y1"],
                -item["confidence"],
            )
        )

        return {
            "image_width": int(image_rgb.shape[1]),
            "image_height": int(image_rgb.shape[0]),
            "predictions": predictions,
            "annotated_image_bgr": result.plot(),
        }
