from __future__ import annotations

import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.utils import encode_image_base64, read_image_file
from src.api.yolo_detection_service import YOLODetectionService


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Top-left x coordinate")
    y1: int = Field(..., description="Top-left y coordinate")
    x2: int = Field(..., description="Bottom-right x coordinate")
    y2: int = Field(..., description="Bottom-right y coordinate")


class DetectionPrediction(BaseModel):
    class_id: int = Field(..., description="Predicted class id")
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Detected bounding box")


class DetectionResponse(BaseModel):
    status: str = Field(default="success")
    filename: str
    image_width: int
    image_height: int
    num_detections: int
    predictions: List[DetectionPrediction]
    processing_time_ms: float


class VisualDetectionResponse(DetectionResponse):
    annotated_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    task: str


class ModelInfoResponse(BaseModel):
    model_path: str
    device: str
    task: str
    num_classes: int
    class_names: List[str]
    class_mapping: dict[int, str]


app = FastAPI(
    title="KLGrade Knee YOLO API",
    description=(
        "Detection-only API for the trained knee YOLO model stored under "
        "`knee_yolo11n/.../weights/best.pt`."
    ),
    version="1.0.0",
)

detector_service: Optional[YOLODetectionService] = None


def load_detector(model_path: str) -> None:
    global detector_service
    detector_service = YOLODetectionService(model_path)


def _require_service() -> YOLODetectionService:
    if detector_service is None:
        raise HTTPException(status_code=500, detail="Detector model not initialized")
    return detector_service


def _build_response_payload(filename: str, result: dict, processing_time_ms: float) -> dict:
    predictions = [
        DetectionPrediction(
            class_id=item["class_id"],
            class_name=item["class_name"],
            confidence=item["confidence"],
            bbox=BoundingBox(**item["bbox"]),
        )
        for item in result["predictions"]
    ]
    return {
        "status": "success",
        "filename": filename,
        "image_width": result["image_width"],
        "image_height": result["image_height"],
        "num_detections": len(predictions),
        "predictions": predictions,
        "processing_time_ms": processing_time_ms,
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=detector_service is not None,
        task="knee_detection",
    )


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info():
    service = _require_service()
    return ModelInfoResponse(**service.get_model_info())


@app.post("/predict", response_model=DetectionResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25, ge=0.0, le=1.0),
    iou: float = Form(0.7, ge=0.0, le=1.0),
    max_det: int = Form(10, ge=1, le=1000),
):
    service = _require_service()
    start_time = time.time()

    contents = await file.read()
    image_rgb = read_image_file(contents, file.filename)
    result = service.predict(image_rgb, conf=conf, iou=iou, max_det=max_det)

    processing_time_ms = (time.time() - start_time) * 1000.0
    payload = _build_response_payload(file.filename, result, processing_time_ms)
    return DetectionResponse(**payload)


@app.post("/predict_visual", response_model=VisualDetectionResponse)
async def predict_visual(
    file: UploadFile = File(...),
    conf: float = Form(0.25, ge=0.0, le=1.0),
    iou: float = Form(0.7, ge=0.0, le=1.0),
    max_det: int = Form(10, ge=1, le=1000),
):
    service = _require_service()
    start_time = time.time()

    contents = await file.read()
    image_rgb = read_image_file(contents, file.filename)
    result = service.predict(image_rgb, conf=conf, iou=iou, max_det=max_det)

    processing_time_ms = (time.time() - start_time) * 1000.0
    payload = _build_response_payload(file.filename, result, processing_time_ms)
    payload["annotated_image_base64"] = (
        encode_image_base64(result["annotated_image_bgr"])
        if result["annotated_image_bgr"] is not None
        else None
    )
    return VisualDetectionResponse(**payload)


def start_detection_api_server(host: str, port: int, model_path: str) -> None:
    load_detector(model_path)
    uvicorn.run(app, host=host, port=port)
