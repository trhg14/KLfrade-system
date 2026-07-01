"""Minimal FastAPI wrapper around the legacy two-stage knee grading pipeline."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
from src.api.pipeline import KneePipeline
from src.api.utils import read_image_file, encode_image_base64
import uvicorn
import cv2

app = FastAPI(title="Knee KL Grading Pipeline API", description="Two-stage pipeline: Knee Detection -> KL Grading")
# A single in-process pipeline instance is shared across requests.
pipeline_instance = None

class KLGrade(BaseModel):
    grade_class: str
    grade_conf: float
    grade_id: int
    grade_bbox: List[int]

class KneePrediction(BaseModel):
    knee_bbox: List[int]
    knee_conf: float
    kl_grade: Optional[KLGrade]

class PipelineResponse(BaseModel):
    filename: str
    predictions: List[KneePrediction]
    image_base64: Optional[str] = None

def load_pipeline(knee_path: str, grade_path: str):
    global pipeline_instance
    pipeline_instance = KneePipeline(knee_path, grade_path)

@app.post("/predict_pipeline", response_model=PipelineResponse)
async def predict_pipeline(
    file: UploadFile = File(...),
    knee_conf: float = Form(0.25),
    grade_conf: float = Form(0.25),
    return_image: bool = Form(False)
):
    global pipeline_instance
    if pipeline_instance is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    contents = await file.read()
    image_rgb = read_image_file(contents, file.filename)
    
    # The API stays transport-focused; model-specific orchestration lives in
    # `src.api.pipeline` so this layer only handles I/O and response shaping.
    results = pipeline_instance.process(image_rgb, knee_conf=knee_conf, grade_conf=grade_conf)

    predictions = []
    for res in results:
        # Normalize the raw dict into explicit response models for validation/docs.
        grade_info = res.get("kl_grade")
        kl_obj = None
        if grade_info:
            kl_obj = KLGrade(
                grade_class=grade_info["grade_class"],
                grade_conf=grade_info["grade_conf"],
                grade_id=grade_info["grade_id"],
                grade_bbox=grade_info.get("grade_bbox", [])
            )
            
        predictions.append(KneePrediction(
            knee_bbox=res["knee_bbox"],
            knee_conf=res["knee_conf"],
            kl_grade=kl_obj
        ))
        
    encoded_image = None
    if return_image and len(predictions) > 0:
        # Visualization is optional because base64 payloads are much larger than the
        # plain JSON response and are not needed for all clients.
        vis_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        for p in predictions:
            # Draw bbox
            x1, y1, x2, y2 = p.knee_bbox
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            label = "Knee"
            if p.kl_grade:
                label += f" | {p.kl_grade.grade_class} ({p.kl_grade.grade_conf:.2f})"
            
            cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        encoded_image = encode_image_base64(vis_img)
        
    return PipelineResponse(
        filename=file.filename,
        predictions=predictions,
        image_base64=encoded_image
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_loaded": pipeline_instance is not None}

def start_api_server(host: str, port: int, knee_model: str, grade_model: str):
    load_pipeline(knee_model, grade_model)
    uvicorn.run(app, host=host, port=port)
