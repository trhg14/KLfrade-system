"""Lightweight two-stage API pipeline: detect knee first, grade each crop second."""
import numpy as np
import cv2
from src.api.inference import YOLOModel

class KneePipeline:
    def __init__(self, knee_model_path: str, grade_model_path: str):
        print(f"🔄 Initializing Knee Pipeline...")
        print(f"   - Knee Model: {knee_model_path}")
        print(f"   - Grade Model: {grade_model_path}")
        
        # Keep the detector and grader separate so each model can evolve independently.
        self.knee_model = YOLOModel(knee_model_path)
        self.grade_model = YOLOModel(grade_model_path)
        print("✅ Pipeline loaded successfully.")

    def process(self, image_rgb: np.ndarray, knee_conf: float = 0.25, grade_conf: float = 0.25):
        """
        Full pipeline: Detect Knee -> Crop -> Grade KL.
        """
        # 1. Detect Knees
        knee_results = self.knee_model.predict(image_rgb, conf=knee_conf)
        
        if not knee_results:
            return []
            
        final_predictions = []
        knee_result = knee_results[0]
        
        # Each detection becomes an independent crop for the grading model.
        for i, box in enumerate(knee_result.boxes):
            # Extract box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            knee_conf_score = float(box.conf[0].item())
            
            # Crop logic (ensure bounds)
            h, w, _ = image_rgb.shape
            
            # Optional: Add padding? For now strict crop as usually preferred for classification if trained on crops
            # Safety check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_img = image_rgb[y1:y2, x1:x2]
            
            # The second-stage model runs on the cropped knee only, not on the full
            # X-ray, to stay close to its training distribution.
            grade_results = self.grade_model.predict(crop_img, conf=grade_conf)
            
            grade_pred = None
            if grade_results:
                g_res = grade_results[0]
                if len(g_res.boxes) > 0:
                    # Reduce possibly multiple outputs on the crop to the single
                    # highest-confidence grading hypothesis.
                    best_box = max(g_res.boxes, key=lambda b: b.conf[0].item())
                    
                    cls_id = int(best_box.cls[0].item())
                    conf = float(best_box.conf[0].item())
                    
                    # Extract grade box (relative to crop)
                    gx1, gy1, gx2, gy2 = best_box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Convert to global coordinates
                    global_gx1 = gx1 + x1
                    global_gy1 = gy1 + y1
                    global_gx2 = gx2 + x1
                    global_gy2 = gy2 + y1
                    
                    class_name = self.grade_model.class_mapping.get(cls_id, str(cls_id))
                    
                    grade_pred = {
                        "grade_class": class_name,
                        "grade_conf": conf,
                        "grade_id": cls_id,
                        "grade_bbox": [int(global_gx1), int(global_gy1), int(global_gx2), int(global_gy2)]
                    }
                else:
                    # No grade detected? Maybe 'Healthy' or low conf?
                    grade_pred = {"grade_class": "Unknown", "grade_conf": 0.0, "grade_id": -1, "grade_bbox": []}
            
            # Keep the response as plain JSON-friendly objects for the API layer.
            final_predictions.append({
                "knee_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "knee_conf": knee_conf_score,
                "kl_grade": grade_pred
            })
            
        return final_predictions
