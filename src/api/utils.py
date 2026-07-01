import io
import cv2
import numpy as np
import base64

try:
    import pydicom
except ImportError:
    pydicom = None

from fastapi import HTTPException

def read_image_file(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Read image bytes (PNG/JPG/DICOM) into numpy array (RGB).
    """
    is_dicom = filename.lower().endswith('.dcm')
    
    if is_dicom:
        if pydicom is None:
            raise HTTPException(status_code=500, detail="pydicom not installed")
        
        try:
            with io.BytesIO(file_bytes) as bio:
                ds = pydicom.dcmread(bio)
                pixel_array = ds.pixel_array
                
                # Normalize
                if pixel_array.max() > 0:
                    pixel_array = (pixel_array / pixel_array.max()) * 255.0
                pixel_array = pixel_array.astype(np.uint8)
                
                if len(pixel_array.shape) == 2:
                    image_rgb = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
                else:
                    image_rgb = pixel_array
                return image_rgb
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid DICOM file: {str(e)}")
            
    else:
        # Standard image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def encode_image_base64(image_bgr: np.ndarray) -> str:
    """Encode BGR numpy image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')
