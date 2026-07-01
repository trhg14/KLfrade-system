# ==============================================================================
# CLASS DEFINITIONS
# ==============================================================================

# 8-class (KL1-a → KL4-b, no KL0) - Detailed + pathological focus
# Even suffix 'a' -> Osteophytes (gai xương)
# Odd suffix 'b'  -> Joint Space Narrowing (khe khớp)
CLASSES_8_CLASS = {
    0: "KL1-a",
    1: "KL1-b",
    2: "KL2-a",
    3: "KL2-b",
    4: "KL3-a",
    5: "KL3-b",
    6: "KL4-a",
    7: "KL4-b",
}


import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "analysis"  # Centralized analysis output

# Image size for resizing all input images (height, width)
IMG_SIZE = 640  # Default input size for classification and transforms

# Detection utility
NUM_CLASSES = len(CLASSES_8_CLASS) + 1  # +1 for background

# Training hyperparameters
BATCH_SIZE = 1
EPOCHS = 30

# Detection Classes (Lesions)
# Even classes (0, 2, 4, 6) -> Suffix 'a' -> Osteophytes
# Odd classes (1, 3, 5, 7)  -> Suffix 'b' -> Joint Space Narrowing
OST_CLASSES = {0, 2, 4, 6}
JS_CLASSES = {1, 3, 5, 7}

# Knee detection class (Stage 1 - YOLO11n)
KNEE_CLASS_ID = 0