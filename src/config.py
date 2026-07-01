# ==============================================================================
# CLASS DEFINITIONS FOR ALL DATASET VARIANTS
# ==============================================================================

# Dataset 1: 5-class (KL0-4) - Traditional KL grading
# Used for: processed/knee/
CLASSES = {
    0: "KL0",
    1: "KL1",
    2: "KL2",
    3: "KL3",
    4: "KL4",
}

# Dataset 2: 10-class (KL0-a → KL4-b) - Detailed structure classification
# Used for: processed/knee_10_class/
# Extended mapping for tag-style labels (e.g., "3a"/"3b") generated in label_new.
# The base id is kept for compatibility while suffixes describe sub-structures (joint space vs osteophyte).
CLASSES_10_CLASS = {
    0: "KL0-a",  # Osteophyte (gai xương)
    1: "KL0-b",  # Joint space (khe khớp)
    2: "KL1-a",
    3: "KL1-b",
    4: "KL2-a",
    5: "KL2-b",
    6: "KL3-a",
    7: "KL3-b",
    8: "KL4-a",
    9: "KL4-b",
}

# Dataset 3: 4-class (KL1-4, no KL0) - Focus on pathological cases
# Used for: processed/knee_4_class/
# Remapped from 5-class by removing KL0 and shifting down
CLASSES_4_CLASS = {
    0: "KL1",  # Original class 1
    1: "KL2",  # Original class 2
    2: "KL3",  # Original class 3
    3: "KL4",  # Original class 4
}

# Dataset 4: 8-class (KL1-a → KL4-b, no KL0) - Detailed + pathological focus
# Used for: processed/knee_8_class/
# Remapped from 10-class by removing KL0-a, KL0-b and shifting down
CLASSES_8_CLASS = {
    0: "KL1-a",  # Original class 2
    1: "KL1-b",  # Original class 3
    2: "KL2-a",  # Original class 4
    3: "KL2-b",  # Original class 5
    4: "KL3-a",  # Original class 6
    5: "KL3-b",  # Original class 7
    6: "KL4-a",  # Original class 8
    7: "KL4-b",  # Original class 9
}


import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "analysis"  # Centralized analysis output

# Image size for resizing all input images (height, width)
IMG_SIZE = 640  # Default input size for classification and transforms

# Detection utility
NUM_CLASSES = len(CLASSES) + 1  # +1 for background

# Training hyperparameters
BATCH_SIZE = 1
EPOCHS = 30

# Detection Classes (Lesions)
# Updated based on tools/check_dataset/class_split_report.py
# Even classes (0, 2, 4, 6, 8) -> Suffix 'a' -> Osteophytes
# Odd classes (1, 3, 5, 7, 9) -> Suffix 'b' -> Joint Space Narrowing
OST_CLASSES = {0, 2, 4, 6, 8}
JS_CLASSES = {1, 3, 5, 7, 9}

# Knee detection class (in `dataset/dataset_v0/labels-knee/`)
KNEE_CLASS_ID = 0
