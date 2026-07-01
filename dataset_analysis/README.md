# Dataset Analysis Tools

**Location**: `tools/dataset_analysis/`  
**Purpose**: Comprehensive dataset validation, analysis, and visualization

---

## 📁 Structure

```
tools/dataset_analysis/
├── __init__.py
├── validate.py              # Dataset validation
├── analyze.py               # Comprehensive analysis  
├── compare.py               # Dataset comparison
├── class_split_report.py    # Class distribution reports
├── check_augment.py         # Aug

mentation validation
├── resize_images.py         # Image resizing utility
└── visualize/               # Visualization subpackage
    ├── __init__.py
    ├── augmentations.py     # Augmentation viz
    ├── samples.py           # Sample viz
    └── class_samples.py     # Class-wise samples
```

---

## 🔧 Main Tools

### `validate.py` - Dataset Validation
Validates YOLO format datasets for correctness.

```bash
python tools/dataset_analysis/validate.py \
    --data_dir datasets/processed/knee_cropped \
    --split train
```

### `analyze.py` - Comprehensive Analysis
Full dataset analysis with statistics and visualizations.

```bash
python tools/dataset_analysis/analyze.py \
    --dataset_path datasets/processed/knee_cropped \
    --output_dir analysis/knee_cropped
```

### `compare.py` - Dataset Comparison
Compare multiple dataset variants.

```bash
python tools/dataset_analysis/compare.py \
    --datasets datasets/processed/knee_cropped datasets/balanced/knee_cropped \
    --output analysis/comparison
```

### `class_split_report.py` - Class Distribution
Generate detailed class distribution reports.

```bash
python tools/dataset_analysis/class_split_report.py \
    --images_dir datasets/dataset_v0/images \
    --labels_dir datasets/dataset_v0/labels \
    --output analysis/class_distribution
```

---

## 🎨 Visualization Tools

### Augmentation Visualization
```bash
python tools/dataset_analysis/visualize/augmentations.py \
    --dataset_path datasets/processed/knee_cropped \
    --num_samples 10
```

### Sample Visualization
```bash
python tools/dataset_analysis/visualize/samples.py \
    --dataset_path datasets/processed/knee_cropped \
    --num_samples 20
```

### Class-wise Samples
```bash
python tools/dataset_analysis/visualize/class_samples.py \
    --dataset_path datasets/processed/knee_cropped \
    --samples_per_class 5
```

---

## 📋 Usage

These are **general-purpose dataset analysis tools** that work with YOLO-format datasets.

### Used by YOLO Data Pipeline

These tools are called by YOLO dataset preparation pipeline scripts:

- `scripts/pipelines/run_full_pipeline.sh` - **YOLO data pipeline** (uses `analyze.py`)
- `scripts/pipelines/step_2_generate_labels.sh` - **YOLO label generation** (uses `class_split_report.py`)
- `scripts/analyzes/run_comprehensive_analysis_all.sh` - **Dataset analysis** (uses `analyze.py`)

> **Note**: These tools validate and analyze YOLO-format datasets. They are NOT model-specific and can be used for any YOLO dataset (YOLO detection, KIOCMIL CADA input data, etc.).

---

## 🔗 Related

### Shared Utilities
These tools use **shared visualization utilities** for consistent plotting:
- `src/utils/dataset_viz.py` - Dataset viz functions (draw boxes, class distributions, grids)
- `src/utils/visualization.py` - Model evaluation plots (CM, ROC, PR curves)

### Other Components
- **Model viz**: `src/visualization/` (GradCAM, predictions)
- **Analysis scripts**: `scripts/analyzes/` (batch analysis workflows)
- **Evaluation**: `scripts/evaluation/` (model evaluation workflows)

**Architecture**: Specific tools → Use shared utilities → Avoid code duplication

---

**Last Updated**: 2026-01-21  
**Status**: Refactored & Organized
