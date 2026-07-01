# Bản Đồ Project KLGrade


## 1. Mục tiêu dự án


Hệ thống hiện tại bao gồm 3 nhóm chức năng chính:

1. Chuẩn bị dữ liệu từ ảnh gốc, sinh nhãn, cân bằng dữ liệu và tạo split.
2. Huấn luyện mô hình YOLO để phát hiện đầu gối hoặc tổn thương.
3. Đóng gói mô hình thành API nhẹ để suy luận trên ảnh đầu vào.

**Ghi chú:** Các nhánh mô hình nâng cao (KIOCMIL CADA, DETR, CDT-CAD) được tham chiếu trong tài liệu nhưng không có trong checkout này. Repo hiện tại tập trung vào pipeline dữ liệu + YOLO training + API nhẹ.

## 2. Luồng xử lý tổng thể

Luồng dữ liệu và mô hình trong repo đi theo hướng sau:

1. Lấy dữ liệu gốc từ file nén `Data_Images_E-...zip`.
2. Import vào `datasets/dataset_v0`.
3. Sinh `labels-knee` nếu chỉ có nhãn tổn thương/KL.
4. Crop vùng đầu gối thành `datasets/dataset_knees_cropped`.
5. Tạo các biến thể nhãn `5/4/8/10 class`.
6. Cân bằng dữ liệu bằng oversampling và flip.
7. Tạo các biến thể preprocessing trong `datasets/processed_balanced`.
8. Tạo `train/val/test split`.
9. Huấn luyện mô hình.
10. Đánh giá, xuất báo cáo và phục vụ API.

## 3. Cấu trúc root của repo

| Thư mục / File | Vai trò | Ghi chú |
| --- | --- | --- |
| `src/` | Thư viện mã nguồn chính | Chứa logic model, dataset, training, API, utils. |
| `scripts/` | Entry point và workflow | Chứa script chạy trực tiếp, pipeline `.ps1/.sh`, deploy, evaluate, test nhanh. |
| `tools/` | Công cụ phân tích dataset | Validate, analyze, visualize dữ liệu. |
| `configs/` | Cấu hình huấn luyện | Chủ yếu là YAML cho YOLO và các biến thể enhanced. |
| `docs/` | Tài liệu dự án | Hướng dẫn, tham chiếu kỹ thuật, kết quả thực nghiệm. |
| `datasets/` | Dữ liệu làm việc chính | Raw, cropped, balanced, processed_balanced và split. |
| `processed/` | Thư mục tương thích cũ | Hiện gần như để trống, không còn là trung tâm dữ liệu. |
| `splits/` | Split cấp root | Hiện chỉ còn một phần nhỏ; split đầy đủ nằm trong `datasets/splits/`. |
| `runs/` | Kết quả huấn luyện | Checkpoint, metric, ảnh, kết quả thực nghiệm. |
| `log/`, `logs/` | Log huấn luyện và API | `log/` nghiêng về train, `logs/` nghiêng về API. |
| `wandb/` | Artifact từ Weights & Biases | Không phải source code. |
| `weight/`, `*.pt` | Weight dùng chung | Backbone và checkpoint dùng lại trong nhiều luồng. |
| `kiocmil_cada_v1/` | Snapshot/phân nhánh cũ | Một mini-project riêng để tái hiện phiên bản CADA trước đó. |
| `analysis_2/`, `test_model/`, `knee_yolo11n/`, `lesion_8class_balanced/` | Artifact / thử nghiệm | Chủ yếu là output mô hình, notebook, file sinh ra. |
| `README.md`, `QUICKSTART.md`, `README_SETUP.md` | Tài liệu đầu vào | Dễ đọc để nắm tổng quan, nhưng có chỗ đã cũ. |
| `setup_env.*`, `start_api.*`, `start_knee_yolo_api.bat` | Script khởi động nhanh | Hỗ trợ setup và chạy nhanh API. |
| `utils.py` | Helper nhỏ ở root | Chức năng trùng ý tưởng với `src/utils/bbox.py`. |

## 4. Phần nào là “source of truth”

Nếu cần sửa logic thật của hệ thống, ưu tiên theo thứ tự:

1. `src/`
2. `scripts/`
3. `tools/`
4. `configs/`

Các thư mục không nên sửa nếu không có chủ đích rõ ràng:

- `runs/`, `log/`, `logs/`, `wandb/`
- `analysis_2/`, `knee_yolo11n/`, `test_model/`
- `*.zip`, `*.pt`
- phần lớn nội dung trong `kiocmil_cada_v1/` nếu bạn không chủ động làm việc với snapshot cũ

## 5. Bản đồ mã nguồn chi tiết

### 5.1 `src/` - thư viện chính

Hiện tại có khoảng `85` file Python.

#### `src/__init__.py`, `src/config.py`, file root trong `src/`

- `src/__init__.py`
  Vai trò: package entry, lazy-load `datasets` và `utils`.
- `src/config.py`
  Vai trò: ánh xạ class cho `5/10/4/8 class`, khai báo `OST_CLASSES`, `JS_CLASSES`, `KNEE_CLASS_ID`, `PROJECT_ROOT` và vài tham số mặc định.
- `src/Object-Context.txt`
  Vai trò: tài liệu/tham chiếu nội bộ, không phải code thực thi.

#### `src/api/` - API và wrapper suy luận

- `src/api/inference.py`
  Wrapper `YOLOModel`; load model Ultralytics và tự động chọn class mapping theo số lớp.
- `src/api/pipeline.py`
  Pipeline hai bước `detect knee -> crop -> grade`; là nhánh suy luận YOLO gọn.
- `src/api/server.py`
  FastAPI tối giản cho pipeline hai giai đoạn với endpoint chính `/predict_pipeline`.
- `src/api/utils.py`
  Đọc ảnh upload (`png/jpg/dicom`) và encode base64 để trả ảnh annotate.
- `src/api/response_schemas.py`
  Khai báo schema Pydantic cho API KIOCMIL: bbox, lesion summary, health, error, response.
- `src/api/logger_config.py`
  Thiết lập logger cho API và các helper log request/response/model load.
- `src/api/gradcam.py`
  Tạo heatmap, overlay, annotation cho nhánh API KIOCMIL.
- `src/api/kiocmil_inference.py`
  Wrapper suy luận đầy đủ cho KIOCMIL CADA; load `kiocmil model + knee detector + lesion detector`, cắt knee, cắt lesion, dựng batch, chạy model cuối.
- `src/api/yolo_detection_service.py`
  Service wrapper gọn cho detection-only API.
- `src/api/yolo_detection_server.py`
  FastAPI riêng cho YOLO detection.

Ghi chú quan trọng:

- `src/api/server.py` và `src/api/pipeline.py` là nhánh API gọn, dễ đọc nhưng ít tính năng hơn.
- `scripts/deployment/kiocmil_api_server.py` mới là API đầy đủ nhất cho KIOCMIL CADA.
- `src/api/kiocmil_inference.py` đang là nơi quan trọng nhất nếu muốn hiểu API KIOCMIL vận hành ra sao.

#### `src/data/` - xử lý dữ liệu mức thấp

- `src/data/image_io.py`
  Đọc/ghi ảnh và lấy kích thước ảnh theo `Path`.
- `src/data/loader.py`
  Load mapping dữ liệu.
- `src/data/utils.py`
  Helper để duyệt ảnh, parse nhãn YOLO, ghi split file, tạo YAML dataset.
- `src/data/filter.py`
  `DatasetFilter`; lọc dataset theo tần suất class và copy output.
- `src/data/splitter.py`
  `StratifiedSplitter`; chia train/val/test theo phân bố class.
- `src/data/preprocessing.py`
  Helper kiểu cũ cho preprocessing, scale bbox, flip và cân bằng bằng flip.

##### `src/data/filters/`

- `class_filters.py`
  Remap và filter class theo mapping.
- `empty_labels.py`
  Lọc các file nhãn rỗng.
- `__init__.py`
  Package marker.

##### `src/data/utils/`

- `yolo_utils.py`
  Hàm trung tâm để load/save/convert/expand bbox YOLO.
- `__init__.py`
  Package marker.

##### `src/data/balancing/`

- `augmentor.py`
  Flip và augment có điều chỉnh nhãn.
- `sampler.py`
  Thống kê phân bố class và tính target cân bằng.
- `validators.py`
  Kiểm tra image-label pair, validate nhãn YOLO, thống kê dataset.
- `yolo_utils.py`
  Helper xử lý label sau resize/filter/remap.
- `__init__.py`
  Export API balancing.

##### `src/data/preprocessing/`

- `pipeline.py`
  `PreprocessingPipeline`; bộ ghép nhiều bước preprocessing.
- `presets.py`
  Preset có sẵn cho resize/CLAHE/legacy/custom.
- `__init__.py`
  Export API mức package.

##### `src/data/preprocessing/core/`

- `base.py`
  Load/save/resize/normalize ảnh và scale bbox.
- `blur.py`
  Gaussian, median, bilateral blur.
- `clahe.py`
  CLAHE và histogram equalization.
- `augmentation.py`
  Flip và chỉnh brightness/contrast.
- `knee_crop.py`
  Crop vùng gối và biến đổi nhãn theo hệ tọa độ mới.
- `__init__.py`
  Export các phép xử lý core.

#### `src/datasets/` - dataset loader cho train/eval

- `src/datasets/yolo_dataset.py`
  Loader tổng quát nhất cho YOLO detection; hỗ trợ split file, cache, Albumentations, xuất bbox dạng YOLO hoặc Pascal VOC.
- `src/datasets/augmentation.py`
  Các transform conservative/moderate/validation cho YOLO.
- `src/datasets/converters.py`
  Chuyển đổi `YOLO <-> COCO`, sinh file JSON COCO.
- `src/datasets/coco_dataset.py`
  Loader COCO cho DETR và các mô hình detection kiểu transformer.
- `src/datasets/cdt_cad_dataset.py`
  Adapter cho CDT-CAD; đọc label YOLO, augment, trả tensor + target dict.
- `src/datasets/transforms.py`
  Processor/collate cho DETR.
- `src/datasets/samplers.py`
  `RepeatFactorSampler` để cân bằng instance hiếm.
- `src/datasets/kiocmil_dataset_v1.py`
  Dataset KIOCMIL đời cũ.
- `src/datasets/kiocmil_dataset_v2.py`
  Dataset KIOCMIL V2, augmentation pipeline đã chỉnh.
- `src/datasets/kiocmil_dataset_v3.py`
  Dataset quan trọng nhất cho CADA; trả về context patch, lesion patch JS/OST, bbox normalized và image-level label.
- `src/datasets/kiocmil_transforms_v1.py`
  Transform đời cũ cho KIOCMIL.
- `src/datasets/kiocmil_transforms_v2.py`
  Transform mới cho KIOCMIL; tách `GeometricAugmentation` và `PhotometricAugmentation`, có helper normalize/tensorize patch.
- `src/datasets/__init__.py`
  Export dataset và helper chính.

Ghi chú quan trọng:

- `src/datasets/kiocmil_dataset.py` không phải implementation đầy đủ; file này chỉ đóng vai trò “pointer”.
- `src/datasets/kiocmil_transforms.py` cũng là file “pointer”.
- Nếu đang làm việc với CADA, `src/datasets/kiocmil_dataset_v3.py` là file nên đọc đầu tiên.

#### `src/models/` - định nghĩa mô hình

- `src/models/kiocmil_model.py`
  KIOCMIL đời trước có `AttentionPool`.
- `src/models/kiocmil_model_v3.py`
  KIOCMIL V3, bước đệm trước CADA.
- `src/models/attention_modules.py`
  Tập hợp block cốt lõi cho CADA: `PositionalEncoding`, `DeformableAttention`, `CrossAttentionWithDeformable`, `ContextEncoder`, `LesionInstanceAggregation`, `FusionTransformer`.
- `src/models/kiocmil_model_cada.py`
  Model CADA chính; load backbone YOLO, dựng context encoder, lesion aggregation, fusion transformer, image-level attention và 3 head phân loại.

Ghi chú quan trọng về CADA:

- Code đã có đủ các block attention/deformable quan trọng.
- Tuy nhiên đường đi `forward()` hiện tại vẫn có phần fusion “nhẹ” kiểu local-context, chưa phải một luồng deformable attention đầy đủ ở mọi chỗ.
- Vì vậy khi phân tích model, nên đọc code thực tế thay vì chỉ dựa vào tài liệu mô tả kiến trúc.

##### `src/models/cdt_cad/`

- `cdt_cad_model.py`
  CDT-CAD hoàn chỉnh, backbone ResNet-50 + deformable transformer + head class/bbox.
- `feature_extractor.py`
  Khối trích đặc trưng context-aware dạng lặp.
- `deformable_transformer.py`
  Các layer transformer deformable tự cài đặt.
- `__init__.py`
  Export package CDT-CAD.

#### `src/losses/`

- `giou_loss.py`
  Box utility + `GIoULoss`.
- `focal_loss.py`
  Focal loss cho DETR kèm class-balanced weight.
- `cdt_cad_loss.py`
  `HungarianMatcher` và loss tổng hợp cho CDT-CAD.
- `__init__.py`
  Package marker.

#### `src/training/`

- `src/training/train_kiocmil_cada.py`
  Trainer chính cho CADA; setup dataset, transform, sampler, optimizer, scheduler, early stopping, checkpoint và wandb.
- `src/training/evaluate_kiocmil_cada.py`
  Đánh giá model CADA và sinh metric/report.
- `src/training/train_kiocmil_v1.py`, `train_kiocmil_v2.py`, `train_kiocmil_v3.py`
  Trainer cho các đời model cũ.
- `src/training/evaluate_kiocmil.py`
  Evaluate cho nhánh KIOCMIL cũ.
- `src/training/detr_trainer.py`
  Trainer cho DETR dùng COCO json + processor HuggingFace.
- `src/training/focal_loss.py`
  Focal loss cho bài toán classification trong KIOCMIL.
- `src/training/early_stopping.py`
  Callback early stopping.
- `src/training/trainer.py`
  Base trainer tối giản.

Ghi chú:

- `src/training/train_kiocmil.py` không phải implementation đầy đủ; đây cũng là file “pointer”.
- Nếu muốn hiểu nhanh cách train CADA, `train_kiocmil_cada.py` là file quan trọng nhất.

#### `src/evaluation/`, `src/analysis/`, `src/visualization/`, `src/utils/`

- `src/evaluation/evaluator.py`
  `YOLOEvaluator`; tạo temp YAML và gọi `model.val()` của Ultralytics.
- `src/analysis/generate_experiment_report.py`
  Tổng hợp `metrics.json` thành `docs/EXPERIMENT_REPORT_FULL.md`.
- `src/visualization/gradcam.py`
  GradCAM cho YOLO, phục vụ giải thích và visualize.
- `src/utils/visualization.py`
  Vẽ confusion matrix, ROC, PR, training curves.
- `src/utils/dataset_viz.py`
  Vẽ bbox, phân bố class, grid ảnh, phân bố kích thước bbox.
- `src/utils/logging_utils.py`
  Tạo `log/run_*`, lưu `config.json`.
- `src/utils/yaml_config.py`
  Tạo/đọc/cập nhật/validate YAML cho YOLO.
- `src/utils/bbox.py`
  Helper `yolo_to_xyxy_norm`.
- `src/utils/__init__.py`
  Export shared utilities.

### 5.2 `scripts/` - entry point và orchestration

Hiện tại có khoảng `51` file Python, chưa tính `.ps1` và `.sh`.

#### `scripts/pipelines/` - data pipeline

**Ghi chú:** Các shell script orchestration pipeline (.ps1, .sh) không có trong checkout này. Các bước dữ liệu được thực thi bằng cách gọi Python script trực tiếp từ thư mục `data_preparation/`.

Để chạy pipeline từng bước, gọi các Python script trong `data_preparation/` trực tiếp với các argument tương ứng (xem phần `data_preparation/` bên dưới).

#### `data_preparation/` - các script chuẩn bị dữ liệu

Các script Python nằm trực tiếp trong thư mục `data_preparation/` (không phải `scripts/data_preparation/`):

- `import_data_images_e.py`
  Import archive `Data_Images_E` vào `datasets/datasets/dataset_v0`.
- `generate_knee_labels_from_kl.py`
  Sinh `labels-knee` từ nhãn KL/lesion.
- `crop_knee_regions.py`
  Crop knee từ full X-ray và biến đổi nhãn lesion sang tọa độ crop.
- `filter_no_labels.py`
  Bỏ ảnh crop không có label.
- `filter_kl0.py`
  Bỏ class KL0, dùng để tạo 4-class hoặc 8-class.
- `filter_dataset.py`
  Lọc dataset theo phân bố class.
- `balance_dataset.py`
  Oversampling/flip để cân bằng, đồng bộ label phụ nếu có.
- `split_dataset.py`
  Chia stratified split với multi-label stratification.
- `create_splits_v2.py`
  Helper chia split tối giản.
- `remap_labels.py`
  Remap class id sau khi lọc.
- `validate_dataset.py`
  Validate workflow chuẩn bị dữ liệu.

#### `scripts/preprocessing/` - không có trong checkout này

Preprocessing được thực thi thông qua các module trong `src/data/preprocessing/` thay vì script riêng.

#### `scripts/training/` - training scripts

Hiện tại chỉ có:

- `train_yolo.py`
  Train YOLO11 theo config YAML. Đây là entrypoint chính cho training YOLO.

**Ghi chú:** Các script training khác (train_yolo_enhanced.py, train_knee_detection.py, train_detr.py, train_cdt_cad.py, các shell wrapper, etc.) không có trong checkout này.

#### `scripts/evaluation/` - không có trong checkout này

Evaluation được thực thi thông qua `src/evaluation/evaluator.py` hoặc model.val() của Ultralytics.

#### `scripts/deployment/` - không có trong checkout này

**Để chạy API nhẹ:**

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

API entrypoint là `src/api/server.py` với module chính là `src/api/pipeline.py`.

#### `scripts/testing/` - không có trong checkout này

Testing utilities được giữ trong `src/` hoặc module tương ứng.

#### `scripts/visualization/` - không có trong checkout này

Visualization utilities nằm trong `src/utils/visualization.py` và `src/utils/dataset_viz.py`.

#### `scripts/analysis/`, `scripts/analyzes/`, `scripts/create_reports/`, `scripts/legacy/` - không có trong checkout này

Analysis và reporting được thực thi thông qua `src/analysis/generate_experiment_report.py`.

### 5.3 `dataset_analysis/` - bộ công cụ dataset

**Đường dẫn thực tế:** `dataset_analysis/` nằm ở root (không phải `tools/dataset_analysis/`).

Hiện có các file chính:

- `dataset_analysis/validate.py`
  Validate label file và image-label pairing.
- `dataset_analysis/analyze.py`
  Phân tích dataset tổng hợp và sinh report/hình.
- `dataset_analysis/class_split_report.py`
  File rất quan trọng để tách nhãn thành nhánh con `a/b` dựa trên hình học và vị trí.
- `dataset_analysis/visualize/samples.py`
  Vẽ overlay label lên sample.
- `dataset_analysis/visualize/class_samples.py`
  Lấy sample theo class để xem nhanh.
- `dataset_analysis/visualize/augmentations.py`
  Visualize tác động augmentation.
- `dataset_analysis/README.md`
  Tài liệu cho bộ công cụ.

### 5.4 File Python ở root

- `utils.py`
  Chỉ có một helper nhỏ `yolo_to_xyxy_norm`; không phải nơi logic chính đang sống.

## 6. Config, tài liệu và startup script

### `configs/`

Các nhóm config nổi bật:

- `yolo_5_class_baseline.yaml`, `yolo_10_class_baseline.yaml`, `yolo_4_class_baseline.yaml`, `yolo_8_class_baseline.yaml`
  Config huấn luyện cơ bản.
- `yolo_*_conservative.yaml`
  Cùng nhóm trên nhưng có conservative augmentation.
- `yolo_dataset_v0_*`
  Config cho full X-ray.
- `yolo_enhanced_*`
  Config cho các thử nghiệm preprocessing/enhanced training.
- `yolo_knees_cropped_local.yaml`
  Config local cho cropped knee.

### Tài liệu sẵn có trong `docs/`

- `README.md`
  Index tài liệu.
- `DATASETS.md`
  Tổng hợp dataset.
- `KIOCMIL_CADA_REFERENCE.md`
  Mô tả kiến trúc CADA.
- `KIOCMIL_CADA_TRAINING.md`, `TRAINING_INDEX.md`
  Hướng dẫn huấn luyện.
- `API_README.md`, `KNEE_YOLO_API.md`
  Mô tả API.
- `reference/`, `experiments/`, `logs/`
  Tham chiếu cấu trúc, kết quả, report thực nghiệm.

### Startup script ở root

- `setup_env.bat`, `setup_env.sh`
  Tạo môi trường làm việc.
- `start_api.bat`, `start_api.sh`
  Chạy nhanh API pipeline.
- `start_knee_yolo_api.bat`
  Chạy nhanh API knee detector.
- `test_setup.bat`, `fix_pytorch.bat`
  Utility setup/fix nhanh.

## 7. Dữ liệu, artifact và snapshot

### `datasets/`

Đây là nơi dữ liệu thực tế đang được dùng nhiều nhất:

- `datasets/dataset_v0/`
  Full X-ray gốc đã import.
- `datasets/dataset_knees_cropped/`
  Crop đầu gối từ full X-ray.
- `datasets/dataset_knees_cropped_4_class/`, `datasets/dataset_v0_4_class/`
  Biến thể đã lọc KL0.
- `datasets/balanced/`
  Các dataset đã cân bằng.
- `datasets/processed_balanced/`
  Các biến thể preprocessing sau khi cân bằng.
- `datasets/splits/`
  Split file cho nhiều biến thể dataset.
- `datasets/data_examples/`
  Mẫu dữ liệu.

### `runs/`, `log/`, `logs/`, `wandb/`

- `runs/kiocmil_cada/`
  Checkpoint và metric của CADA.
- `log/`
  Log train dạng versioned.
- `logs/`
  Log API và log tổng hợp.
- `wandb/`
  Artifact phục vụ tracking.

### `weight/` và weight ở root

- `weight/yolo11l.pt`
  Backbone/model dùng lại trong nhiều nơi.
- `weight/custom_data.yaml`
  Config data phụ.
- `yolo11l.pt`, `knee_yolo11n_*.zip`, `lesion_8class_balanced.zip`
  Artifact mô hình hoặc gói output.

### `kiocmil_cada_v1/`

Thư mục này là một snapshot tương đối độc lập:

- có `src/`, `scripts/`, `processed/`, `splits/`, `runs/`, `log/`, `docs/`
- phục vụ tái hiện hoặc giữ lại phiên bản CADA cũ
- không phải nơi ưu tiên sửa nếu bạn đang maintain repo chính hiện tại

## 8. Những điểm cần để ý khi đọc hoặc sửa repo

1. **Thực tế vs. Tài liệu:** Repo này chứa chủ yếu là data preparation + YOLO training + API nhẹ. Các nhánh mô hình nâng cao (KIOCMIL CADA, DETR, CDT-CAD) được tham chiếu trong tài liệu nhưng không có trong checkout này.

2. Repo trộn cả `code + data + checkpoint + log`.
   Cần phân biệt rõ `source` với `artifact` trước khi sửa.

2. Có hai tầng API khác nhau.
   `src/api/server.py` là pipeline gọn.
   `scripts/deployment/kiocmil_api_server.py` + `src/api/kiocmil_inference.py` là nhánh API đầy đủ cho CADA.

3. Có các file “pointer” thay vì implementation thật.
   Các file nổi bật:
   `src/datasets/kiocmil_dataset.py`
   `src/datasets/kiocmil_transforms.py`
   `src/training/train_kiocmil.py`

4. Một số tài liệu hoặc script đã mang dấu vết lịch sử.
   Ví dụ `run_full_pipeline.sh` vẫn nhắc tới `tools/check_dataset/*` trong khi code hiện hành nằm ở `tools/dataset_analysis/*`.

5. Phần mô tả CADA trong docs và phần triển khai thực tế không hoàn toàn trùng tuyệt đối ở mọi đoạn.
   Khi cần quyết định sửa model, nên lấy code thực tế làm chuẩn.

6. `datasets/` mới là trung tâm dữ liệu hiện tại.
   `processed/` ở root gần như không còn đóng vai trò chính.

7. Repo hiện khá hợp với môi trường Windows/PowerShell.
   Nếu đang làm việc ngay trên máy này, nên ưu tiên các script `.ps1`.

## 9. Nên bắt đầu đọc từ đâu

### Nếu muốn hiểu toàn bộ luồng dữ liệu

Đọc theo thứ tự:

1. `scripts/pipelines/setup_from_data_images_e.ps1`
2. `scripts/data_preparation/import_data_images_e.py`
3. `scripts/data_preparation/generate_knee_labels_from_kl.py`
4. `scripts/data_preparation/crop_knee_regions.py`
5. `tools/dataset_analysis/class_split_report.py`
6. `scripts/data_preparation/balance_dataset.py`
7. `scripts/data_preparation/split_dataset.py`

### Nếu muốn train YOLO

Đọc theo thứ tự:

1. `configs/*.yaml`
2. `scripts/training/train_yolo.py`
3. `src/datasets/yolo_dataset.py`
4. `src/datasets/augmentation.py`
5. `src/evaluation/evaluator.py`

### Nếu muốn train KIOCMIL CADA

Đọc theo thứ tự:

1. `src/datasets/kiocmil_dataset_v3.py`
2. `src/datasets/kiocmil_transforms_v2.py`
3. `src/models/attention_modules.py`
4. `src/models/kiocmil_model_cada.py`
5. `src/training/train_kiocmil_cada.py`
6. `src/training/evaluate_kiocmil_cada.py`

### Nếu muốn chạy API

Đọc theo thứ tự:

1. `scripts/deployment/kiocmil_api_server.py`
2. `src/api/kiocmil_inference.py`
3. `src/api/response_schemas.py`
4. `src/api/logger_config.py`
5. `src/api/gradcam.py`

### Nếu muốn phân tích dataset

Đọc theo thứ tự:

1. `tools/dataset_analysis/analyze.py`
2. `tools/dataset_analysis/validate.py`
3. `tools/dataset_analysis/compare.py`
4. `tools/dataset_analysis/visualize/*`

## 10. Kết luận ngắn

Về bản chất, đây là một repo nghiên cứu/ứng dụng kiểu “all-in-one”:

- có code train và deploy,
- có data pipeline,
- có tool validate/analyze,
- có artifact thực nghiệm và snapshot cũ ngay trong cùng workspace.

Khi maintain lâu dài, nên xem:

- `src/ + scripts/ + tools/ + configs/` là lớp logic thật,
- `datasets/ + runs/ + logs/ + wandb/ + *.pt/*.zip` là lớp tài nguyên và kết quả.
