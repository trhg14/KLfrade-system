# Pipeline Chi Tiết 

## 1. Cấu trúc thư mục dự án
code/
├── data_preparation/         # Script xử lý dữ liệu
├── dataset_analysis/         # Công cụ phân tích dataset
├── datasets/
│   └── datasets/
│       ├── dataset_v0/                      # Dữ liệu gốc
│       ├── dataset_knees_cropped_p15/        # Crop đầu gối
│       ├── dataset_knees_cropped_p15_4class/ # Nhãn 4-class
│       ├── dataset_knees_cropped_p15_8class/ # Nhãn 8-class
│       ├── balanced/knees_cropped_8class/    # Balanced dataset
│       └── splits/knees_cropped_8class/      # Train/val/test split
├── configs/                  # YAML config training
├── scripts/training/         # Script huấn luyện YOLO
└── src/
    ├── api/                  # API server và inference
    └── data/                 # Data utilities
## 2. Tổng quan

Pipeline của dự án có thể chia thành 5 lớp:

1. **Chuẩn bị dữ liệu gốc**
2. **Huấn luyện mô hình**
3. **Đánh giá và tổng hợp kết quả**
4. **Suy luận và triển khai API**

luồng end-to-end:

```text
Zip dữ liệu gốc (Data_Images_E.zip)
  -> import vào datasets/datasets/dataset_v0
  -> sinh labels-knee (padding=1.5)
  -> crop đầu gối (min_size=150)
  -> lọc KL0 (sinh 4-class)
  -> sinh nhãn 8-class
  -> cân bằng dữ liệu
  -> tạo train/val/test split
  -> train mô hình YOLO #1 (knee detector)
  -> train mô hình YOLO #2 (KL Grade 8-class)
  -> evaluate 
  -> deploy API 2-stage
```

## 3. Pipeline dữ liệu chi tiết

### 3.1. Bước 0 - Import dữ liệu gốc

**Script chính**

- `data_preparation/import_data_images_e.py`

**Đầu vào**

- File nén: `Data_Images_E.zip`

**Đầu ra**

- `datasets/datasets/dataset_v0/`

**Mục tiêu**

- Giải nén và chuẩn hóa dữ liệu ban đầu thành cấu trúc mà project có thể dùng được.

**Cấu trúc dữ liệu đích**

```text
dataset_v0/
  ├── images/  <- 1489 ảnh full X-ray
  ├── labels/  <- 1490 nhãn 5-class gốc
  └── ...
```

**Ý nghĩa**

- `dataset_v0` là bộ dữ liệu full X-ray gốc sau khi import.
- Đây là điểm xuất phát của toàn bộ pipeline downstream.

---

### 3.2. Bước 0b - Sinh `labels-knee`

**Script**

- `data_preparation/generate_knee_labels_from_kl.py`

**Đầu vào**

- `datasets/datasets/dataset_v0/labels/`

**Đầu ra**

- `datasets/datasets/dataset_v0/labels-knee/`

**Kết quả**

- `1490 files, 2446 knee boxes`

**Mục tiêu**

- Tạo bounding box vùng đầu gối từ nhãn KL/lesion có sẵn.

**Ý nghĩa**

- Bước crop phía sau cần viết vị trí đầu gối để crop đúng vùng
- padding=1.5 là bắt buộc vì nhãn gốc rất nhỏ so với kích thước ảnh (750x1512 đến 3488x4256px) - nếu không có padđing đủ lớn, crop sẽ bị quá nhỏ và không dùng được

---

### 3.3. Bước 1 - Crop đầu gối từ full X-ray

**Script**

- `data_preparation/crop_knee_regions.py`

**Đầu vào**

- `datasets/datasets/dataset_v0/images/`
- `datasets/datasets/dataset_v0/labels/`
- `datasets/datasets/dataset_v0/labels-knee/`

**Đầu ra**

- `datasets/datasets/dataset_knees_cropped_p15/`

**Cấu trúc đầu ra**

```text
dataset_knees_cropped_p15/
  ├── images/        <- 2002 ảnh crop
  ├── labels/        <- 2002 nhãn
  ├── labels-knee/
  └── crop_report.json
```

**Logic chính**

1. Đọc ảnh full X-ray.
2. Đọc các box đầu gối từ `labels-knee`.
3. Với mỗi box đầu gối:
   - chuyển tọa độ YOLO (cx,cy,w,h) sang tọa độ pixel (x1,y1,x2,y2),
   - mở rộng box đầu gối nếu cần để không cắt mất lesion liên quan,
   - crop thành ảnh đầu gối riêng,
   - biến đổi nhãn từ tọa độ full-image sang tọa độ crop.
4. Kiểm tra kích thước crop - bỏ qua nếu nhỏ hơn min_size=150px.
5. Lưu ảnh crop và nhãn đã biến đổi.

**Điểm quan trọng**

- Mỗi ảnh có 2 đầu gối sẽ sinh 2 crop riêng biệt (*_knee0, *_knee1), giúp tăng số mẫu từ 1.489 lên 2.002.
- padding=1.5 ở bước 0b là yếu tố quyết định — không đủ padding thì knee box quá nhỏ, crop bị loại hàng loạt.
- min_size=150 được chọn sau thử nghiệm, kết hợp với padding=1.5 đạt tỷ lệ crop thành công 81.8% (2.002/2.447).
- Nhãn KL được tự động biến đổi tọa độ sang không gian crop, đảm bảo nhãn chính xác sau khi crop.

---

### 3.4. Bước 2a - Lọc KL0 (sinh 4 class)

**Script**

- `data_preparation/filter_kl0.py`

**Đầu vào**

- Cropped knees:
  - `datasets/dataset_knees_cropped_p15/`

**Đầu ra**

- Cropped knees:
  - `datasets/datasets/dataset_knees_cropped_p15_4class/`

**Kết quả**

- 1924 ảnh giữ lại (96.1%).
- 78 ảnh KL0 bị lọc (3.9%)

**Mục tiêu**

- Loại bỏ các mẫu KL0 (không có bệnh)khỏi dataset trước khi tách a/b.

**ý nghĩa**

- KL0 là trường hợp bình thường, không có tổn thương rõ ràng để phân biệt osteophyte (a) và joint space narrowing (b). Giữ KL0 trong quá trình tách a/b sẽ gây nhiễu và làm giảm chất lượng nhãn 8-class
- Sau bước này chỉ còn các cấp độ bệnh lý KL1-Kl4 phục vụ grading.

---

### 3.5. Bước 2b - Sinh nhãn 8-class

**Script**

- `dataset_analysis/class_split_report.py`

**Đầu vào**

- `datasets/datasets/dataset_knees_cropped_p15_4class/labels/`

**Đầu ra**

- `datasets/datasets/dataset_knees_cropped_p15_8class/labels/`

**Kết quả** 

- Tách mỗi cấp độ KL thành 2 nhánh con dựa trên hình học bounding box:
    a: osteophyte (gai xương) — bbox thường nằm ngang, rộng hơn cao
    b: joint space narrowing (hẹp khe khớp) — bbox thường đứng, cao hơn rộng

**Mục tiêu**

- Sinh nhãn 8-class bằng cách tách KL1–KL4 thành nhánh a/b.

**Logic chính**

1. Đọc từng bounding box trong nhãn KL1–KL4 (class id 0–3).
2. Tính tỉ lệ width/height của box để phân nhánh:
- w > h → osteophyte → nhãn *-a
- h > w → joint space narrowing → nhãn *-b
3. Ánh xạ sang class id mới (0–7), giữ nguyên tọa độ box, ghi ra file nhãn 8-class.

**Điểm quan trọng**

- Việc phân nhánh a/b dựa hoàn toàn vào hình học bounding box (tỉ lệ w/h), không cần annotation thủ công thêm.
- KL0 đã bị loại ở bước 2a — vì KL0 không có tổn thương rõ ràng, không thể phân biệt a/b có ý nghĩa lâm sàng.
- Tọa độ box giữ nguyên, chỉ thay class id → file nhãn 8-class tương thích hoàn toàn với định dạng YOLO.
---

### 3.6. Bước 3 - Cân bằng dữ liệu

**Script**

- `data_preparation/balance_dataset.py`
- `src/data/balancing/`

**Đầu vào**

- images: `datasets/datasets/dataset_knees_cropped_p15_4class/images/`
- labels: `datasets/datasets/dataset_knees_cropped_p15_8class/labels/`

**Đầu ra**

- `datasets/datasets/balanced/knees_cropped_8class/`

**Kết quả**

- Trước balance:
    Class 0 (KL1-a):  458 instances
    Class 1 (KL1-b):   80 instances
    Class 2 (KL2-a): 1105 instances
    Class 3 (KL2-b):  133 instances
    Class 4 (KL3-a):  480 instances
    Class 5 (KL3-b):  100 instances
    Class 6 (KL4-a):  250 instances
    Class 7 (KL4-b):   50 instances

- Sau balance:
    Tổng ảnh: 2324
    Tổng instances: 3057
    Target: 191 instances/class (median)

**Ý nghĩa**

- Dataset 8-class mất cân bằng nghiêm trọng: KL2-a chiếm 41.6% trong khi KL4-b chỉ 1.88% — nếu không cân bằng, model sẽ thiên lệch về class đông mẫu và bỏ qua class hiếm.
- Horizontal flip được chọn vì cấu trúc giải phẫu đầu gối đối xứng, không làm mất tính hợp lệ của nhãn.

**Logic chính**

1. Thống kê số mẫu theo class.
2. Tính target cân bằng theo median.
3. Oversample class ít mẫu bằng horizontal flip.
4. Lưu dataset cân bằng ra thư mục mới.

---

### 3.7. Bước 4 - Tạo train/val/test split

**Script**

- `data_preparation/split_dataset.py`

**Tỉ lệ mặc định**

- Train: `70%`
- Validation: `15%`
- Test: `15%`

**Seed mặc định**

- `42`

**Đầu vào**

- `datasets/datasets/balanced/knees_cropped_8class/`

**Đầu ra**

- `datasets/datasets/splits/knees_cropped_8class/`

**Logic chính**

1. Duyệt image + label.
2. Với mỗi ảnh, tổng hợp danh sách class xuất hiện trong nhãn tương ứng.
3. Tạo split theo hướng **stratified multi-label**.
4. Cố đảm bảo:
   - giữ phân bố class tương đối đồng đều,
   - không làm mất class hiếm ở val/test.

**Ý nghĩa**

- Stratified split giúp đảm bảo các class hiếm (KL1-b, KL4-b) đều xuất hiện trong cả 3 tập train/val/test, tránh trường hợp val/test không có đủ mẫu để đánh giá.
- Seed cố định (42) đảm bảo kết quả tái lập được.
- Đây là bước chốt để dataset trở thành "ready for training".
---

## 4. Pipeline huấn luyện mô hình

Sau khi có dataset và split, project đi vào pha train.

### 4.1. YOLO11n - Knee Detector

**Đầu vào**

- Ảnh full X-ray từ `dataset_v0/images/`
- Nhãn từ `dataset_v0/labels-knee/`
- 1 class: knee

**Script chính:**

- `scripts/training/train_yolo.py`

**Đầu ra**

- `src/data/knee_yolo11n/.../weights/best.pt`

**Luồng bên trong**

1. Nhận ảnh full X-ray, resize về kích thước chuẩn YOLO11n.
2. Trích xuất đặc trưng qua backbone YOLO11n.
3. Head dự đoán bbox và confidence cho class knee.
4. NMS lọc box trùng lặp, giữ lại box confidence cao nhất cho mỗi đầu gối.
5. Xuất tọa độ bbox từng đầu gối → đưa sang bước crop phục vụ YOLO #2.

---

### 4.2. YOLO11l - KL Grader 8-class

**Script chính** 

- `scripts/training/train_yolo.py`
- `configs/yolo_8_class_baseline.yaml`

**Dữ liệu đầu vào mô hình**

- datasets/datasets/splits/knees_cropped_8class/
- 8 class: KL1-a đến KL4-b

**Luồng bên trong**

1. Nhận ảnh crop đầu gối từ YOLO #1, resize về kích thước chuẩn YOLO11n.
2. Trích xuất đặc trưng qua backbone YOLO11n.
3. Head dự đoán bbox và confidence cho 8 class tổn thương (KL1-a đến KL4-b).
4. Tính loss huấn luyện gồm Box loss (CIoU), Class loss (BCE) và DFL loss.
5. NMS lọc box trùng lặp, giữ lại box confidence cao nhất cho mỗi class.
6. Xuất kết quả: tọa độ bbox + class tổn thương + confidence → đưa vào API.

**Đầu ra**

- `lesion_8class_balanced/.../weights/best.pt`
- `lesion_8class_balanced/.../results.csv`

---

### 4.3. Pipeline đánh giá

**Thành phần chính**

- Output tự động của Ultralytics sau training.

**Đầu ra**

lesion_8class_balanced/.../
  ├── results.csv          ← metrics theo epoch
  ├── confusion_matrix.png ← ma trận nhầm lẫn
  ├── PR_curve.png         ← precision-recall curve
  ├── BoxP_curve.png
  ├── BoxR_curve.png
  └── BoxF1_curve.png

## 5. Pipeline suy luận và API

### Đánh giá API 2-stage

**Thành phần chính**
- `src/api/server.py`
- `src/api/pipeline.py`
- `src/api/inference.py`

**Luồng**

1. Upload ảnh X-ray.
2. YOLO #1 detect knee bbox.
3. crop từng đầu gối.
4. YOLO11l detect KL grade trên crop.
5. trả JSON: bbox + class + confidence.

ví dụ
{
  "knee_bbox": [1308, 1947, 2382, 2672],
  "knee_conf": 0.768,
  "kl_grade": {
    "grade_class": "KL4-a",
    "grade_conf": 0.877,
    "grade_id": 6,
    "grade_bbox": [1383, 2359, 1470, 2503]
  }
}

**Ý nghĩa**
- API nhận ảnh X-ray thô và trả về kết quả grading KL đầy đủ chỉ trong một request, bao gồm vị trí đầu gối, loại tổn thương và độ tin cậy.
- Thiết kế 2-stage giúp pipeline linh hoạt — có thể thay thế từng model riêng biệt mà không ảnh hưởng đến phần còn lại.

---

## 6. Quan hệ giữa các pipeline

### Pipeline dữ liệu

Chuẩn bị và xử lý dữ liệu đầu vào cho training.

### Pipeline train

Nhận dữ liệu đã xử lý, huấn luyện YOLO #1 và YOLO #2 tuần tự.

### Pipeline evaluate

Chạy song song với training, đánh giá model sau mỗi epoch.

### Pipeline inference/API

Nhận checkpoint từ training, đóng gói thành REST API phục vụ inference thực tế.

Nói cách khác:

```
Data pipeline
  -> Training pipeline
    -> Evaluation pipeline
    -> Inference/API pipeline
```

## 7. Thư mục đầu ra theo từng giai đoạn

### Sau khi chuẩn bị dữ liệu

- `datasets/datasets/dataset_v0/`
- `datasets/datasets/dataset_knees_cropped_p15/`
- `datasets/datasets/dataset_knees_cropped_p15_4class/`
- `datasets/datasets/dataset_knees_cropped_p15_8class/`
- `datasets/datasets/balanced/knees_cropped_8class/`
- `datasets/datasets/splits/knees_cropped_8class/`

### Sau khi huấn luyện

- `src/data/knee_yolo11n/.../weights/best.pt`
- `lesion_8class_balanced/.../weights/best.pt`
- `lesion_8class_balanced/.../results.csv`

### Sau khi chạy API

- `logs/`

## 8. Kết luận

Pipeline là một **chuỗi end-to-end hoàn chỉnh** từ ảnh X-ray gốc đến REST API, được tổ chức theo 4 trục: 

1. **Data pipeline**: từ full X-ray qua crop, lọc, sinh nhãn 8-class, cân bằng đến split.
2. **Training pipeline**: YOLO #1 knee detection → YOLO #2 KL grading 8-class.
3. **Evaluation pipeline**: đánh giá qua mAP, precision, recall, confusion matrix.
4. **Inference pipeline**: tái dựng đúng schema 2-stage để suy luận ổn định trên API.