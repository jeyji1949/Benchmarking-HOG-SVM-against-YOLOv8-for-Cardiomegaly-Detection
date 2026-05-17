# 🫀 BioVision Cardio

> **Classical meets Clinical** — Benchmarking HOG+SVM against YOLOv8 for Cardiomegaly Detection from Chest X-rays

---

## 📋 Overview

**BioVision Cardio** is a comparative machine learning project that investigates two fundamentally different approaches to detecting **Cardiomegaly** (enlarged heart) from annotated Chest X-ray images:

| Approach | Method | Philosophy |
|----------|--------|------------|
| 🔬 Classical ML | HOG + LinearSVM | Hand-crafted features → explicit decision boundary |
| 🤖 Deep Learning | YOLOv8 | End-to-end learned features → bounding box regression |

The goal is not just to build models, but to **understand and compare** how each method perceives medical images — and what that means for clinical deployment.

---

## ⚠️ Environment Note

> This project was developed across **two different environments** due to hardware constraints:

| Part | Environment | Reason |
|------|-------------|--------|
| 🔬 HOG + SVM pipeline | **Local machine** | Lightweight — runs on CPU, no GPU needed |
| 🤖 YOLOv8 training | **Kaggle (GPU T4)** | Requires GPU — local machine lacks sufficient GPU resources |

If you want to **reproduce this project**:
- Run `hog_svm_pipeline.ipynb` locally (requires only CPU)
- Run `yolov8_training.ipynb` on **Kaggle** with GPU T4 enabled (`Settings → Accelerator → GPU T4 x2`)

---

## 🩺 The Disease: Cardiomegaly

Cardiomegaly is clinically defined by a **Cardiothoracic Ratio (CTR) > 0.5**:

```
CTR = Heart width / Thoracic cage width > 0.5
```

An enlarged heart silhouette is **geometrically distinctive** on a chest X-ray, making it an ideal benchmark disease because:
- It has a clear, measurable binary definition (Normal vs. Enlarged)
- It produces strong gradient patterns at the cardiac border
- HOG can capture this structural signature directly

---

## 📁 Project Structure

```
BioVision-Cardio/
│
├── 📓 notebooks/
│   ├── hog_svm_pipeline.ipynb          # HOG+SVM training (run locally)
│   ├── hog_svm_adapted.ipynb           # HOG+SVM adapted for VinBigData
│   ├── model_verification.ipynb        # Cross-validation & visualization
│   └── yolov8_training.ipynb           # YOLOv8 training (run on Kaggle)
│
├── 📊 results/
│   ├── hog_svm_confusion.png           # HOG+SVM confusion matrix
│   ├── hog_svm_cross_validation.png    # 5-fold CV results
│   ├── hog_svm_score_distribution.png  # Decision score distribution
│   ├── yolov8_predictions.png          # YOLOv8 val set predictions
│   ├── normal_test.png                 # False alarm test results
│   └── external_test.png              # External images test
│
├── 📄 docs/
│   └── Cardio_Analysis_Report.docx     # Full technical documentation
│
└── README.md
```

---

## 🗂️ Datasets

### ⚠️ Dataset Migration — Both Approaches

We initially started with **`spritan1/yolo-annotated-chestxray-8-object-detection`** for both HOG+SVM and YOLOv8, but **switched to VinBigData for both approaches** due to the following reasons:

| Issue | Original Dataset | VinBigData |
|-------|-----------------|------------|
| Cardiomegaly cases (train) | ~124 | **1,757** |
| Total images | 631 | **11,250** |
| Val Cardiomegaly cases | ~29 | **447** |
| YOLOv8 mAP50 | 0.063 ❌ | **0.879** ✅ |
| HOG+SVM val recall | ~52% ❌ | **95%** ✅ |

> **Conclusion:** The original dataset was too small for both approaches to learn meaningful patterns. VinBigData provided **14x more Cardiomegaly cases**, dramatically improving results across the board.

---

### Final Dataset — VinBigData (Both HOG+SVM and YOLOv8)
**Source:** [`buithanhxuan/vinbigdata-yolo-dataset-with-wbf-3x-downscaled`](https://www.kaggle.com/datasets/buithanhxuan/vinbigdata-yolo-dataset-with-wbf-3x-downscaled)

| Split | Images | Cardiomegaly cases |
|-------|--------|--------------------|
| Train | 11,250 | **1,757** |
| Val | 3,000 | **447** |

**Classes (15 total):**

| ID | Disease |
|----|---------|
| 0 | Aortic_enlargement |
| **3** | **Cardiomegaly ← our target** |
| 14 | No finding |
| ... | ... |

---

## ⚙️ HOG + SVM Pipeline

### How it works

```
Chest X-ray PNG
      │
      ▼
Parse YOLO .txt  →  Extract bounding box coordinates
      │
      ▼
Crop heart region  →  Resize to 128×128  →  Grayscale
      │
      ▼
HOG Descriptor  →  Feature vector
   (12 orientations, 16×16 cells, 2×2 blocks)
      │
      ▼
Random Oversampling  →  Balance classes
      │
      ▼
LinearSVC (C=1.0)  →  Decision boundary
      │
      ▼
Prediction: Cardiomegaly (1) or Other (0)
```

### HOG Parameters

| Parameter | Value |
|-----------|-------|
| `orientations` | 12 |
| `pixels_per_cell` | (16, 16) |
| `cells_per_block` | (2, 2) |
| `image_size` | 128×128 |

### Results — HOG + SVM (VinBigData)

```
Val Set Cardiomegaly Results:
  Total found : 476
  Correct     : 452 (95.0%)
  Wrong       : 24

5-Fold Cross Validation F1 : 0.935 ± 0.02
```

---

## 🤖 YOLOv8 Pipeline

> ⚠️ **Run this notebook on Kaggle with GPU T4 enabled.**

### Models trained

| Version | Model | Classes | mAP50 | Notes |
|---------|-------|---------|-------|-------|
| v1 | `yolov8n` | Cardiomegaly only | **0.879** | Best for comparison |
| v2 | `yolov8s` | All 15 classes | **0.891** | Best for production |
| v3 | `yolov8s` | Cardiomegaly only | 0.861 | Larger ≠ better here |

### Training Configuration (v1 — used for comparison)

```python
model.train(
    data='vinbig.yaml',
    epochs=50,
    imgsz=512,
    batch=16,
    device=0,       # GPU T4
    classes=[3],    # Cardiomegaly only
)
```

### Results — YOLOv8n v1

```
Cardiomegaly detection:
  mAP50     : 0.879
  Recall    : 0.799
  Precision : 0.835

False alarm test (50 normal images) : 0/50 ✅
External images test (6 images)     : 6/6 correct ✅
```

---

## 📊 Final Comparison — HOG+SVM vs YOLOv8

| Dimension | HOG + SVM | YOLOv8n (v1) |
|-----------|-----------|--------------|
| **Recall (Cardiomegaly)** | **95.0%** | 79.9% |
| **mAP50** | — | **0.879** |
| **Localization** | ❌ Manual crop needed | ✅ Automatic |
| **False alarms** | Low | **0/50 (0%)** |
| **Training data needed** | ~500 crops | ~1700+ images |
| **Inference speed** | **~1ms/image** | ~1.2ms/image |
| **Interpretability** | ✅ HOG visualizable | ❌ Black box |
| **GPU required** | ❌ CPU only | ✅ Recommended |
| **Clinical analogy** | Tracing the heart border with a ruler | Radiologist scanning the full X-ray |

### Key Insight

> *HOG+SVM achieves higher recall (95%) when given the correct heart crop, while YOLOv8 automatically localizes the heart but misses ~20% of cases. The hybrid approach — using YOLOv8 to localize, then HOG+SVM to classify — combines the strengths of both.*

---

## 🚀 How to Run

### HOG + SVM (Local Machine)

```bash
# Install dependencies
pip install opencv-python scikit-image scikit-learn numpy matplotlib joblib

# Run notebook
jupyter notebook notebooks/hog_svm_adapted.ipynb
```

### YOLOv8 (Kaggle — GPU Required)

```
1. Go to kaggle.com and create a new notebook
2. Add dataset: buithanhxuan/vinbigdata-yolo-dataset-with-wbf-3x-downscaled
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Enable Internet: Settings → Internet → ON
5. Run notebooks/yolov8_training.ipynb
```

### Dependencies

```
# Local (HOG+SVM)
opencv-python
scikit-image
scikit-learn
numpy
matplotlib
joblib

# Kaggle (YOLOv8)
ultralytics
torch (pre-installed on Kaggle)
```

---

## 📌 Status

- [x] HOG + SVM pipeline — complete (95% recall on VinBigData)
- [x] 5-Fold Cross Validation — F1 = 0.935
- [x] YOLOv8n — Cardiomegaly only (mAP50 = 0.879)
- [x] YOLOv8s — All 15 diseases (mAP50 = 0.891)(in the script you will find classe = [3] you can remove it to train the model for all diseases)
- [x] False alarm testing — 0/50
- [x] External image testing — 6/6 correct
- [x] Hybrid pipeline (YOLOv8 → HOG+SVM)
- [ ] Full comparative analysis write-up

---

## 👩‍💻 Author

**me xoxo**

---

## 📚 References

- Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR.
- Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* CVPR.
- Nguyen, T. et al. (2022). *VinBigData Chest X-ray Abnormalities Detection.* Kaggle Competition.
- Jocher, G. et al. (2023). *Ultralytics YOLOv8.* [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Cortes, C. & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.