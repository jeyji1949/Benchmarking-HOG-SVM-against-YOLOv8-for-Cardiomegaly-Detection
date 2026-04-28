# 🫀 BioVision Cardio

> **Classical meets Clinical** — Benchmarking HOG+SVM against YOLOv8 for Cardiomegaly Detection from Chest X-rays

---

## 📋 Overview

**BioVision Cardio** is a comparative machine learning project developed as part of the *Master BIAM — Bioinformatics & AI for Precision Medicine* program. It investigates two fundamentally different approaches to detecting **Cardiomegaly** (enlarged heart) from annotated Chest X-ray images:

| Approach | Method | Philosophy |
|----------|--------|------------|
| 🔬 Classical ML | HOG + LinearSVM | Hand-crafted features → explicit decision boundary |
| 🤖 Deep Learning | YOLOv8 | End-to-end learned features → bounding box regression |

The goal is not just to build models, but to **understand and compare** how each method perceives medical images — and what that means for clinical deployment.

---

## 🩺 The Disease: Cardiomegaly

Cardiomegaly is clinically defined by a **Cardiothoracic Ratio (CTR) > 0.5**:

```
CTR = Heart width / Thoracic cage width > 0.5
```

An enlarged heart silhouette is **geometrically distinctive** on a chest X-ray — making it an ideal benchmark disease because:
- It has a clear, measurable binary definition (Normal vs. Enlarged)
- It produces strong gradient patterns at the cardiac border
- HOG can capture this structural signature directly

---

## 📁 Project Structure

```
BioVision-Cardio/
│
├── 📓 notebooks/
│   ├── 01_hog_svm_pipeline.ipynb       # Full HOG+SVM training pipeline
│   ├── 02_model_verification.ipynb     # Cross-validation & visualization
│   └── 03_yolov8_training.ipynb        # YOLOv8 setup & training (coming)
│
├── 📊 results/
│   ├── hog_svm_confusion.png           # Confusion matrix
│   └── model_verification.png         # Score distribution + HOG visualization
│
├── 📄 docs/
│   └── HOG_SVM_Documentation.docx     # Full technical documentation
│
└── README.md
```

---

## 🗂️ Dataset

**Source:** [`spritan1/yolo-annotated-chestxray-8-object-detection`](https://www.kaggle.com/datasets/spritan1/yolo-annotated-chestxray-8-object-detection) on Kaggle

| Split | Images | Labels |
|-------|--------|--------|
| Train | 631    | 631    |
| Val   | ~130   | ~130   |

**Classes (5 total):**

| ID | Disease |
|----|---------|
| 0  | Atelectasis |
| **1** | **Cardiomegaly** ← our target |
| 2  | Effusion |
| 3  | Infiltration |
| 4  | Mass/Nodule |

**Annotation format:** YOLO (`class x_center y_center width height`, normalized 0→1)

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
Crop heart region  →  Resize to 64×64  →  Grayscale
      │
      ▼
HOG Descriptor  →  1764-dimensional feature vector
   (9 orientations, 8×8 cells, 2×2 blocks)
      │
      ▼
Random Oversampling  →  Balance 124 vs 470 samples
      │
      ▼
LinearSVC (C=1.0)  →  Decision boundary in 1764-dim space
      │
      ▼
Prediction: Cardiomegaly (1) or Other (0)
```

### HOG Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `orientations` | 9 | 9 gradient direction bins (0°–160°) |
| `pixels_per_cell` | (8, 8) | 64 cells for a 64×64 image |
| `cells_per_block` | (2, 2) | 49 overlapping normalization blocks |
| **Feature vector size** | **1764** | 49 × 4 × 9 |

---

## 📊 Results — HOG + SVM

```
              precision    recall    f1-score   support

       Other       0.98      0.90      0.94        94
Cardiomegaly       0.91      0.98      0.94        94

    accuracy                           0.94       188
```

### Clinical Interpretation

- **Recall = 0.98** → The model detects 98% of real Cardiomegaly cases. Only 2% are missed.
- **Precision = 0.91** → 9% of detections are false alarms — acceptable in a screening context.
- In medicine, **high recall > high precision** for disease detection. Missing a case is more dangerous than a false alarm.

---

## 🚀 How to Run

### On Kaggle (recommended)

```python
# 1. Add the dataset to your notebook
# Dataset: spritan1/yolo-annotated-chestxray-8-object-detection

# 2. Install dependencies (already available on Kaggle)
# cv2, skimage, sklearn, numpy, matplotlib

# 3. Run notebook 01_hog_svm_pipeline.ipynb
```

### Dependencies

```
opencv-python
scikit-image
scikit-learn
numpy
matplotlib
```

---

## 🔬 Benchmarking Philosophy

This project is built around one central question:

> *Does a model that explicitly engineers gradient features (HOG) perform differently from a model that learns its own features (YOLO) — and why?*

| Dimension | HOG + SVM | YOLOv8 |
|-----------|-----------|--------|
| Feature learning | Manual (hand-crafted) | Automatic (convolutional) |
| What it "sees" | Gradient orientations at edges | Hierarchical patterns: edges → shapes → objects |
| Localization | None (crop-level only) | Full bounding box regression |
| Training data needed | Low (~500 crops) | Higher (~500+ annotated images) |
| Interpretability | High (HOG is visualizable) | Low (needs Grad-CAM) |
| Clinical analogy | Tracing the heart border with a ruler | Radiologist scanning the full X-ray contextually |
| Inference speed | ~1ms/image | ~10ms/image (GPU) |

---

## 📌 Status

- [x] HOG + SVM pipeline — complete (94% accuracy)
- [x] Model verification — cross-validation, score distribution, HOG visualization
- [x] Technical documentation
- [ ] YOLOv8 training — in progress
- [ ] Comparative analysis & thesis write-up

---

## 👩‍💻 Author

**Jihane El Khraibi**
Master BIAM — Bioinformatics & AI for Precision Medicine
Faculté des Sciences Dhar El Mahraz, Fès, Morocco

---

## 📚 References

- Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR.
- Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* CVPR.
- Jocher, G. et al. (2023). *Ultralytics YOLOv8.* [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Cortes, C. & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273–297.