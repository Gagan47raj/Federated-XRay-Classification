# 🩺 Federated Attention-Based Deep Learning for Pneumonia Detection from Chest X-Ray Images

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-green.svg)
![CBAM](https://img.shields.io/badge/Attention-CBAM-orange.svg)
![Status](https://img.shields.io/badge/Project-Completed-success.svg)

</div>

---

# 📖 Overview

This project presents a complete deep learning pipeline for **Pneumonia Detection from Chest X-Ray Images** using:

- 🧠 Convolutional Neural Networks (CNN)
- 🎯 Attention Mechanisms (CBAM)
- 🌐 Federated Learning (FedAvg)
- 📊 Comprehensive Evaluation Metrics
- 🔬 Explainable AI (Grad-CAM - Upcoming)

The objective is to develop a system capable of accurately detecting pneumonia from chest X-ray images while exploring privacy-preserving distributed learning through Federated Learning.

---

# 🚀 Key Features

### ✅ End-to-End Pipeline

- Dataset Analysis
- Data Cleaning
- Data Preprocessing
- Train/Validation/Test Split
- Baseline CNN Training
- Attention-Based CNN Training
- Federated Learning
- Performance Evaluation

### ✅ Attention Mechanism

Implemented:

- Channel Attention
- Spatial Attention
- CBAM (Convolutional Block Attention Module)

to improve feature extraction from medical images.

### ✅ Federated Learning

Implemented a complete Federated Learning framework:

- Client Training
- Model Synchronization
- Global Aggregation
- Federated Averaging (FedAvg)

without sharing raw medical images.

### ✅ Research-Oriented Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

---

# 🏗️ System Architecture

```text
Chest X-Ray Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Dataset Partitioning
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
Baseline CNN            CBAM CNN
        │                      │
        ▼                      ▼
Centralized Training   Attention Training
        │                      │
        └──────────────┬───────┘
                       │
                       ▼
             Federated Learning
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     Client 1       Client 2       Client N
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                 FedAvg Server
                       │
                       ▼
                Global Model
                       │
                       ▼
                 Evaluation
                       │
                       ▼
               Research Results
```

---

# 📂 Project Structure

```text
project/
│
├── configs/
├── data/
│   ├── raw/
│   ├── preprocessed/
│
├── notebooks/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── report/
│
├── scripts/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── federated/
│   └── explainability/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

### Chest X-Ray Dataset

Classes:

| Class | Description |
|---------|------------|
| NORMAL | Healthy Chest X-Ray |
| PNEUMONIA | Pneumonia Infected Chest X-Ray |

Dataset Structure:

```text
train/
    NORMAL/
    PNEUMONIA/

test/
    NORMAL/
    PNEUMONIA/

val/
    NORMAL/
    PNEUMONIA/
```

---

# ⚙️ Data Preprocessing Pipeline

Implemented:

### Image Transformation

- Resize Images
- Normalization
- Tensor Conversion

### Data Quality Checks

- Corrupted Image Detection
- Duplicate Detection
- Class Distribution Analysis

### Dataset Analysis

Generated:

- Resolution Distribution
- Class Distribution
- Train/Validation/Test Ratios
- Statistical Reports

---

# 🧠 Baseline CNN Architecture

The baseline model consists of:

### Feature Extractor

- Conv2D
- BatchNorm
- ReLU
- MaxPooling

### Classification Head

- Adaptive Average Pooling
- Dropout
- Fully Connected Layer

Architecture:

```text
Input
 ↓
Conv Block 1
 ↓
Conv Block 2
 ↓
Conv Block 3
 ↓
Global Average Pooling
 ↓
Dropout
 ↓
Linear Layer
 ↓
Output
```

---

# 🎯 CBAM Attention CNN

To improve performance, CBAM was integrated.

CBAM contains:

### Channel Attention

Learns:

> "What features are important?"

### Spatial Attention

Learns:

> "Where should the model focus?"

Architecture:

```text
Feature Maps
      │
      ▼
Channel Attention
      │
      ▼
Spatial Attention
      │
      ▼
Refined Feature Maps
```

Benefits:

- Better localization
- Improved feature selection
- Reduced irrelevant activations
- Better medical image understanding

---

# 🌐 Federated Learning Framework

Traditional deep learning requires centralized data collection.

Federated Learning allows:

```text
Hospital A
      │
Hospital B
      │
Hospital C
      │
Hospital D
      ▼
Model Updates Only
      ▼
Global Server
```

Raw patient data never leaves local devices.

---

## Federated Learning Workflow

```text
Global Model
      │
      ▼
Distribute to Clients
      │
      ▼
Local Training
      │
      ▼
Model Updates
      │
      ▼
FedAvg Aggregation
      │
      ▼
Updated Global Model
```

---

# 🔍 Evaluation Metrics

The following metrics were used:

### Accuracy

Measures overall correctness.

### Precision

Measures reliability of positive predictions.

### Recall

Measures ability to detect pneumonia cases.

### F1 Score

Balance between precision and recall.

### ROC-AUC

Measures classifier separability.

---

# 📈 Experimental Results

## Baseline CNN

| Metric | Value |
|----------|----------|
| Accuracy | 83.81% |
| Precision | 80.81% |
| Recall | 97.18% |
| F1 Score | 88.24% |
| ROC-AUC | 93.24% |

---

## CBAM CNN

| Metric | Value |
|----------|----------|
| Accuracy | 83.81% |
| Precision | 80.68% |
| Recall | 97.44% |
| F1 Score | 88.27% |
| ROC-AUC | 93.59% |

---

## Federated CBAM

| Metric | Value |
|----------|----------|
| Accuracy | 79.65% |
| Precision | 76.46% |
| Recall | 97.44% |
| F1 Score | 85.68% |
| ROC-AUC | 91.61% |

---

# 📊 Model Comparison

| Model | Accuracy | F1 Score | ROC-AUC |
|----------|----------|----------|----------|
| Baseline CNN | 83.81% | 88.24% | 93.24% |
| CBAM CNN | ⭐ 83.81% | ⭐ 88.27% | ⭐ 93.59% |
| Federated CBAM | 79.65% | 85.68% | 91.61% |

---

# 🎯 Key Findings

### CBAM Improved Performance

CBAM achieved:

- Higher Recall
- Higher F1 Score
- Higher ROC-AUC

compared to the baseline CNN.

### Federated Learning Preserved Privacy

Despite distributed training:

- High Recall
- Strong ROC-AUC
- Competitive Performance

were achieved without sharing raw medical data.

---

# 🔬 Research Contributions

This project demonstrates:

✅ Medical Image Classification

✅ Attention-Based Deep Learning

✅ Federated Learning Framework

✅ Privacy-Preserving AI

✅ Research-Oriented Evaluation Pipeline

---

# 🛠️ Installation

Clone repository:

```bash
git clone https://github.com/Gagan47raj/Federated-XRay-Classification.git

cd pneumonia-federated-cbam
```

Create environment:

```bash
python -m venv venv
```

Activate:

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Train Baseline CNN

```bash
python scripts/train_baseline.py
```

Train CBAM CNN

```bash
python scripts/train_cbam.py
```

Run Federated Training

```bash
python scripts/train_federated.py
```

Evaluate Model

```bash
python scripts/evaluate_baseline.py
```

---

# 🔮 Future Improvements

This project can be extended through:

### Federated Learning

- FedProx
- FedNova
- Scaffold
- Adaptive Federated Optimization

### Deep Learning

- EfficientNet
- DenseNet121
- ResNet50
- Vision Transformers (ViT)

### Explainable AI

- Grad-CAM
- Grad-CAM++
- Score-CAM
- SHAP

### Medical AI

- Multi-Class Pneumonia Classification
- COVID-19 Detection
- Tuberculosis Detection
- Multi-Disease Screening

### Deployment

- FastAPI Backend
- Streamlit Dashboard
- Docker Containerization
- Cloud Deployment

---

# 👨‍💻 Author

**Gagan**

M.Tech Artificial Intelligence & Machine Learning

Research Area:

- Deep Learning
- Medical Imaging
- Federated Learning
- Explainable AI

---

# ⭐ If you found this project useful

Please consider giving the repository a star.

It helps support future development and research.
