# Image Forgery Detection in Social Media Images Using Deep Learning

> **Major Capstone Project — Phase 1 (Semester 6)**
> Nitte Meenakshi Institute of Technology | VTU | Academic Year 2025–26

A robust, explainable hybrid deep learning system that detects forged images specifically under real-world **social media conditions** — where images are automatically compressed, resized, and filtered by platforms like Instagram, Facebook, WhatsApp, and Twitter.

---

## Team

| Name | USN |
|------|-----|
| Lekha R | 1NT23CS120 |
| Athul Krishna | 1NT23CS035 |
| Darshan S S | 1NT23CS052 |
| Kanagala Sri Harshitha | 1NT23CS098 |

**Guide:** Ms. Sandhya B R, Assistant Professor - III, Dept. of CSE, NMIT

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Running the Web App](#4-running-the-web-app)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Project Overview

Most image forgery detection systems are trained on clean benchmark datasets and fail when applied to real-world social media images — because platforms automatically compress, resize, and filter every uploaded image, destroying conventional forensic traces.

This project addresses that gap through three contributions:

1. **Social Media Simulation Pipeline** — replicates Facebook, Instagram, WhatsApp, and Twitter compression behaviour during training so the model learns to detect forgeries even after platform processing
2. **Hybrid CNN Architecture** — fuses EfficientNet-B4 spatial features (1792-dim) with Laplacian frequency-domain features (256-dim) into a 2048-dim representation that captures manipulation artifacts invisible to spatial-only models
3. **Explainable AI Output** — Grad-CAM heatmaps highlight the exact manipulated regions, and a three-tier trust scoring system (Low / Suspicious / High Risk) makes results actionable for non-technical users

---

## Key Features

- Upload any JPEG or PNG image via a Flask web interface
- Classify images as **Authentic** or **Manipulated**
- Generate a **confidence score** and **trust level** (Low Risk / Suspicious / High Risk)
- Visualize manipulated regions via **Grad-CAM heatmap overlay**
- Compare performance of all 3 models side-by-side
- Robust against Facebook, Instagram, WhatsApp, and Twitter compression pipelines

---

## System Architecture

```
Input Image
     │
     ▼
Image Preprocessing (resize 224×224, normalize)
     │
     ▼
Social Media Simulation Pipeline
(Facebook QF:71-95 | Instagram QF:70-85 | WhatsApp QF:50-80 | Twitter QF:80-90)
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼
EfficientNet-B4 Spatial Branch     Laplacian Frequency Branch
   (1792-dim features)                  (256-dim features)
     │                                      │
     └──────────────┬───────────────────────┘
                    ▼
           Fused Feature Vector (2048-dim)
                    │
                    ▼
         Classification Head
        (Authentic / Manipulated)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Confidence Score      Grad-CAM Heatmap
   + Trust Level         (suspicious regions)
          │
          ▼
     Flask Web Interface
```

---

## Dataset Setup

We use three publicly available benchmark datasets. **Do not commit dataset files to this repo** — download them from the official sources below.

| Dataset | Images | Forgery Types | Download |
|---------|--------|--------------|----------|
| CASIA v2.0 | 12,614 | Splicing, Copy-move | [Kaggle — CASIA v2](https://www.kaggle.com/datasets/sophatvathana/casia-dataset) |
| Columbia Uncompressed | 363 | Splicing | [Columbia IFP Group](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) |
| NIST16 | 564 | Multiple | [NIST MFC Dataset](https://www.nist.gov/system/files/documents/2019/01/08/mfc2018-evalplan-2018-10-04-vs2.pdf) |

After downloading, place them as:

```
dataset/
├── CASIA/
│   ├── Au/          ← authentic images
│   └── Tp/          ← tampered images
├── Columbia/
│   ├── authentic/
│   └── tampered/
└── NIST16/
    └── ...
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or Google Colab

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/nmit-1NT23CS120/22CSP69_57-Image-Forgery-Detection.git
cd 22CSP69_57-Image-Forgery-Detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt includes:
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
flask>=3.0.0
grad-cam>=1.4.8
tqdm>=4.65.0
pandas>=2.0.0
```

---

## Usage

### 1. Preprocessing

Prepare and split the dataset, and run the social media simulation pipeline:

```bash
# Step 1: Prepare dataset (split into train/val/test)
python preprocessing/prepare_dataset.py

# Step 2: Apply social media compression simulation
python preprocessing/simulate_compression.py
```

Processed images will be saved to `dataset/processed/`.

---

### 2. Training

Train all three models individually:

```bash
# Model 1: ResNet50 baseline (clean images only)
python training/train_resnet.py

# Model 2: EfficientNet-B4 (compression-aware)
python training/train_efficientnet.py

# Model 3: Hybrid Model — our main contribution
python training/train_hybrid.py
```

Trained model weights are saved to `saved_models/`.

> **Google Colab**: Upload the repo and mount your Drive for dataset access. GPU runtime is strongly recommended — training ResNet50 takes ~45 mins, Hybrid ~90 mins on Colab T4.

---

### 3. Evaluation

Run evaluation on all 3 models across all test conditions:

```bash
# Evaluate all models (clean + 4 platform conditions)
python evaluation/evaluate_all.py

# Generate comparison table
python evaluation/generate_comparison.py
```

Output comparison tables and plots are saved to `results/`.

---

### 4. Running the Web App

```bash
cd web_app
python app.py
```

Open your browser and go to `http://localhost:5000`

**What you can do in the web app:**
1. Upload a JPEG or PNG image (max 10 MB)
2. View the prediction: Authentic / Manipulated
3. See the confidence score and trust level
4. View the Grad-CAM heatmap highlighting suspicious regions
5. Compare all 3 models on the same image

---

## Models

### Model 1 — ResNet50 (Baseline)
- Pretrained on ImageNet, fine-tuned for forgery detection
- Trained on clean images only
- Parameters: ~24.5M
- Val Accuracy: 80.7%
- Purpose: Demonstrates failure of standard models on social media images

### Model 2 — EfficientNet-B4 (Compression-Aware)
- Pretrained on ImageNet, fine-tuned with social media augmentation
- Trained on clean + compressed images
- Parameters: ~18.4M
- Val Accuracy: 76.5%
- Purpose: Shows improved robustness over baseline

### Model 3 — Hybrid Model (Our Contribution)
- EfficientNet-B4 spatial branch (1792-dim) + Laplacian frequency branch (256-dim)
- Fused 2048-dim feature vector
- Trained with all 4 platform simulations
- Parameters: ~19.4M
- Purpose: Best real-world performance; primary contribution of this project

---

## Results

*(To be updated with final evaluation numbers after full training run)*

| Model | Clean | Facebook | Instagram | WhatsApp | Twitter |
|-------|-------|----------|-----------|----------|---------|
| ResNet50 (Baseline) | 80.7% | — | — | — | — |
| EfficientNet-B4 | 76.5% | — | — | — | — |
| Hybrid Model (Ours) | — | — | — | — | — |

Grad-CAM heatmap examples are available in `results/`.

---

## Project Structure

```
forgery_detection/
│
├── dataset/                    ← datasets go here (not committed)
│   ├── CASIA/
│   ├── Columbia/
│   ├── NIST16/
│   └── processed/              ← generated after preprocessing
│
├── preprocessing/
│   ├── simulate_compression.py ← social media pipeline simulation
│   └── prepare_dataset.py      ← dataset split and preparation
│
├── models/
│   ├── resnet_baseline.py      ← Model 1 architecture
│   ├── efficientnet_model.py   ← Model 2 architecture
│   └── hybrid_model.py         ← Model 3 architecture (our contribution)
│
├── training/
│   ├── train_resnet.py         ← train Model 1
│   ├── train_efficientnet.py   ← train Model 2
│   └── train_hybrid.py         ← train Model 3
│
├── evaluation/
│   ├── evaluate_all.py         ← test all 3 models
│   └── generate_comparison.py  ← generate comparison table
│
├── explainability/
│   └── gradcam.py              ← Grad-CAM heatmap generation
│
├── web_app/
│   ├── app.py                  ← Flask backend
│   ├── templates/
│   │   ├── index.html          ← upload page
│   │   └── results.html        ← results display page
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── saved_models/               ← .pth files go here (not committed)
│   ├── .gitkeep
│
├── results/                    ← evaluation outputs, heatmaps
│   ├── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch + TorchVision |
| Model Architectures | ResNet50, EfficientNet-B4 |
| Image Processing | OpenCV, Pillow |
| Explainable AI | Grad-CAM (pytorch-grad-cam) |
| Evaluation | Scikit-learn |
| Web Framework | Flask |
| Visualization | Matplotlib, Seaborn |
| Training Platform | Google Colab (T4 GPU) |

---

## References

1. Y. Wang et al., "Robust Image Forgery Detection Over Online Social Network Shared Images," CVPR 2022
2. Z. Guo and Y. Liu, "Hybrid Spatial-Frequency Domain Deepfake Detection Network," IEEE Access, 2023
3. X. Zhang et al., "Frequency-Aware Deepfake Detection for Improving Generalization," IEEE Access, 2022
4. J. Patel et al., "Explainable Deepfake Detection Using Deep Learning Techniques," IEEE Access, 2023

Full reference list available in the project synopsis.

---

## CV Entry

> **Image Forgery Detection Using Deep Learning** | PyTorch, EfficientNet-B4, Flask
> Developed a hybrid CNN and frequency-domain model for detecting manipulated social media images with Grad-CAM explainability and platform-aware compression training.
> GitHub: https://github.com/nmit-1NT23CS120/22CSP69_57-Image-Forgery-Detection
> NMIT | Semester 6 | 2025–26

---

*Submitted for Major Project Work – Phase 1 | Academic Year 2025–26 | VTU*
