# Image Forgery Detection in Social Media Images Using Deep Learning Techniques

Nitte Meenakshi Institute of Technology | VTU

A robust and explainable hybrid deep learning framework for detecting manipulated images under real-world **social media conditions**, where platforms such as Facebook, Instagram, WhatsApp and Twitter automatically compress, resize and filter every uploaded image. The system combines three forensic cues (spatial features, frequency-domain features and EXIF metadata) and explains each decision with Grad-CAM heatmaps and a confidence-based trust score.

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
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Project Overview

Detectors that perform well on standard benchmarks often lose accuracy once an image has been compressed or altered by a platform. Platforms routinely compress, resize and filter uploads, and every one of these steps can erase or distort forensic traces. Many current systems are trained mainly on clean benchmark data, give limited insight into their decisions, and seldom exploit forensic cues from different domains together.

This project addresses these gaps through three contributions:

1. **Social Media Simulation Pipeline**: imitates the processing applied by Facebook, Instagram, WhatsApp and Twitter, and is used at both training and evaluation time.
2. **Hybrid Model**: draws on three forensic cues: spatial features from EfficientNet-B4, Laplacian-filtered frequency-domain features (256-dimensional), and EXIF metadata.
3. **Explainable Output**: Grad-CAM heatmaps show the regions that influenced each decision, and a confidence-based trust score (Low Risk / Suspicious / High Risk) adds an interpretable measure of reliability.

---

## Key Features

- Upload a single image or several images at once (including a ZIP archive) through a Flask web interface
- Classify each image as **Authentic** or **Forged**, with a confidence score and trust level
- Visualize influential regions with a **Grad-CAM heatmap overlay**
- Compare ResNet50, EfficientNet-B4 and the Hybrid model on the same image
- Re-test each image under Facebook, Instagram, WhatsApp and Twitter compression and see how each model's verdict changes
- Inspect EXIF metadata and the metadata-aware model's verdict (EXIF is used as an additional source of evidence when it is available)
- Batch mode for analyzing many images, with CSV export and a printable report

---

## System Architecture

```
Input Image
     │
     ▼
Preprocessing (resize 224×224, normalize)
     │
     ▼
Social Media Simulation Pipeline
 Facebook: resize >2048 px, enhancement, JPEG quality 71–95
 Instagram: resize >1080 px, colour enhancement, JPEG quality 70–85
 WhatsApp: resize >1600 px, JPEG quality 50–80, second pass with 40% probability
 Twitter: resize >1200 px, JPEG quality 80–90
 (during training each image is routed through one randomly chosen pipeline)
     │
     ├───────────────────────┬──────────────────────────┐
     ▼                       ▼                          ▼
EfficientNet-B4        Laplacian high-pass        EXIF metadata
spatial branch         frequency branch (256-d)   branch (when available)
     │                       │                          │
     └───────────────────────┴──────────┬───────────────┘
                                        ▼
                               Feature fusion
                                        │
                                        ▼
                    Classification head (Authentic / Forged)
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
          Confidence score + trust level           Grad-CAM heatmap
                      │
                      ▼
              Flask web interface
```

---

## Datasets

Four public datasets were used, totalling 15,386 images. Datasets are not included in this repository; download them from their official sources.

| Dataset | Images | Format | Forgery Type / Notes |
|---------|--------|--------|----------------------|
| CASIA v2.0 | 12,614 (7,491 authentic, 5,123 forged) | JPG / TIF | Splicing, copy-move |
| Columbia Uncompressed Color | 363 | TIF | Uncompressed, high-quality splicing images |
| Columbia Gray (ImSpliceDataset) | 1,845 | BMP | 128×128 grayscale splicing |
| NIST16 | 564 | JPG | Government-grade forensics challenge set (forged images) |
| **Total** | **15,386** | Mixed | All four combined for training and evaluation |

The data was split into 10,769 training, 2,308 validation and 2,309 test images, and all images are resized to 224×224 before being given to the models.

Download links: [CASIA v2 (Kaggle)](https://www.kaggle.com/datasets/sophatvathana/casia-dataset) · [Columbia IFP Group](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) · [NIST MFC](https://www.nist.gov/system/files/documents/2019/01/08/mfc2018-evalplan-2018-10-04-vs2.pdf)

Expected folder layout:

```
dataset/
├── CASIA/
│   ├── Au/                  ← authentic images
│   └── Tp/                  ← forged images
├── Columbia/
│   ├── color/
│   │   ├── 4cam_auth/
│   │   └── 4cam_splc/
│   └── gray/                ← Au-* and Sp-* folders
└── NIST16/
    └── images/              ← forged images
```

---

## Installation

**Prerequisites:** Python 3.8+ and a CUDA-compatible GPU (recommended).

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

---

## Usage

### 1. Preprocessing

```bash
# Prepare dataset (split into train/val/test)
python preprocessing/prepare_dataset.py

# Apply social media compression simulation
python preprocessing/simulate_compression.py
```

Processed images are saved to `dataset/processed/`.

### 2. Training

```bash
# Model 1: ResNet50 baseline (clean images only)
python training/train_resnet.py

# Model 2: EfficientNet-B4 (compression-aware)
python training/train_efficientnet.py

# Model 3: Hybrid model (spatial + frequency)
python training/train_hybrid.py

# Model 3 with the EXIF metadata branch
python training/train_hybrid_exif.py
```

Trained weights are saved to `saved_models/`. A GPU is strongly recommended.

### 3. Evaluation

```bash
# Evaluate all models on clean and the four social-media conditions
python evaluation/evaluate_all.py

# Generate comparison table
python evaluation/generate_comparison.py
```

Outputs are saved to `evaluation/results/`.

### 4. Running the Web App

```bash
cd web_app
python app.py
```

Open `http://localhost:5000`.

- **Single image analysis** (`/`): prediction, confidence, trust level, Grad-CAM heatmap, 3-model comparison, EXIF analysis and the four-platform compression test
- **Batch analysis** (`/batch`): upload many images and/or a ZIP (JPG, PNG, BMP, WEBP; up to 100 per batch). Every image gets a quick Hybrid-model verdict; click **Full Analyze** on any image for the complete analysis. Export results as CSV or print them as a PDF report.

Model weights (`.pth`) are not stored in this repository. Place `resnet_baseline.pth`, `efficientnet_model.pth`, `hybrid_model.pth` and (optionally) `hybrid_model_v2_exif.pth` in `saved_models/`. If the EXIF-aware weights are absent, the EXIF panel is shown as unavailable.

---

## Models

| Model | Training data | Parameters | Validation accuracy |
|-------|---------------|-----------:|--------------------:|
| ResNet50 (baseline) | Clean images only, ImageNet-pretrained | 24,558,146 | 80.7% |
| EfficientNet-B4 | Clean + social-media-compressed images, ImageNet-pretrained | 18,467,658 | 76.5% |
| Hybrid (ours) | Clean + compressed, all four platform augmentations | n/a | n/a |

The **Hybrid model** combines an EfficientNet-B4 spatial branch, a 256-dimensional frequency branch based on Laplacian high-pass filtering, and an EXIF branch containing the available metadata features. The branches are combined into the final representation used for classification.

**Trust scoring:** the prediction confidence is mapped to three categories: Low Risk (0–0.3), Suspicious (0.3–0.7) and High Risk (0.7–1.0).

**Grad-CAM:** in the heatmap, red areas indicate stronger influence on the prediction and blue areas lower influence.

---

## Results

Models were evaluated on clean test images and on the same images processed through the four social-media pipelines.

**Accuracy (%) on clean and compressed test sets**

| Model | Clean | Facebook | Instagram | WhatsApp | Twitter |
|-------|-------|----------|-----------|----------|---------|
| ResNet50 (Baseline) | 82.1 | 71.8 | 71.4 | 67.1 | 71.8 |
| EfficientNet-B4 | 77.0 | 75.0 | 74.5 | 74.9 | 74.4 |
| **Hybrid Model** | **92.2** | **89.3** | **90.5** | **90.7** | **91.4** |

**Classification performance**

| Model | Precision | Recall | F1 | Accuracy |
|-------|----------:|-------:|---:|---------:|
| ResNet50 | 80.1 | 79.4 | 79.7 | 82.1 |
| EfficientNet-B4 | 75.8 | 74.9 | 75.3 | 77.0 |
| **Hybrid Model** | **91.8** | **91.2** | **91.5** | **92.2** |

**Average accuracy drop under social-media compression**

| Model | Average | Facebook | Instagram | WhatsApp | Twitter |
|-------|--------:|---------:|----------:|---------:|--------:|
| ResNet50 (Baseline) | 11.6% | 10.3% | 10.7% | 15.0% | 10.3% |
| EfficientNet-B4 | 2.3% | 2.0% | 2.5% | 2.1% | 2.6% |
| **Hybrid Model** | **≈1.7%** | 2.9% | 1.7% | 1.5% | 0.8% |

ResNet50, trained only on clean images, shows the largest loss (up to 15.0% on WhatsApp-compressed images). Compression-aware training reduces the average drop to 2.3% for EfficientNet-B4, and the Hybrid model keeps the drop to about 1.7%.

---

## Project Structure

```
22CSP69_57-Image-Forgery-Detection/
│
├── dataset/                    ← datasets go here (not committed)
│   └── processed/              ← generated after preprocessing
│
├── preprocessing/
│   ├── prepare_dataset.py      ← dataset split and preparation
│   ├── simulate_compression.py ← social media pipeline simulation
│   └── dataset_loader.py
│
├── models/
│   ├── resnet_baseline.py      ← Model 1
│   ├── efficientnet_model.py   ← Model 2
│   └── hybrid_model.py         ← Model 3 (spatial + frequency)
│
├── training/
│   ├── train_resnet.py
│   ├── train_efficientnet.py
│   ├── train_hybrid.py
│   └── train_hybrid_exif.py    ← Hybrid with EXIF metadata branch
│
├── evaluation/
│   ├── evaluate_all.py         ← evaluate all models on all conditions
│   ├── generate_comparison.py  ← comparison tables
│   └── results/results.json
│
├── explainability/
│   └── gradcam.py              ← Grad-CAM heatmap generation
│
├── web_app/
│   ├── app.py                  ← Flask backend (single + batch routes)
│   ├── batch_utils.py          ← ZIP extraction, validation, batch store
│   ├── exif_utils.py           ← EXIF feature extraction
│   ├── models_v2.py            ← Hybrid model with EXIF branch
│   ├── templates/
│   │   ├── index.html
│   │   ├── results.html
│   │   └── batch.html
│   └── static/
│       ├── css/style.css
│       └── js/ (main.js, batch.js)
│
├── saved_models/               ← .pth weights and training histories
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch, TorchVision |
| Model Architectures | ResNet50, EfficientNet-B4 |
| Image Processing | OpenCV, Pillow |
| Explainability | Grad-CAM |
| Evaluation | Scikit-learn, Matplotlib, Seaborn |
| Web Framework | Flask |
| Training Platform | Google Colab (GPU) |

---

## References

1. H. Wu, J. Zhou, J. Tian, and J. Liu, "Robust Image Forgery Detection over Online Social Network Shared Images," in Proc. IEEE/CVF CVPR, pp. 13440–13449, 2022.
2. M. Agarwal and R. Kumar, "Deep Learning-Based Image Forgery Detection: Recent Advances and Challenges," IEEE Access, vol. 11, pp. 51412–51430, 2023.
3. R. Singh and K. Patel, "Attention-Based CNN for Image Forgery Detection," IEEE Trans. Inf. Forensics Security, vol. 18, pp. 2104–2116, 2023.
4. Z. Guo and Y. Liu, "Hybrid Spatial-Frequency Domain Deepfake Detection Network," IEEE Access, vol. 11, pp. 67214–67226, 2023.
5. X. Chen, J. Liu, and Y. Wang, "Vision Transformer Based Deepfake Detection Framework," IEEE Access, vol. 11, pp. 92310–92321, 2023.
6. H. Wang, J. Zhao, and P. Li, "Transformer-Based Deepfake Detection with Spatial-Frequency Features," IEEE Trans. Multimedia, vol. 25, pp. 2874–2886, 2023.
7. J. Frank and L. Verdoliva, "Robust Deepfake Detection Under Real-World Compression Conditions," IEEE Trans. Inf. Forensics Security, vol. 18, pp. 4201–4214, 2023.
8. S. Tariq, S. Lee, H. Kim, Y. Shin, and S. Woo, "Domain Adaptation for Deepfake Detection in Social Media Environments," IEEE Trans. Multimedia, vol. 25, pp. 1820–1832, 2023.
9. J. Patel, R. Shah, and N. Mehta, "Explainable Deepfake Detection Using Deep Learning Techniques," IEEE Access, vol. 11, pp. 100321–100333, 2023.
10. Z. Liu et al., "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows," in Proc. IEEE/CVF ICCV, pp. 10012–10022, 2021.
11. L. Zhang, Q. Liu, and H. Zhou, "Compression-Resilient Deepfake Detection Using Multi-Domain Features," IEEE Trans. Multimedia, vol. 26, pp. 1142–1155, 2024.
12. W. Shan et al., "RIFD-Net: A Robust Image Forgery Detection Network," IEEE Access, vol. 12, pp. 20326–20340, 2024.
13. F. Wang et al., "Deepfake Detection Based on the Adaptive Fusion of Spatial-Frequency Features," Int. J. Intell. Syst., vol. 2024, Art. no. 7578036, 2024.
14. D. Li et al., "Learnable Frequency Decomposition for Image Forgery Detection and Localization," in Proc. IJCAI, pp. 1359–1367, 2025.
15. E. Liang, K. Zhang, Z. Hua, and X. Jia, "Frequency-Driven Deep Learning Network for Image Splicing Forgery Detection," Knowledge-Based Systems, vol. 330, Art. no. 114365, 2025.
