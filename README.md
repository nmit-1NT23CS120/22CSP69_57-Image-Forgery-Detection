# 22CSP69_57-Image-Forgery-Detection
Image Forgery Detection in Social Media Images using Deep Learning Techniques
## Overview

This project focuses on detecting manipulated or forged images shared on social media platforms using deep learning techniques.

Images shared on platforms such as Instagram, Facebook, and Twitter undergo compression, resizing, and filtering. These transformations make traditional forgery detection methods less effective.

This project proposes a robust deep learning framework that can detect image manipulation even after social media compression.

## Key Features
- Deep learning based image forgery detection
- Robust to social media compression artifacts
- Hybrid spatial and frequency domain analysis
- Attention based forgery localization
- Explainable AI visualization using Grad-CAM
- Trust score prediction instead of simple binary classification
- Web interface prototype for real-time detection

## Technologies Used
- Python
- PyTorch / TensorFlow
- OpenCV
- NumPy
- Scikit-learn
- Streamlit / Flask

## Dataset
Datasets used for training and evaluation include:
- FaceForensics++
- DeepFake Detection Challenge (DFDC)
- CASIA Image Tampering Dataset

Additional social media compression simulations are applied to improve robustness.

## Model Architecture
The proposed system uses:
- ResNet / EfficientNet backbone
- Frequency domain feature extraction
- Attention mechanisms
- Feature fusion layer
- Binary classification with trust score output

## Project Structure
dataset/ → datasets used for training
preprocessing/ → image preprocessing scripts
models/ → deep learning architectures
training/ → training pipeline
evaluation/ → testing and metrics
web_app/ → deployment prototype
notebooks/ → experimentation notebooks
results/ → predictions and visual outputs

## How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run preprocessing
python preprocessing/image_preprocessing.py

3. Train model
python training/train_model.py

4. Run web application
python web_app/app.py

## Expected Output
- Forgery detection result
- Confidence / trust score
- Heatmap highlighting manipulated regions

## Future Work
- Video deepfake detection
- Browser extension integration
- Real-time social media monitoring
