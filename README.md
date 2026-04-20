# Attention-Enhanced-Deepfake-Detection
Deepfake detection using ResNet50V2 with SE and CBAM attention mechanisms
# Title: Attention-Enhanced Deepfake Detection Using ResNet50V2 and Hybrid Attention Mechanisms

## Overview
This repository presents a hybrid deepfake detection framework that combines **Convolutional Neural Networks (CNNs)** with **attention mechanisms** to improve detection accuracy and interpretability.

The proposed model integrates:
- ResNet50V2 (pre-trained backbone)
- Squeeze-and-Excitation (SE)
- Convolutional Block Attention Module (CBAM)
- Optional Transformer layer

The goal is to detect subtle facial manipulations in images and improve robustness across multiple datasets.

---

## Key Contributions
- Hybrid architecture combining CNN + multi-level attention
- Improved detection of subtle deepfake artifacts
- Strong generalization across multiple datasets
- Attention visualization using Grad-CAM

---

## Model Architecture

### Backbone
- ResNet50V2 (ImageNet pre-trained)

### Attention Modules
- SE Block → Channel attention
- CBAM → Channel + Spatial attention
- Transformer 
<p align="center">
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/29b4e0ec-4104-466b-ba6f-11cb74a712db" />
  <br>
  <em>Fig.1: Pseudocode for our model function</em>
</p>

### Final Layer
  - Dense (1024 → 512 → 256)
  - Dropout (0.4 / 0.3)
  - Sigmoid output (binary classification)

---

## Environment Setup

- OS: macOS 15.7.4 | 3.7 GHz 6-Core Intel Core i5 | Radeon Pro 580X 8 GB | 8 GB 2667 MHz DDR4
- Python Version: Python 3.8.2
- GPU: Not used

### Required Libraries

| Library        | Version  |
|----------------|----------|
| TensorFlow     | 2.13.0   |
| Keras          | 2.13.1   |
| opencv-python  | 4.9.0 84 |
| NumPy          | 1.24.3   |
| mediapipe      | 0.10.11  |
| mtcnn          | 0.1.1    |

All dependencies are listed in requirements.txt

## Performance

### Best Model: ResNet50V2 + SE + CBAM

| Metric | Value |Best Epoch
|------|------|------
| Accuracy | 95% | 27
| AUC | 98.6% | 27


### Best Result: ResNet50V2 + SE + CBAM With Benchmark Dataset FF++ balanced performance

| Metric | Value 
|------|------|
| Accuracy | 92% | 
| AUC | 98% | 
| Precision | 92% | 
| Recall | 93% |
| F1-score | 93% |
---
## Data Availability

The datasets used in this study are publicly available on Kaggle and other benchmark sources, including:
  
- FaceForensics++ (FF++)
- Celeb-DF (v2)
- Deepfake Detection Dataset (DFD)
- Real and Fake Face Detection (RFFD)
  
These datasets can be accessed via Kaggle:  
https://www.kaggle.com/  
Note: The dataset was preprocessed and curated (filtering, normalization, and balancing) before training.

## Dataset Sources

- C. I. P. Lab, “Real and Fake Face Detection,” Kaggle, 2019. [Online]. Available: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection 
[Accessed: Jun. 11, 2025].
- H. Le, “FaceForensics,” Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/hungle3401/faceforensics 
[Accessed: Jun. 11, 2025].
- R. Suju, “Celeb-DF-v2,” Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2 
[Accessed: Jun. 11, 2025]. 
- T. Padhy, “Deepfake Dataset,” Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset 
[Accessed: Jun. 11, 2025].

 ## Dataset Overview

The dataset used in this study is collected from multiple benchmark sources and undergoes several preprocessing stages including normalization and cleaning.

### Data Collection Summary

<img width="850" height="200" alt="Screenshot 2026-04-19 at 2 40 11 PM" src="https://github.com/user-attachments/assets/24391bd6-e251-4be2-b07c-06b5bf4bc83e" />


### Final Dataset

- **30,000 Real Images**
- **30,000 Fake Images**
- Balanced dataset for fair training and evaluation



---

## Training Configuration

| Parameter | Value |
|----------|------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Batch Size | 32 |
| Epochs | 30 |
| Image Size | 224 × 224 |

---

## Data Preprocessing
- Face extraction from videos
- Image normalization
- Quality filtering (brightness, sharpness)
- Resizing to 224×224
- Data augmentation:
  - Rotation
  - Zoom
  - Shift
  - Horizontal flip
 
<p align="center">
<img width="550" height="160" alt="image" src="https://github.com/user-attachments/assets/a6104d4c-fbe6-45c8-8027-f636a3c9015a" />
  <br>
  <em>Fig.3: Structure of preprocessing data</em>
  <br>
  <img width="749" height="383" alt="Screenshot 2026-04-20 at 10 24 14 AM" src="https://github.com/user-attachments/assets/2c6d3f09-851c-401c-8fcb-47fc522b0144" />

</p>

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- AUC (Area Under Curve)
<p align="center">
<img width="724" height="226" alt="Screenshot 2026-04-20 at 10 26 16 AM" src="https://github.com/user-attachments/assets/79edb8fb-cba0-426c-baf4-c7ea946919e3" />
</p>
---

## Explainability (Grad-CAM)
The model uses Grad-CAM to highlight:
- Mouth region
- Eye corners
- Facial boundaries
<p align="center">
<img width="688" height="314" alt="Screenshot 2026-04-19 at 2 42 34 PM" src="https://github.com/user-attachments/assets/83ce8ef9-8e8e-49a6-bb72-95e51661e378" />
</p>
This improves interpretability by showing where the model focuses.

---
The proposed system follows a multi-stage pipeline. 
Video → Face Extraction → Normalization → Quality Check → Model → Output
---
### Experiment Result
<p align="center">
<img width="480" height="212" alt="image" src="https://github.com/user-attachments/assets/1914462f-0f6e-4035-a677-7e113d8b415a" />

<img width="456" height="295" alt="image" src="https://github.com/user-attachments/assets/3cd16105-8609-4cc5-a5c7-6d6f8568781c" />


<img width="470" height="258" alt="image" src="https://github.com/user-attachments/assets/0e598e2a-10b5-4dfa-ba9f-797d2c6a9a8a" />

<img width="470" height="311" alt="image" src="https://github.com/user-attachments/assets/e4d8f158-f415-4ffe-bc64-75c5a9621adc" />

<img width="717" height="1020" alt="Screenshot 2026-04-19 at 2 46 59 PM" src="https://github.com/user-attachments/assets/31c34d82-88f0-4409-88b8-0cbd14ba8bda" />
</p>


### Citation

If you use this work, please cite:

**Attention-Enhanced Deepfake Detection Using ResNet50V2 and Hybrid Attention Mechanisms** 
Naif A. Almalki, Mahmoud Ragab, Faris Kateb;  
Manuscript under review at *The Visual Computer*, Springer, 2026.
---
### Contact

For questions, collaborations, or further information, please contact:

**Naif A. Almalki**  
📧 nattiahalmalki@stu.kau.edu.sa
