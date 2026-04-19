# attention-enhanced-deepfake-detection
Deepfake detection using ResNet50V2 with SE and CBAM attention mechanisms
# Attention-Enhanced Deepfake Detection Using ResNet50V2 and Hybrid Attention Mechanisms

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
- Transformer (optional) → Global context

<img width="253" height="183" alt="image" src="https://github.com/user-attachments/assets/29b4e0ec-4104-466b-ba6f-11cb74a712db" />

### Final Layers
- Dense (1024 → 512 → 256)
- Dropout (0.4 / 0.3)
- Sigmoid output (binary classification)

---

## Performance

### Best Model: ResNet50V2 + SE + CBAM

| Metric | Value |
|------|------|
| Accuracy | 95% |
| AUC | 98.6% |
| Precision | 88% |
| Recall | 91% |
| F1-score | 89% |

---

## Datasets Used

- FaceForensics++ (FF++)
- Celeb-DF (v2)
- Deepfake Detection Dataset (DFD)
- Real and Fake Face Detection (RFFD)

### Final Dataset
- 30,000 Real Images
- 30,000 Fake Images

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

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- AUC (Area Under Curve)

---

## Explainability (Grad-CAM)
The model uses Grad-CAM to highlight:
- Mouth region
- Eye corners
- Facial boundaries

This improves interpretability by showing where the model focuses.

---

## Dataset Sources

- C. I. P. Lab, “Real and Fake Face Detection,” Kaggle, 2019. [Online]. Available: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection 
[Accessed: Jun. 11, 2025].
- H. Le, “FaceForensics,” Kaggle, 2022. [Online]. Available: https://www.kaggle.com/datasets/hungle3401/faceforensics 
[Accessed: Jun. 11, 2025].
- R. Suju, “Celeb-DF-v2,” Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2 
[Accessed: Jun. 11, 2025]. 
- T. Padhy, “Deepfake Dataset,” Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset 
[Accessed: Jun. 11, 2025].


