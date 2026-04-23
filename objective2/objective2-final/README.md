# EEG Emotion Recognition - Objective 2

## Overview
This repository implements a multimodal emotion recognition pipeline using EEG and Eye-tracking data. The project compares window-level (Stratified) and subject-level (GroupKFold) validation strategies across several deep learning architectures.

## Dataset & Features
- **Modalities**: Multimodal EEG and Eye tracking.
- **Confirmed Features**: Both **Differential Entropy (DE)** and **Power Spectral Density (PSD)** features are included to capture spectral power and entropy-based dynamics.
- **Fusion**: Feature-level concatenation and Decision-level averaging.

## Model: Hybrid (DNN + Attention)
The core model for this project is the **Hybrid (DNN + Attention)** architecture. This model utilizes:
- **DNN blocks** for high-dimensional feature extraction.
- **Attention mechanisms** to dynamically prioritize informative features from both EEG and Eye modalities.

## Evaluation Track
1. **GroupKFold (LOSO)**: Rigorous subject-independent evaluation.
2. **Stratified KFold**: Performance-oriented window-level evaluation.

## How to Run
Execute the full pipeline with a single command:
```bash
python run.py
```
This will trigger both evaluation tracks sequentially and generate results in the `results/` folder.

---
*Maintained for Objective 2 final submission.*
