# EEG Emotion Recognition - Objective 2

## Project Overview
This repository implements a multimodal emotion recognition pipeline using EEG and Eye-tracking data. We compare various deep learning architectures and investigate the critical role of validation strategy in physiological signal processing.

## Dataset & Features
- **Modalities**: EEG (62 channels) + Eye-tracking (31 features)
- **Features**: Both **Differential Entropy (DE)** and **Power Spectral Density (PSD)** features are utilized.

## Models Implemented
- **MLP**: Baseline Multi-layer Perceptron.
- **DNN**: Deep Neural Network.
- **Attention**: Dynamic feature weighting.
- **Hybrid (Best Architecture)**: Combines **DNN** layers with **Attention fusion**.
- **Decision Fusion**: Late fusion of modality-specific predictions.

## Evaluation Methods
We utilize two distinct cross-validation strategies:
1. **Stratified K-Fold (With Leakage)**: A window-level split that yields high accuracy (~88%) but suffers from identity leakage.
2. **GroupKFold LOSO (No Leakage)**: A subject-level Leave-One-Subject-Out split that provides a rigorous, scientific measure of generalization (~40–45%).

## 🔑 Key Finding: The Generalization Gap
The observed accuracy drop from **88% (Stratified)** to **40% (GroupKFold)** is the most significant finding of this project. 
- The **88% accuracy is misleading** as it represents the model's ability to recognize subjects it has already seen (identity recognition).
- The **40-45% accuracy is correct** for real-world applications, as it proves the high degree of subject variability in physiological signals and the necessity of subject-independent evaluation.

**Note:** Stratified K-Fold overestimates performance due to subject overlap. GroupKFold enforces subject independence and is the correct evaluation for real-world robustness.

## How to Run
Execute the full pipeline:
```bash
python run.py
```

---
*Organized for Objective 2 Final Submission.*
