Stratified vs GroupKFold — Performance & Improvements Summary

---

📊 Accuracy Overview

Method| Mean Accuracy| Highest Accuracy| Lowest Accuracy
Stratified K-Fold| ~80% – 88%| ~85%| ~80%
Group K-Fold| 40.87%| 46.07%| 32.00%

---

📈 GroupKFold Fold-wise Results

Fold| Accuracy
Fold 1| 46.07%
Fold 2| 42.58%
Fold 3| 44.73%
Fold 4| 32.00%
Fold 5| 38.98%

---

⚙️ Implemented Changes (Our Model)

Compared to the baseline (teammate Stratified pipeline), the following modifications were applied:

🔹 Data Splitting

- Replaced StratifiedKFold with GroupKFold
- Ensured subject-wise separation across folds

---

🔹 Model Architecture Improvements

- Added Multimodal Fusion (EEG + Eye features)
- Introduced Residual Blocks for deeper feature learning
- Applied Cross-Modal Gating to weight EEG vs Eye inputs
- Integrated Attention Mechanism for feature importance
- Added Multi-Head Attention (2 heads) on fused features
- Implemented ensemble inside model (2 classifier heads averaged)

---

🔹 Regularization & Stability

- Feature Dropout (input-level regularization)
- Batch Normalization across layers
- Dropout (0.3) in deep fusion layers
- Label Smoothing (0.05) to reduce overconfidence

---

🔹 Data Augmentation

- MixUp augmentation (alpha = 0.4)
- Random Gaussian noise injection during training and testing

---

🔹 Optimization Improvements

- Optimizer: AdamW
- Learning rate: 1e-4
- Scheduler:
  - CosineAnnealingLR (earlier)
  - ReduceLROnPlateau (final version)
- Gradient clipping (max_norm = 1.0)
- Gradient accumulation (simulate larger batch)

---

🔹 Training Strategy

- Added 10% validation split inside each fold
- Implemented early stopping (patience = 10)
- Best model selected based on validation loss

---

🔹 Evaluation Enhancements

- Applied Test-Time Augmentation (TTA = 8 runs)
- Final prediction = average of multiple noisy forward passes

---

📌 Explanation

- Stratified K-Fold maintains class balance but ignores subject grouping
- GroupKFold enforces subject-level separation

Observed effect:

- Stratified → higher, tightly clustered accuracy (~85%)
- GroupKFold → lower but variable accuracy (32%–46%)

---

📊 Final Summary

- Highest observed accuracy: ~85% (Stratified K-Fold)
- Highest fold accuracy (GroupKFold): 46.07%
- Final mean accuracy (GroupKFold): 40.87%

---
