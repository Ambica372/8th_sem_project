Stratified vs GroupKFold — Performance & Model Improvements

---

📊 Accuracy Comparison

Method| Mean Accuracy| Highest Accuracy| Lowest Accuracy
Stratified K-Fold| ~80% – 88%| ~85%| ~80%
Group K-Fold| 40.87%| 46.07%| 32.00%

---

📈 GroupKFold Fold-wise Accuracy

Fold| Accuracy
Fold 1| 46.07%
Fold 2| 42.58%
Fold 3| 44.73%
Fold 4| 32.00%
Fold 5| 38.98%

---

⚙️ Implemented Changes

The following improvements were applied over the baseline Stratified pipeline:

🔹 Data Splitting

- Switched from StratifiedKFold to GroupKFold
- Ensured subject-wise separation across folds

---

🔹 Model Architecture

- Multimodal fusion of EEG + Eye features
- Residual Blocks for deeper feature learning
- Cross-Modal Gating for adaptive feature weighting
- Attention Mechanism for feature importance
- Multi-Head Attention (2 heads) on fused features
- Dual classifier heads (ensemble averaging)

---

🔹 Regularization

- Feature Dropout
- Batch Normalization
- Dropout (0.3)
- Label Smoothing (0.05)

---

🔹 Data Augmentation

- MixUp augmentation (alpha = 0.4)
- Gaussian noise injection during training and testing

---

🔹 Optimization

- Optimizer: AdamW
- Learning rate: 1e-4
- Scheduler: ReduceLROnPlateau
- Gradient clipping
- Gradient accumulation

---

🔹 Training Strategy

- 10% validation split from training data
- Early stopping (patience = 10)
- Best model selected using validation loss

---

🔹 Evaluation

- Test-Time Augmentation (TTA = 8 runs)
- Final prediction = average of multiple runs

---

📌 Explanation

- Stratified K-Fold
  
  - Splits based on class distribution
  - Produces higher and stable accuracy (~85%)

- Group K-Fold
  
  - Splits based on subject groups
  - Produces lower but more variable accuracy (32%–46%)

---

📊 Final Summary

- Highest observed accuracy: ~85% (Stratified K-Fold)
- Highest GroupKFold accuracy: 46.07%
- Final GroupKFold mean accuracy: 40.87%

---
