# 📊 Stratified vs GroupKFold — Emotion Classification

## 🔥 Final Accuracy Summary

| Method             | Model Version        | Accuracy |
|-------------------|---------------------|----------|
| Stratified K-Fold | Enhanced Model      | *88% (peak)* |
| Stratified K-Fold | Baseline Model      | ~45% |
| GroupKFold        | Enhanced Model      | *40.87% (mean)* |

---

## ⚙️ Implemented Improvements

### 🧠 Model Architecture
- Deep Fusion Block (Linear + BatchNorm + GELU + Dropout)
- Residual Blocks on fused features
- Multi-Head Attention (2 heads)
- Cross-Modal Gating (EEG + Eye weighting)
- Dual Classifier Heads (internal ensemble)

---

### 📈 Training Enhancements
- Mixup Augmentation (α = 0.4)
- Noise Injection
- Label Smoothing (0.05)
- Gradient Accumulation
- Gradient Clipping
- Test-Time Augmentation (TTA = 8)

---

### ⚡ Optimization
- AdamW optimizer
- Weight Decay regularization
- ReduceLROnPlateau scheduler
- Early Stopping (patience = 10)

---

## 📊 Results Analysis

### Stratified K-Fold
- Accuracy improved from *~45% → 88%*
- Strong performance across folds
- Reflects model learning under random data distribution

---

### GroupKFold (Subject-wise)
- Mean Accuracy: *40.87%*
- Performance drops significantly
- Higher difficulty due to unseen subjects

---

## 📌 Key Difference

| Method        | Behavior |
|--------------|---------|
| Stratified   | Random split (same subject may appear in train & test) |
| GroupKFold   | Subject-wise split (no subject overlap) |

---

## 📊 Interpretation

- Stratified results show *upper-bound performance*
- GroupKFold results show *real-world generalization*

---

## 📌 Final Conclusion

- Model improvements significantly boosted performance under Stratified setup (*up to 88%*)  
- Under strict subject-wise evaluation, performance is *~40.87%*, representing realistic behavior  

---

## ⚠️ Note

- Both evaluation strategies are reported  
- Stratified → best-case scenario  
- GroupKFold → real deployment scenario  

---
