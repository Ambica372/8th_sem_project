# Objective 2 — Stratified vs GroupKFold Evaluation

## 📊 Final Results (Key Takeaway First)

| Evaluation Method | Accuracy Range | Reality Level |
|------------------|--------------|-------------|
| Stratified K-Fold | **80% – 88%** | ❌ Inflated (data leakage present) |
| Group K-Fold      | **40% – 45%** | ✅ Realistic (true generalization) |

---

## 🔍 Overview

This work focuses on improving the evaluation methodology for multimodal emotion classification using EEG and Eye-tracking data.

The original pipeline produced unreliable results due to hidden data leakage.  
We implemented improved training strategies and compared two validation methods:

- Stratified K-Fold (sample-level split)
- Group K-Fold (subject-level split)

---

## ⚙️ Methodological Changes Implemented

### 1. Model Architecture Improvements
- Increased network depth
- Added **Batch Normalization**
- Added **Dropout (0.3)** for regularization
- Improved Decision Fusion architecture (balanced EEG + Eye branches)

---

### 2. Optimization Enhancements
- Switched optimizer → **Adam → AdamW**
- Added **Weight Decay (1e-4)** to prevent overfitting
- Learning rate tuning → `5e-5`

---

### 3. Training Stabilization
- **Gradient Clipping** (max_norm=1.0)
- **Class Weights** for imbalance handling
- **ReduceLROnPlateau Scheduler**
  - Reduces LR when performance plateaus
- **Early Stopping**
  - Stops training when no improvement

---

### 4. Data Handling Improvements
- Removed **NaN / Inf rows consistently across all modalities**
- Ensured strict alignment:
  - EEG ↔ Eye ↔ Labels
- Per-fold normalization:
  - `StandardScaler` fit ONLY on training data

---

## ⚠️ Critical Observation: Stratified K-Fold

Stratified K-Fold splits data randomly at the sample level.

### Problem:
- Same subject appears in both **train and test**
- Adjacent time windows are highly similar

### Effect:
- Model learns **subject-specific patterns**
- Not true emotion generalization

### Result:
- High accuracy (**80–88%**)
- But **NOT reliable**

---

## ✅ Correct Evaluation: Group K-Fold

GroupKFold splits data based on **subjects**.

### Key Idea:
- Each subject appears in ONLY one fold
- No subject overlap between train and test

### Effect:
- Model must generalize to unseen individuals
- Prevents memorization

### Result:
- Lower accuracy (**40–45%**)
- But **scientifically valid**

---

## 📉 Why Accuracy Drops

| Reason | Explanation |
|------|------------|
| Subject variability | Brain signals differ per person |
| No memorization | Model can't rely on identity patterns |
| Harder task | True emotion recognition is difficult |

---

## 📈 Interpretation of Results

- Random baseline (4 classes): **25%**
- GroupKFold (~40–45%) → significantly above chance ✅
- Indicates model learns **real emotional patterns**

---

## ⚖️ Final Comparison

| Aspect | Stratified K-Fold | Group K-Fold |
|------|------------------|-------------|
| Split Type | Sample-level | Subject-level |
| Leakage | High | None |
| Accuracy | High (80–88%) | Moderate (40–45%) |
| Reliability | Low | High |
| Real-world validity | ❌ | ✅ |

---

## 🧠 Final Conclusion

- Stratified K-Fold provides **upper-bound performance** but is misleading
- GroupKFold reflects **true model capability**
- All optimization improvements were tested, but realistic performance saturates around **40–45%**

---
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
