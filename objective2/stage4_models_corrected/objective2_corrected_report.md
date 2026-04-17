# Objective 2 — Corrected Pipeline Report

**Date:** 2026-04-18 02:07  
**Validation Subject:** 15 (held-out from all folds)  
**Cross-Validation:** 5-fold subject-level CV  
**PCA Variance Retained:** 95%  

---

## 1. Issues Identified in Original Pipeline

| Issue | Description | Severity |
|-------|-------------|----------|
| Random row split | `train_test_split(X, y)` mixed windows from same subject across train/test | CRITICAL |
| Scaler leakage | `StandardScaler.fit_transform(X_full)` called on full dataset before split | CRITICAL |
| PCA leakage | Pre-generated `X_eeg_pca.npy` fitted on full dataset (no per-fold PCA) | CRITICAL |
| Test used for model selection | Best epoch chosen by test accuracy, not validation accuracy | HIGH |
| Unscaled Decision Fusion inputs | EEG/Eye inputs to Decision Fusion were not normalized | MEDIUM |
| No cross-validation | Single fixed split — no variance estimation | MEDIUM |

---

## 2. Fixes Applied

| Fix | Implementation |
|-----|---------------|
| Subject-level splitting | Used `stage2_output/fold_k_*` CSVs — subjects fully isolated per fold |
| PCA on train only | `PCA(0.95).fit_transform(X_eeg_train)` per fold; `.transform()` for test/val |
| Scaler on train only | `StandardScaler().fit_transform(X_train)` per fold; `.transform()` for test/val |
| Separate validation set | Subject 15 held out for early stopping and model selection |
| Model selection on val | `if val_acc > best_val_acc: save_model()` — test set never seen during training |
| Scaled Decision Fusion | Separate StandardScalers for EEG-PCA and Eye streams |
| 5-fold cross-validation | Results reported as mean +/- std across 5 folds |

---

## 3. Updated Results

### 3a. Cross-Validated Test Performance (Corrected — Subject-Level)

| Model | Accuracy (mean +/- std) | Precision | Recall | F1-Score |
|-------|-----------------------|-----------|--------|----------|
| MLP | 46.94% +/- 2.97% | 48.20% | 47.00% | 45.40% +/- 3.67% |
| DNN | 44.01% +/- 3.97% | 45.20% | 44.00% | 43.00% +/- 4.15% |
| Attention | 43.89% +/- 3.01% | 44.80% | 43.60% | 42.40% +/- 2.73% |
| Hybrid | 42.99% +/- 5.72% | 44.00% | 43.00% | 42.20% +/- 5.34% |
| Decision Fusion | 44.96% +/- 3.69% | 47.60% | 45.20% | 43.80% +/- 3.66% |

### 3b. Before vs After (Inflated vs Valid)

| Model | Original (Inflated) | Corrected (Valid) | Difference |
|-------|--------------------|-------------------|------------|
| MLP | 76.03% | 46.94% | DOWN 29.09% |
| DNN | 90.35% | 44.01% | DOWN 46.34% |
| Attention | 68.26% | 43.89% | DOWN 24.37% |
| Hybrid | 92.84% | 42.99% | DOWN 49.85% |
| Decision Fusion | 50.55% | 44.96% | DOWN 5.59% |

---

## 4. Key Insights

### Why accuracy dropped dramatically

The original pipeline achieved up to **92.84%** accuracy due to three compounding leakages:

1. **Within-subject window leakage**: Windows from the same subject appeared in both train and
   test sets. Since consecutive EEG windows share signal characteristics (same brain state,
   same session), the model effectively memorized subject-specific patterns rather than
   learning generalizable emotion features. This is not a real generalization test.

2. **StandardScaler leakage**: `fit_transform(X_full)` was called before splitting,
   meaning the scaler's mean/std were computed using test subjects' data.

3. **PCA leakage**: The pre-generated `X_eeg_pca.npy` was fitted on all 15 subjects.
   PCA components were shaped by test variance, providing an unfair advantage.

4. **Model selection leakage**: The 'best' model was selected using test accuracy across
   epochs, turning the test set into a de facto validation set.

### Why corrected results (43-47%) are realistic

- Each test fold contains subjects **completely unseen** during training.
- PCA and scaling use **only training statistics** — test subjects are strictly held out.
- Subject-independent emotion recognition on SEED-IV is genuinely hard:
  - EEG signals vary significantly between individuals (non-stationary, session-dependent).
  - Without fine-tuning, cross-subject performance typically falls in **40-70%** range.
  - State-of-the-art subject-independent models (with domain adaptation, transformers)
    achieve 65-80%.
- 25% = random baseline (4 classes). Results of 43-47% show the models ARE learning
  some generalization, just not as much as the leaky pipeline suggested.

### Scientific validity comparison

| Aspect | Original | Corrected |
|--------|---------|----------|
| Evaluation type | Random window split | Subject-level CV |
| Leakage present | YES (3 types) | NO |
| Cross-validation | No (single split) | Yes (5-fold) |
| Model selection | On test set | On held-out val subject |
| Claimable in research | NO | YES |

---

## 5. Model Architecture Notes (Unchanged)

| Model | Architecture | Corrected Acc |
|-------|-------------|--------------|
| MLP | 2-layer FC (128->64->4) + BN + Dropout | 46.94% +/- 2.97% |
| DNN | 3-layer FC (256->128->64->4) + Dropout | 44.01% +/- 3.97% |
| Attention | Sigmoid-gated element-wise attention + FC | 43.89% +/- 3.01% |
| Hybrid | FC + Attention gate (128->64->4) | 42.99% +/- 5.72% |
| Decision Fusion | Separate EEG-PCA + Eye streams, averaged logits | 44.96% +/- 3.69% |

---

## 6. Files Generated

```
stage4_models_corrected/
  fold_1/ ... fold_5/          # Per-fold model weights + per-fold metrics
    mlp/  dnn/  attention/  hybrid/  decision_fusion/
      best_model_fold{k}.pth
      accuracy_fold{k}.txt
      report_fold{k}.txt
      confusion_matrix_fold{k}.png
  comparison/
    model_comparison_corrected.csv    # Aggregated 5-fold results
    performance_corrected.png         # Before vs After chart
  objective2_corrected_report.md      # This report
```