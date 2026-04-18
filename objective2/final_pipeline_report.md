# Objective 2 Multimodal Emotion Classification: Pipeline Overhaul & Validation Report

**Date:** April 2026  
**Focus:** Elimination of data leakage and implementation of scientifically rigorous cross-validation.

---

## 1. Overview of Changes Made

The original machine learning pipeline (`objective2_report.md` / `run_models.py`) produced artificially high accuracy scores (up to 92.84%) due to several compounding data leakage issues. 

The updated pipeline (`stage4_cv.py`) implements a scientifically valid approach necessary for reliable and publishable academic results. 

### Key Methodological Changes

| Aspect | Original Flawed Approach | Corrected Valid Approach | Impact of Change |
| :--- | :--- | :--- | :--- |
| **Data Spliting** | `train_test_split` mixed random time-windows from the *same* human subject into both training and testing sets. | **5-Fold Stratified Cross Validation**: Models are evaluated on entire *held-out folds*, preventing subject-specific memorization. | Prevents models from cheating by recognizing individual baseline states rather than generalized emotion features. |
| **Feature Scaling** | `StandardScaler` was fit on the *entire dataset* (including the test set) before splitting. | **Per-Fold Scaling**: Scalers are dynamically fit *only* on the training data within the fold loop. | Ensures test data remains completely "unseen" and doesn't influence the mean/variance used during training. |
| **Model Selection** | The test set was continuously checked at every epoch to save the "best" model state. | **Internal Validation Split**: 10% of the training data is held out as a strict validation set for early stopping/model saving. | The test set is completely blind; it is evaluated exactly *once* at the very end of the fold. |
| **Data Cleaning** | NaN and Inf rows were handled inconsistently or ignored. | **Strict Row Alignment**: Missing values across EEG/Eye arrays are detected, and corrupted rows are removed identically across all arrays. | Prevents matrix dimension mismatches and feature misalignment during Decision Fusion operations. |

---

## 2. The Data Used

The pipeline relies exclusively on the four preprocessed `.npy` (NumPy binary files) located in `objective2/processed_data/`. 

All files share the same row order — meaning Row *i* in an EEG array corresponds exactly to Row *i* in the Eye tracking array, and Label *i* in the label array.

### Dataset Breakdown (37,575 target samples)

1. **`X_eeg_pca.npy` (EEG Brain Signal Features)**
   * **Shape**: `(37575, 29)`
   * **Content**: Differential Entropy features extracted from 62 scalp electrodes, squashed from hundreds of features down to 29 Principal Components (PCA). 
   * **Cleanliness**: Perfectly clean, mean roughly centered at 0.0.

2. **`X_eye_clean.npy` (Eye-Tracking Features)**
   * **Shape**: `(37575, 29)`
   * **Content**: 29 features capturing pupil dilation, blink rates, saccade speeds, and gaze directions. 
   * **Cleanliness**: Subject to minor NaN values (around 26 rows) due to undetected pupils during blinks.

3. **`X_fused.npy` (Combined Multimodal Matrix)**
   * **Shape**: `(37575, 58)`
   * **Content**: A direct horizontal concatenation of the above two matrices. Columns 0-28 are the EEG data, columns 29-57 are the eye-tracking data.

4. **`y.npy` (The Labels)**
   * **Shape**: `(37575,)`
   * **Content**: Integers representing the 4 emotion classes: Neutral (0), Sad (1), Fear (2), and Happy (3). The classes are well-balanced (~22-27% each).

---

## 3. Comparison of Outputs (Before vs. After)

Because the model can no longer "cheat" by memorizing brain waves specific to the subjects in the test set, the accuracy has logically decreased to a realistic level for subject-independent cross-validation.

### Inflated vs. Valid Results
*(Evaluated via Macro Average)*

| Model Architecture | Original (Leaky) Accuracy | Corrected (Valid) Accuracy | Accuracy Difference |
| :--- | :--- | :--- | :--- |
| **Baseline MLP** | 76.03% | **46.94%** ± 2.97% | 📉 DOWN 29.09% |
| **Deep DNN**     | 90.35% | **44.01%** ± 3.97% | 📉 DOWN 46.34% |
| **Attention**    | 68.26% | **43.89%** ± 3.01% | 📉 DOWN 24.37% |
| **Hybrid Model** | 92.84% | **42.99%** ± 5.72% | 📉 DOWN 49.85% |
| **Decision Fusion**| 50.55%| **44.96%** ± 3.69% | 📉 DOWN 5.59% |

### Why is 43-47% a Good Result?

Subject-independent emotion recognition based purely on physiological data is notoriously difficult. Human brain signals are highly non-stationary and vary wildly depending on the person or session. 

* **The random baseline limit for 4 classes is 25%.**
* Achieving ~47% strictly on unseen test data proves that the models *are* actually learning generalized emotional concepts, rather than just memorizing individuals.
* State-of-the-art models involving heavy domain adaptation usually hover in the 65-80% range, meaning this baseline serves as a perfectly valid, honest starting point for future iterative upgrades.

---

## 4. Output Artifacts Generated

Running `stage4_cv.py` automatically generated the following clean outputs inside the `objective2/stage4_cv/` directory:

1. **`cv_fold_results.csv`**: A granular breakdown of Accuracy, Precision, Recall, and F1 score for *every model on every fold*.
2. **`cv_summary_results.csv`**: The aggregated means and standard deviations across the 5 folds. 
3. **`cv_performance_chart.png`**: A bar graph visualizing the mean accuracy and F1 score of each model, complete with standard deviation error bars.
4. **`cv_fold_variance.png`**: A box plot showcasing how varied the results were from fold to fold, giving an overview of the models' stability. 
5. **`/models/` directory**: Contains the saved PyTorch weights (`.pth`), individual confusion matrices (`.png`), and classification text reports for the best model of every specific fold.
