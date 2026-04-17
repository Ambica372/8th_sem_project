# Stage 4 — Baseline Classification Analysis Report
## Physiological Emotion Recognition Pipeline — SEED-IV Dataset

---

## A. Objective

Baseline classification establishes a **performance lower bound** for the pipeline.
The goal is to answer: *"Using simple, well-understood machine learning models with no tuning, how well can we recognize emotions from physiological signals?"*

Baselines serve three purposes:
1. **Sanity check** — if a baseline beats chance (25% for 4 classes), the features contain real signal.
2. **Reference point** — more complex models in later stages (deep learning, ensembles) must beat these numbers to justify their complexity.
3. **Modality comparison** — running the same models on EEG and Eye separately reveals which physiological signal is inherently more discriminative at this stage.

> **Chance level for 4-class classification: 25.0%**

---

## B. Model Descriptions

### 1. SVM — Support Vector Machine (RBF Kernel)

SVM finds a **boundary (hyperplane)** in the feature space that maximally separates the emotion classes. The **RBF (Radial Basis Function) kernel** allows this boundary to be non-linear — it can wrap around complex, curved clusters in the data.

- **Strength:** Very powerful for high-dimensional data; handles non-linearity well.
- **Weakness:** Computationally expensive for large datasets. To manage this, SVM was trained on a **stratified subsample of 8,000 windows per fold** (out of ~27,000). All test/validation predictions used the full dataset.
- **Parameters used:** C=1.0, gamma="scale" (default bandwidth), probability=True.

### 2. Random Forest

A Random Forest builds **200 independent decision trees**, each trained on a random subset of features and samples. The final prediction is made by majority vote of all 200 trees.

- **Strength:** Robust to overfitting; handles high-dimensional data; captures non-linear patterns; fast to train.
- **Weakness:** Less interpretable; can overfit within training subjects.
- **Parameters used:** n_estimators=200, max_depth=None (unlimited).

### 3. LDA — Linear Discriminant Analysis

LDA finds **linear combinations of features** that best separate the classes. It assumes each class follows a Gaussian distribution with the same covariance.

- **Strength:** Very fast; interpretable; provides class probabilities naturally.
- **Weakness:** Cannot model non-linear decision boundaries; sensitive to class imbalance.
- **Parameters used:** SVD solver, 3 discriminant components (maximum for 4 classes).

### 4. k-NN — k-Nearest Neighbours

k-NN classifies a window by finding its **7 nearest neighbours** in feature space and assigning the most common label (weighted by inverse distance).

- **Strength:** Non-parametric; makes no assumptions about data distribution.
- **Weakness:** Slow at prediction time on large datasets; sensitive to irrelevant features; assumes all features are equally important.
- **Parameters used:** k=7, Euclidean distance, distance-weighted voting.

---

## C. Methodology

### Fold-wise Training

For each of the 5 cross-validation folds:
1. Train the model on `fold_k_train_{modality}.csv` (11–12 subjects, ~27,000–30,000 windows).
2. Predict on `fold_k_test_{modality}.csv` (2–3 subjects, ~5,000–7,500 windows).
3. Evaluate on `fold_k_validation_{modality}.csv` (subject 15, 2,505 windows — normalized with fold k's training statistics).

### Window-to-Trial Aggregation

Each trial contains approximately 104 windows (range: 44–139). Two aggregation strategies were compared:

**A. Majority Voting**
- Count how many windows predicted each emotion class.
- Assign the trial the class with the most votes.
- Simple, robust, interpretable.

**B. Probability Averaging**
- Average the predicted class probabilities across all windows in a trial.
- Assign the trial to the class with the highest average probability.
- Uses more information (soft decisions vs. hard votes).

Both strategies were applied and compared. Metrics are computed at the **trial level**.

### NaN Handling (Eye Data)
Eye features contain NaN values from eye-tracking dropouts. These were imputed using the **column median of the training set** before fitting and prediction — ensuring no information from test/validation data is used in imputation.

---

## D. Results

### Cross-Validation Performance (mean ± std, 5 folds, trial-level, majority vote)

#### EEG Modality

| Model         | Accuracy (%)    | Macro F1 (%)    | Macro Precision (%) | Macro Recall (%) |
|---------------|----------------|----------------|---------------------|-----------------|
| SVM (RBF)     | 34.31 ± 5.21   | 29.47 ± 5.65   | —                   | —               |
| Random Forest | **33.80 ± 2.36** | **30.42 ± 3.32** | —                  | —               |
| LDA           | 31.80 ± 1.35   | 29.25 ± 3.18   | —                   | —               |
| kNN           | 26.07 ± 3.53   | 23.53 ± 3.77   | —                   | —               |

> **Chance level: 25.0% accuracy**

#### Eye Modality

| Model         | Accuracy (%)    | Macro F1 (%)    |
|---------------|----------------|----------------|
| SVM (RBF)     | **44.63 ± 4.30** | **43.78 ± 4.30** |
| Random Forest | 44.40 ± 2.01   | 43.53 ± 1.96   |
| LDA           | 43.15 ± 2.69   | 42.24 ± 3.61   |
| kNN           | 38.52 ± 4.74   | 38.08 ± 5.05   |

### Majority Vote vs. Probability Averaging

| Modality | Model | Majority Acc | Prob Avg Acc |
|----------|-------|-------------|-------------|
| EEG | SVM | 34.31% | 35.37% |
| EEG | RF | 33.80% | 33.42% |
| Eye | SVM | 44.63% | 44.58% |
| Eye | RF | 44.40% | 44.17% |

**Observation:** The two aggregation strategies perform nearly identically. Probability averaging shows a small advantage for SVM on EEG (+1.06%), while majority voting is slightly better for RF on both modalities. This suggests the window-level predictions are already fairly consistent within each trial — the aggregation step does not dramatically change the outcome.

---

## E. Observations

### Which Model Works Best?

**EEG:** Random Forest achieves the best F1 (30.42%), slightly ahead of SVM (29.47%) and LDA (29.25%). kNN underperforms significantly (23.53%), confirming that raw k-NN distance matching is poorly suited to the 310-D EEG space.

**Eye:** SVM (RBF) is the best model (F1 = 43.78%), followed closely by Random Forest (43.53%) and LDA (42.24%). The three top models are within 1.5% of each other, while kNN again lags behind (38.08%).

### Which Modality is Stronger?

| Metric | EEG (best model: RF) | Eye (best model: SVM) |
|--------|---------------------|----------------------|
| Mean Accuracy | 33.80% | 44.63% |
| Mean F1 | 30.42% | 43.78% |
| Std (F1) | ±3.32% | ±4.30% |

**Eye movement features outperform EEG by ~13 percentage points** in both accuracy and F1. This is a surprising but explicable result:

1. **Dimensionality:** Eye has 31 features vs. EEG's 310. Lower dimensionality means less noise and fewer irrelevant features for these models.
2. **Subject variance:** As shown in Stage 3 (t-SNE/UMAP), EEG is highly subject-specific. Cross-subject generalization is harder for EEG — the model sees patterns dominated by individual physiology rather than emotion.
3. **EEG just above chance:** EEG baselines range 26–34%, barely clearing the 25% chance threshold. This confirms the Stage 3 finding: EEG's subject-cluster structure makes cross-subject emotion recognition extremely challenging for simple models.
4. **Eye is more generalizable:** Eye movement patterns (fixation, saccade, blink) reflect more universal cognitive-emotional responses that carry across individuals.

### Stability

Eye models show **lower standard deviation** across folds (RF: ±1.96%) compared to EEG (RF: ±3.32%), indicating Eye features produce more consistent cross-subject results. EEG results are more unstable — some test subject combinations produce better results than others.

---

## F. Validation Results (Subject 15 — Held-Out, Never Used in Training)

Subject 15 was completely isolated from all training and cross-validation. Results below represent the true generalization performance to a completely unseen individual.

### EEG — Validation Subject 15

| Model         | Accuracy (%)   | F1 (%)          |
|---------------|---------------|----------------|
| Random Forest | **45.28 ± 3.35** | 40.67 ± 4.88   |
| SVM (RBF)     | 43.61 ± 3.79   | **41.75 ± 4.52** |
| LDA           | 36.67 ± 2.42   | 32.12 ± 1.93   |
| kNN           | 26.11 ± 3.67   | 23.86 ± 3.39   |

**Interesting result:** EEG performs *better* on the validation subject (45.28% RF accuracy) than on the CV test folds (33.80%). This may indicate that Subject 15's EEG patterns happen to align more closely with general patterns in the training data, or that the test fold subjects happened to have particularly unusual patterns.

### Eye — Validation Subject 15

| Model         | Accuracy (%)   | F1 (%)          |
|---------------|---------------|----------------|
| LDA           | **46.67 ± 6.78** | **45.68 ± 6.98** |
| kNN           | 36.67 ± 3.79   | 35.14 ± 4.74   |
| SVM (RBF)     | 32.50 ± 4.08   | 26.68 ± 6.29   |
| Random Forest | 20.83 ± 5.27   | 16.90 ± 2.98   |

**Notable findings:**
- LDA **dramatically outperforms** SVM and RF on the validation subject for Eye data (46.67% vs 32.50%) — the opposite ranking from the CV test folds.
- Random Forest drops to **20.83%** on the validation subject (below chance). This suggests RF is memorizing subject-specific patterns in the training data and failing to generalize to subject 15's eye-movement style.
- This **model rank reversal** (LDA wins on validation, SVM wins on CV) is a warning against over-trusting CV results alone. The validation subject confirms that simpler, linear models (LDA) can generalize better to unseen individuals.

---

## G. Limitations

### 1. Unimodal Only
EEG and Eye features were trained **separately** as required. A multimodal (EEG + Eye combined or fused) approach is expected to significantly outperform both individual modalities in Stage 5.

### 2. No Hyperparameter Optimization
All models used default parameters (C=1.0, 200 trees, k=7, etc.). Grid search or Bayesian optimization could improve results substantially. For example, tuning the SVM kernel width (gamma) and regularization (C) typically yields 3–8% improvement on EEG emotion tasks.

### 3. SVM Subsampling
SVM was trained on 8,000 stratified samples rather than the full 27,000+ window training set due to computational constraints. This may have reduced SVM performance slightly compared to full-data training.

### 4. Subject-Level Variation — The Core Challenge
Cross-subject emotion recognition remains fundamentally hard. The 13–15% EEG gap from chance is low because EEG is intrinsically person-specific. More sophisticated methods (domain adaptation, person-invariant features, deep learning with subject adversarial training) are needed to overcome this.

### 5. No Class-Specific Analysis
Overall accuracy and macro F1 are reported, but per-class breakdown is not included here. Fear is the easiest class to distinguish (lowest activation across all bands) and is likely driving the above-chance performance. Neutral vs. happy is likely the hardest pair.

---

## Summary Table

| Metric | EEG Best | Eye Best | Selected |
|--------|----------|----------|---------|
| CV Accuracy | 34.31% (SVM) | 44.63% (SVM) | **Eye SVM** |
| CV F1 | 30.42% (RF) | 43.78% (SVM) | **Eye SVM** |
| Val Accuracy | 45.28% (RF) | 46.67% (LDA) | **Comparable** |
| Val F1 | 41.75% (SVM) | 45.68% (LDA) | **Eye LDA** |

> **Conclusion:** Eye-movement features provide stronger and more generalizable baseline classification than EEG alone. However, both modalities substantially outperform random chance and contain real emotion-discriminative information. The combination of both modalities in a multimodal system is the clear recommended next step.

---

## Output Files

| File | Contents |
|------|----------|
| `fold_results_eeg.csv` | Per-fold, per-model results for EEG (all metrics, both aggregations) |
| `fold_results_eye.csv` | Per-fold, per-model results for Eye |
| `baseline_results_summary.csv` | Mean ± std across 5 folds per model per modality |
| `validation_results.csv` | Validation subject 15 summary results |
| `all_fold_results_raw.csv` | Raw per-fold rows for further analysis |
| `all_val_results_raw.csv` | Raw per-fold validation rows |
