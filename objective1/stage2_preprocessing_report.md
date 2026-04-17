# Stage 2 Preprocessing Report
## Physiological Emotion Recognition Pipeline — SEED-IV Dataset

---

## A. Explanation of Approach

### What is Subject-Level Splitting?

In emotion recognition from physiological signals, each person (subject) produces data windows across multiple trials. A **subject-level split** ensures that every window from a given subject stays entirely within one partition — either training or testing — never both.

**Why this matters:** EEG and eye-movement signals are highly person-specific. If windows from the same subject appear in both the training and test sets, the model can inadvertently learn that subject's personal physiological patterns and score inflated accuracy — a form of **data leakage**. Subject-level splitting forces the model to generalize to *new* unseen subjects, which is the true test of a generalizable emotion recognition system.

### Why is a Held-Out Validation Subject Used?

Beyond cross-validation, one subject (**Subject 15**) is completely removed from all cross-validation folds. This subject is never used to:
- Compute normalization statistics
- Train any model
- Tune any hyperparameters

This held-out subject serves as a **final, unbiased benchmark** — analogous to a real deployment scenario where the system must work for a person it has never encountered. Reporting performance on this subject gives the most honest estimate of real-world generalization.

### Why is Cross-Validation Needed?

With only 15 subjects total and one held out for validation, the development pool has 14 subjects. Training on all 14 and testing on the held-out subject gives only one evaluation point — statistically fragile. **5-Fold Cross-Validation** at the subject level:

1. Gives 5 independent train/test evaluations.
2. Ensures every development subject serves as test data exactly once.
3. Allows robust averaging of performance metrics across folds.
4. Reduces variance from lucky/unlucky assignment of hard subjects to test sets.

---

## B. Methodology

### Step-by-Step: Fold Creation

1. **Identify all unique subjects** in the dataset: subjects 1–15.
2. **Remove the held-out validation subject** (Subject 15). Development pool: subjects 1–14.
3. **Apply scikit-learn's `KFold`** with `n_splits=5`, `shuffle=True`, `random_state=42` **on the subject IDs** (not on individual rows).
4. Each fold receives:
   - **Training fold**: 4/5 of development subjects (11–12 subjects)
   - **Test fold**: 1/5 of development subjects (2–3 subjects)
5. All windows belonging to a subject follow that subject into their assigned partition.

**Resulting fold splits:**

| Fold | Train Subjects              | Test Subjects |
|------|-----------------------------|---------------|
| 1    | 2,3,4,5,6,7,8,9,11,13,14    | 1, 10, 12     |
| 2    | 1,2,3,4,5,7,8,10,11,12,14   | 6, 9, 13      |
| 3    | 1,4,5,6,7,8,9,10,11,12,13   | 2, 3, 14      |
| 4    | 1,2,3,4,6,7,9,10,12,13,14   | 5, 8, 11      |
| 5    | 1,2,3,5,6,8,9,10,11,12,13,14| 4, 7          |

Each development subject appears in the test set exactly **once** across all 5 folds.

---

### Step-by-Step: Normalization

**Z-score normalization** transforms each feature to have zero mean and unit standard deviation:

```
z = (x - mu) / sigma
```

The critical rule: **mu and sigma are computed only from training data.**

For each fold:

1. **Collect training windows**: all rows belonging to training subjects.
2. **Compute per-column statistics** from the training set only:
   - `mu[j]`    = mean of column `j` across all training windows
   - `sigma[j]` = sample standard deviation (ddof=1) of column `j`
3. **Handle zero-variance columns**: if `sigma[j] == 0` (constant feature), set `sigma[j] = 1` to avoid division by zero.
4. **Apply to all three partitions** using the *same* training mu and sigma:
   - Training data:  `(train - mu) / sigma`
   - Test fold:      `(test  - mu) / sigma`
   - Validation:     `(val   - mu) / sigma`
5. **NaN values** (208 NaNs in eye-tracking data from blink/tracking gaps) are preserved. Pandas `mean()` and `std()` skip NaNs by default. NaN positions remain NaN after normalization.

> **EEG and Eye are normalized completely separately** — they have different physical scales and are processed by independent calls to `normalize_data()`.

---

## C. Data Leakage Discussion

### What is Data Leakage?

Data leakage occurs when information from outside the training set is used during model training or preprocessing, giving the model an unfair advantage. In physiological signal processing, two main leakage forms exist:

| Leakage Type | How It Occurs | Prevention |
|---|---|---|
| **Subject leakage** | Splitting windows from the same subject into both train and test | Subject-level KFold — entire subjects assigned to one partition |
| **Statistical leakage** | Computing normalization parameters using test/validation data | mu and sigma computed exclusively on training subjects per fold |
| **Temporal leakage** | Windows from the same trial straddling the train/test boundary | Subjects are split, not individual windows or trials |

### Why Computing mu/sigma from the Full Dataset is Wrong

A common mistake is to normalize the entire dataset first, then split. This is **leakage** because:
- Normalization parameters (mu, sigma) encode distributional information from the test subject.
- Even without exposing labels, feature statistics from the test subject bleed into training preprocessing.
- This inflates performance metrics and produces models that fail in real deployment.

### Our Pipeline's Anti-Leakage Guarantees

1. `split_subjects()` separates Subject 15 **before** any normalization.
2. `create_folds()` splits subjects **before** any normalization.
3. `normalize_data()` computes statistics **strictly from training arrays**.
4. The held-out validation subject is never inspected during fold creation.

---

## D. Observations

### Dataset Structure

| Metric | Value |
|---|---|
| Total subjects | 15 |
| Sessions per subject | 3 |
| Windows per subject | 2,505 |
| Total windows | 37,575 |
| EEG features per window | 310 |
| Eye features per window | 31 |
| Missing (NaN) values — Eye | 208 |
| Missing (NaN) values — EEG | 0 |

### Emotion Label Distribution

| Label | Emotion | Windows | Proportion |
|-------|---------|---------|------------|
| 0 | Neutral | 9,855 | 26.2% |
| 1 | Sad | 8,865 | 23.6% |
| 2 | Fear | 9,825 | 26.1% |
| 3 | Happy | 9,030 | 24.0% |

The dataset is **well-balanced** across four emotion classes.

### Subject Distribution Across Folds

| Subject | Appears in Test | Appears in Train |
|---------|----------------|-----------------|
| 1 | Fold 1 | Folds 2,3,4,5 |
| 2 | Fold 3 | Folds 1,2,4,5 |
| 3 | Fold 3 | Folds 1,2,4,5 |
| 4 | Fold 5 | Folds 1,2,3,4 |
| 5 | Fold 4 | Folds 1,2,3,5 |
| 6 | Fold 2 | Folds 1,3,4,5 |
| 7 | Fold 5 | Folds 1,2,3,4 |
| 8 | Fold 4 | Folds 1,2,3,5 |
| 9 | Fold 2 | Folds 1,3,4,5 |
| 10 | Fold 1 | Folds 2,3,4,5 |
| 11 | Fold 4 | Folds 1,2,3,5 |
| 12 | Fold 1 | Folds 2,3,4,5 |
| 13 | Fold 2 | Folds 1,3,4,5 |
| 14 | Fold 3 | Folds 1,2,4,5 |
| 15 | *Held-out validation* | — |

### Per-Fold Validation Results

| Fold | Subject Overlap | Feature Count | Train Mean ~0 | Train Std ~1 |
|------|----------------|---------------|----------------|--------------|
| 1 | NONE PASS | 310 PASS | PASS | PASS |
| 2 | NONE PASS | 310 PASS | PASS | PASS |
| 3 | NONE PASS | 310 PASS | PASS | PASS |
| 4 | NONE PASS | 310 PASS | PASS | PASS |
| 5 | NONE PASS | 310 PASS | PASS | PASS |

Eye DataFrame: 31 features confirmed, same structural results on all folds.

---

## E. Code Explanation

### `split_subjects(df, val_subject_id)`

Separates the full DataFrame into two disjoint subsets based on `subject_id`. The held-out subject's rows are entirely removed from the development pool before any further processing. Accepts either EEG or Eye DataFrames; called once for each modality.

**Key assurance:** An `assert` statement guarantees the requested validation subject actually exists in the data, failing loudly rather than silently producing an incorrect split.

### `create_folds(dev_subjects, n_folds, random_seed)`

Applies scikit-learn's `KFold` to an *array of subject IDs* (not to rows). This is the critical design choice — splitting on subject IDs means fold assignment operates at the subject level. Returns a list of dicts, each mapping `fold` → `{train_subjects, test_subjects}`.

**Reproducibility:** `shuffle=True` randomizes assignment; `random_state=42` ensures the same shuffle every run.

### `normalize_data(train_df, test_df, val_df, feature_cols)`

The core normalization function:
1. Computes column-wise mean and sample std from `train_df` only.
2. Replaces zero-std values with 1.0 to prevent division-by-zero on constant features.
3. Applies `(x - mu) / sigma` to all three DataFrames using training statistics.
4. Returns three normalized DataFrames plus a statistics dictionary for auditing.

**NaN safety:** Pandas `mean()` and `std()` skip NaN values by default — eye-tracking gaps do not corrupt normalization statistics.

---

## F. Output Files Description

All output files are saved to `stage2_output/`.

### Per-Fold Files (5 folds x 4 files = 20 CSVs)

| Filename | Contents |
|---|---|
| `fold_k_train_eeg.csv` | Training EEG windows for fold k. Z-score normalized with training-set statistics. 27,555–30,060 rows x 315 columns (5 meta + 310 features). |
| `fold_k_test_eeg.csv` | Test EEG windows for fold k. Normalized using the same training statistics. 5,010–7,515 rows x 315 columns. |
| `fold_k_train_eye.csv` | Training Eye windows for fold k. Normalized independently from EEG. 36 columns (5 meta + 31 features). |
| `fold_k_test_eye.csv` | Test Eye windows for fold k. Uses training Eye normalization parameters. |

### Validation Files (2 CSVs)

| Filename | Contents |
|---|---|
| `validation_eeg.csv` | Subject 15 EEG data, normalized using Fold 1 training statistics. 2,505 rows x 315 columns. |
| `validation_eye.csv` | Subject 15 Eye data, normalized using Fold 1 training statistics. 2,505 rows x 36 columns. |

### Manifest File

| Filename | Contents |
|---|---|
| `fold_manifest.json` | Machine-readable record of random seed, fold subject assignments, validation subject ID, feature counts. Guarantees full reproducibility. |

### Column Layout (same for all CSVs)

```
subject_id | session_id | trial_id | window_id | emotion_label | feat_0 | feat_1 | ... | feat_N-1
```

- `emotion_label`: 0=neutral, 1=sad, 2=fear, 3=happy
- `feat_*`: Z-score normalized features (mean=0, std=1 on training data)
- NaN values appear only in Eye feature columns for windows with eye-tracking dropout

---

## Summary

Stage 2 produced a complete, leakage-free cross-validation setup:

- **20 training/test fold CSVs** ready for model training in Stage 3
- **2 validation CSVs** held completely out of the pipeline
- **1 JSON manifest** for full reproducibility
- **All 5 folds passed** all mandatory validation checks

The pipeline is now ready to proceed to **Stage 3: Feature Analysis and Model Training**.
