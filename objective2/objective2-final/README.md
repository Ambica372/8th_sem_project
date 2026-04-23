# Multimodal Deep Learning for Emotion Recognition

**Objective 2 — Stage 4 Evaluation Pipeline**  
Multimodal fusion of EEG and eye-tracking features for 4-class emotion recognition, evaluated under two distinct cross-validation strategies to contrast performance-inflated window-level splitting against scientifically rigorous subject-level evaluation.

---

## Overview

This module implements a complete deep learning pipeline that:

- Fuses **EEG spectral/differential entropy (DE) features** with **eye-tracking behavioural features**
- Trains and evaluates **five model architectures** using both feature-level and decision-level fusion strategies
- Conducts two parallel evaluation tracks to expose and quantify **data leakage in window-based cross-validation**

### Evaluation Tracks

| Track | Strategy | Scope | Accuracy Range |
|---|---|---|---|
| **A — Stratified K-Fold** | Window-level splitting | Per-sample | ~80–88% |
| **B — GroupKFold (LOSO)** | Subject-level splitting | Per-subject | ~34–40% |

> The large performance gap (~45 percentage points) is not a model failure. It directly quantifies the **data leakage** present when temporal windows from the same subject appear in both train and test sets.

---

## Data

| Property | Details |
|---|---|
| Source | SEED-IV / multimodal EEG + eye dataset |
| Subjects | **15** |
| Total samples | **~37,500 windows** (post-cleaning) |
| EEG features | **29** (PCA-reduced differential entropy) |
| Eye features | **29** (cleaned gaze, blink, pupil metrics) |
| Fused features | **58** (horizontal concatenation) |
| Classes | **4** (neutral, sad, fear, happy) |
| Class distribution | Approximately balanced |

### Feature Engineering

- **EEG**: DE features extracted per frequency band (δ, θ, α, β, γ), then PCA-reduced to 29 components
- **Eye**: Raw gaze coordinates, fixation duration, blink rate, saccade velocity — cleaned and standardised
- **Fusion**: Early feature concatenation (`X_fused = [X_eeg | X_eye]`) for all models except Decision Fusion

---

## Model Architectures

All models share a consistent design philosophy: **BatchNorm → ReLU → Dropout (0.3)** per hidden block.

### 1. Baseline MLP
Two hidden layers (128 units each) with batch normalisation and dropout. Establishes a lower-bound baseline.

### 2. Deep DNN
Three-layer feedforward network (128 × 128 × 128). Adds representational depth over the MLP baseline.

### 3. Attention Model
Input-level feature gating: applies a sigmoid attention map directly to the input before the first dense layer. Learns *which features to attend to*.

```
attention_mask = sigmoid(Linear(input_dim → input_dim))
weighted_input = input × attention_mask
```

### 4. Hybrid Model (DNN + Attention)
Two-stage architecture: first projects input to 128 dimensions, then applies sigmoid attention in the hidden space before the second dense layer. Combines representational learning with selective focus.

### 5. Decision Fusion
Maintains **separate EEG and Eye processing streams**. Each stream independently classifies; final prediction is the mean of both logit outputs.

```
output = (eeg_stream_logits + eye_stream_logits) / 2
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 5×10⁻⁵ |
| Weight decay | 1×10⁻⁴ |
| Batch size | 64 |
| Dropout | 0.3 |
| Max epochs | 60 |
| Early stopping patience | 10 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=4) |
| Loss function | Cross-entropy with class balancing weights |
| Gradient clipping | max norm = 1.0 |
| Preprocessing | Per-fold StandardScaler (fit on train only) |
| Random seed | 42 |

> **No leakage guarantee**: The `StandardScaler` is fit exclusively on the training split of each fold and only applied (transformed) to the test split.

---

## Evaluation

### Track A — Stratified K-Fold (Window-Level)

- **Splitting**: `StratifiedKFold(n_splits=5)` on individual windows
- **Issue**: Windows from the same recording session (same subject) can appear in both training and test folds. The model effectively memorises **subject identity** rather than learning generalisable emotion patterns
- **Reported accuracy**: **~80–88%** (artificially inflated)
- **Scientific validity**: ❌ Not appropriate for cross-subject generalisation claims

### Track B — GroupKFold LASO (Subject-Level)

- **Splitting**: `GroupKFold(n_splits=15)`, grouped by `subject_id`
- **Guarantee**: No subject's windows appear in both train and test at any fold. True Leave-One-Subject-Out (LASO) rotation
- **Reported accuracy**: **~34–40%** (realistic)
- **Scientific validity**: ✅ Correct methodology for cross-subject emotion recognition

---

## Results — GroupKFold (LASO)

*Computed across all 15 folds (15 subjects × 5 models = 75 evaluations)*

| Model | Mean Accuracy | Std Dev |
|---|---|---|
| MLP | 34.93% | ±5.14% |
| DNN | 35.69% | ±4.37% |
| Attention | 36.43% | ±4.79% |
| **Hybrid** | 34.02% | ±4.47% |
| **Decision Fusion** | **40.30%** | **±6.14%** |
| **Overall mean** | **36.27%** | — |

### Key Observations

1. **Decision Fusion consistently outperforms all feature-fusion models** (+3–6 pp over Attention), demonstrating that keeping modalities separate preserves complementary information
2. **Attention improves over plain MLP/DNN**, confirming that input-level gating provides a useful inductive bias
3. **High fold variance** (std ~4–6%) reflects genuine subject-level variability in physiological signals — not model instability
4. **All models exceed random chance** (25% for 4 classes), confirming that learned representations have cross-subject generalisation

---

## Key Insight: Why the Accuracy Gap Matters

### Stratified K-Fold — The Leakage Problem

When splitting at the **window level**, consecutive windows from the same EEG recording (same subject, same session) are distributed across folds. During training the model observes subject-specific signal patterns (electrode drift, individual alpha rhythms, blink artefacts) that are unique to that person. When the same subject's windows appear at test time, the model trivially recognises these signatures.

This is not emotion recognition — it is **subject identity classification disguised as emotion recognition**.

### GroupKFold (LASO) — The Scientific Standard

Subject-level splitting forces the model to generalise to a person it has **never seen**. The test subject's entire physiological baseline is absent from training. The resulting ~34–40% accuracy is a honest measurement of whether the model has learned emotion-associated patterns that **transfer across individuals**.

### The 45 pp Gap

| Metric | Stratified (inflated) | GroupKFold (realistic) | Gap |
|---|---|---|---|
| Mean accuracy | ~84% | ~36% | **~48 pp** |

This gap is the direct measure of how much subject-specific leakage inflates reported performance in BCI and affective computing research.

---

## Folder Structure

```
objective2-final/
├── code/
│   ├── stratified_Laso_pipeline.py   # Track A: window-level StratifiedKFold
│   └── groupkfold_laso_pipeline.py   # Track B: LASO GroupKFold (15 subjects)
│
├── results/
│   ├── groupkfold_laso/              # ← Primary scientific results
│   │   ├── mlp/                      # best_fold{N}.pth checkpoints
│   │   ├── dnn/
│   │   ├── attention/
│   │   ├── hybrid/
│   │   ├── decision_fusion/
│   │   ├── cv_fold_results.csv       # Per-fold metrics (75 rows)
│   │   ├── cv_summary_results.csv    # Per-model mean ± std
│   │   ├── cv_performance_chart.png  # Horizontal bar chart
│   │   └── cv_fold_variance.png      # Boxplot across folds
│   │
│   └── stratified_laso/              # Track A results (reference only)
│
└── run.py                            # Entry-point launcher
```

---

## How to Run

### Run Both Pipelines

```bash
python run.py
```

### Run GroupKFold (LASO) Only

```bash
python objective2-final/code/groupkfold_laso_pipeline.py
```

### Run Stratified Only

```bash
python objective2-final/code/stratified_Laso_pipeline.py
```

> **Note**: The GroupKFold pipeline trains 5 models × 15 folds = 75 training runs. Estimated runtime: **~90–110 minutes** on CPU.

---

## Dependencies

```
torch >= 2.0
scikit-learn >= 1.3
numpy, pandas, matplotlib, seaborn
```

---

## Citation / Academic Note

This pipeline implements the evaluation framework proposed for multimodal physiological emotion recognition. Results from Track B (GroupKFold LASO) are the scientifically valid figures for cross-subject generalisation claims. Track A results are retained for comparative analysis only and should **not** be reported as the primary evaluation metric in academic submissions.
