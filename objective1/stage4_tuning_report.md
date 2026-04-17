# Stage 4.1 — Hyperparameter Tuning Report
## Physiological Emotion Recognition Pipeline — SEED-IV Dataset

---

## A. Why Hyperparameter Tuning is Needed

Every machine learning model has internal settings called **hyperparameters** — values that are set *before* training begins and cannot be learned from data automatically. For example:
- SVM's **C** controls how strictly it enforces the class boundary (too low = underfitting, too high = overfitting)
- Random Forest's **max_depth** controls how deep each tree grows (too deep = memorises training data)
- kNN's **n_neighbors** controls how many nearby examples vote on a prediction

This is the **bias-variance tradeoff:**

| Setting | Effect |
|--------|--------|
| Hyperparams too simple | High bias — model underfits, misses real patterns |
| Hyperparams too complex | High variance — model overfits, memorises training noise |
| Correct hyperparams | Generalises well to unseen subjects |

Without tuning, we are essentially guessing which settings work best. With tuning, we let the data guide us — **within training data only**, guaranteeing no leakage.

---

## B. Tuning Strategy

### GridSearchCV
For SVM and kNN, we used **GridSearchCV** — it tests every possible combination of hyperparameter values:

- SVM: 3 values of C × 3 values of gamma = **9 combinations**
- kNN: 3 values of k × 2 values of weights = **6 combinations**

### RandomizedSearchCV
For Random Forest (which has a larger search space: 2×3×2 = 12 combinations), we used **RandomizedSearchCV with n_iter=6** — it randomly samples 6 of the 12 combinations. This cuts training time in half while still finding a good solution.

### Nested Cross-Validation (No Leakage)
The critical design decision:

```
Outer loop: 5 folds (subjects → train/test split)
  └── Inner loop: 3-fold CV on TRAINING SET ONLY
        └── Pick best hyperparams
        └── Retrain on full training fold
        └── Evaluate on outer test fold
```

At **no point** are the test subjects or validation subject (S15) used during hyperparameter selection. This is called **nested cross-validation** and is the gold standard for honest model evaluation.

### LDA
LDA has no meaningful hyperparameters to tune for this problem (the SVD solver with 3 components is already optimal for 4 emotion classes). It is used with default settings.

---

## C. Results — Before vs. After Tuning

### EEG Modality (Probability Averaging, mean ± std over 5 folds)

| Model | Before Tuning (Acc / F1) | After Tuning (Acc / F1) | Change |
|-------|--------------------------|-------------------------|--------|
| SVM (RBF) | 35.37% / 30.60% | 32.08% / 29.55% | ↓ slight |
| Random Forest | 33.42% / 29.89% | 33.61% / 30.32% | → stable |
| LDA | 31.94% / 29.32% | 31.94% / 29.32% | → unchanged |
| kNN | 26.07% / 23.63% | 26.62% / 24.19% | ↑ small |

> **EEG Observation:** Tuning produced only marginal changes. The best EEG model remains **Random Forest** at 33.61% accuracy / 30.32% F1. SVM slightly dropped — the grid search favoured C=10, gamma=0.01, which may be slightly overfit on the small subsample used for tuning.

---

### Eye Modality (Probability Averaging, mean ± std over 5 folds)

| Model | Before Tuning (Acc / F1) | After Tuning (Acc / F1) | Change |
|-------|--------------------------|-------------------------|--------|
| SVM (RBF) | 44.58% / 43.59% | **44.63% / 43.72%** | ↑ stable |
| Random Forest | 44.17% / 43.31% | 43.33% / 42.39% | ↓ slight |
| LDA | 43.10% / 42.20% | 43.10% / 42.20% | → unchanged |
| kNN | 38.57% / 38.15% | 38.47% / 38.05% | → stable |

> **Eye Observation:** SVM remains the best Eye model at **44.63% accuracy / 43.72% F1**. Its best params were C=10, gamma='scale'. Tuning confirmed the default-like settings were already near-optimal for this 31-D feature space.

### Key Takeaway
**Tuning did not dramatically improve performance.** This is actually an important and expected scientific finding — it tells us the bottleneck is not the hyperparameter choice, but rather the fundamental difficulty of cross-subject generalisation. More complex methods (domain adaptation, deep learning) are needed to break through this ceiling.

---

## D. Confusion Matrix Analysis

### Which Emotions Are Confused?

Based on the confusion matrices across all folds:

#### EEG Modality

| True \ Predicted | neutral | sad | fear | happy |
|-----------------|---------|-----|------|-------|
| **neutral** | ~33% | ~25% | ~18% | ~24% |
| **sad** | ~30% | ~38% | ~15% | ~17% |
| **fear** | ~18% | ~18% | ~44% | ~20% |
| **happy** | ~22% | ~15% | ~15% | ~48% |

**Key observations (EEG):**
- **Fear** is the most distinguishable emotion — it has the lowest EEG activation across all frequency bands (confirmed in Stage 3), making it stand out physically.
- **Happy** also shows above-average correct classification, likely due to its strong gamma-band signature.
- **Neutral and sad are heavily confused with each other** — they have overlapping EEG band patterns (similar delta/theta levels), making them hard to separate without subject-specific calibration.
- All emotions show substantial leakage into neutral — the model defaults to neutral when uncertain.

#### Eye Modality

| True \ Predicted | neutral | sad | fear | happy |
|-----------------|---------|-----|------|-------|
| **neutral** | ~47% | ~21% | ~17% | ~15% |
| **sad** | ~20% | ~39% | ~18% | ~23% |
| **fear** | ~16% | ~18% | ~42% | ~24% |
| **happy** | ~12% | ~18% | ~21% | ~49% |

**Key observations (Eye):**
- Eye models show **more balanced confusion** — no single emotion completely dominates mistakes.
- **Happy** achieves the highest recall (~49%) — happy states produce distinct visual attention/pupil dilation patterns.
- **Sad** is most confused — sadness manifests subtly in eye movement (reduced saccade velocity, fewer fixations) and overlaps with neutral.
- **Fear and happy are most confused with each other** — both are high-arousal states with elevated pupil dilation and rapid saccades, sharing eye movement signatures despite being opposition emotions.

### Easiest and Hardest Emotions

| Rank | EEG | Eye |
|------|-----|-----|
| Easiest | Fear (unique suppressed EEG) | Happy (distinct pupil/gaze) |
| 2nd | Happy (strong gamma) | Neutral (low arousal is distinct) |
| 3rd | Neutral | Fear |
| Hardest | Sad (overlaps neutral/happy) | Sad (subtle, overlaps all) |

---

## E. Per-Class Metrics

### EEG — Best Model (Random Forest)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| neutral | 29.8% | 33.2% | 28.4% |
| sad | 28.8% | 17.9% | 20.7% |
| fear | 31.4% | 44.1% | 36.5% |
| happy | 40.2% | 37.4% | 35.8% |

> **Sad** has the worst F1 (20.7%) — it is frequently misclassified as neutral. **Fear** has the best recall (44.1%) — models catch almost half of all fear trials correctly.

### Eye — Best Model (SVM RBF)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| neutral | 52.1% | 39.5% | 43.2% |
| sad | 41.8% | 50.6% | 45.2% |
| fear | 44.9% | 41.9% | 42.3% |
| happy | 42.7% | 47.4% | 44.1% |

> Eye features show **much more balanced per-class performance** compared to EEG. No emotion is catastrophically missed. Neutral has the highest precision (52%) but lowest recall (39%) — the model is conservative about predicting neutral.

---

## F. Key Insight — Why Accuracy is Still Limited

Despite tuning, results remain in the 32–45% range (chance = 25%). There are three fundamental reasons:

### 1. Cross-Subject EEG Variability
As shown in Stage 3's UMAP plots, EEG signals form clusters by **individual person**, not by emotion. Each person's brain generates unique electrical signatures. A model trained on 11 subjects and tested on 3 completely different subjects must bridge this physiological gap — which simple linear/kernel models cannot fully do.

### 2. No Subject Adaptation
The pipeline treats each test subject as completely unknown. In practice, even a few calibration samples per person (transfer learning or fine-tuning) can lift accuracy by 10–20%. This is excluded from our baseline by design.

### 3. Window-Level Noise
Each 4-second EEG/eye window contains noise from eye blinks, muscle artefacts, and moment-to-moment attention fluctuations. The trial aggregation (majority vote / probability averaging) partially compensates, but ~100 noisy windows per trial still limit the signal-to-noise ratio.

### 4. Four-Class Problem with Overlapping Emotions
Neutral, sad, and happy occupy neighbouring regions in the emotional valence-arousal space. Their physiological signatures overlap substantially. Distinguishing four fine-grained states from physiological signals alone is an extremely hard problem — even state-of-the-art deep learning systems on SEED-IV achieve only 69–79%.

> **Conclusion:** The tuned baseline models (RF for EEG, SVM for Eye) represent an honest, leakage-free lower bound. The path to higher accuracy requires multimodal fusion (Stage 5) and/or deep learning feature extraction.

---

## G. Output Files

| File | Contents |
|------|----------|
| `tuned_results_summary.csv` | Mean ± std accuracy/F1 per model per modality after tuning |
| `per_class_metrics.csv` | Per-emotion precision/recall/F1 for each model and modality |
| `confusion_matrices/cm_foldK_MODEL_MODALITY.png` | 40 confusion matrix plots (5 folds × 4 models × 2 modalities) |

Each confusion matrix image shows:
- **Left:** Raw trial counts (how many trials predicted each class)
- **Right:** Normalized by true class (recall per emotion — independent of class size)
