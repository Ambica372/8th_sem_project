# Objective 3 Extension: GA-Based Feature Selection

> **Run date:** April 2026  
> **Runtime:** 36.8 minutes  
> **Extends:** PSO-Based Adaptive Fusion (Objective 3 core)

---

## 1. Problem: Feature-Level Variability

In both Objective 2 and the core Objective 3 PSO experiment, all fused
features (EEG-PCA + Eye) were used equally. The PSO scalar-weight experiment
showed only marginal improvement (+0.03%), suggesting that **variability
exists at the feature level** — some features are noisy, redundant, or
modality-specific in ways that hurt cross-subject generalization.

**Hypothesis:** Selecting a stable, informative subset of features may reduce
the impact of noise-driven variability and improve or clarify performance.

---

## 2. Why Genetic Algorithm?

Feature selection is a combinatorial optimization problem: with ~57 features,
there are $2^{57}$ possible subsets — exhaustive search is impossible.

A **Genetic Algorithm (GA)** is well-suited because:
- It searches a large discrete space efficiently via population-based evolution
- It naturally explores diverse subsets in parallel (via a population)
- It avoids local optima through crossover and mutation
- Binary chromosomes map directly to include/exclude decisions

---

## 3. Method

### 3.1 GA Design

| Parameter | Value |
|---|---|
| Population size | 20 |
| Generations | 15 |
| Crossover rate | 0.8 |
| Mutation rate (per gene) | 0.05 |
| Tournament size | 3 |
| Minimum features | 5 |
| Elitism | Yes (best always survives) |

### 3.2 Chromosome (Individual)

A binary vector of length = total fused features.
`1` = include feature, `0` = drop feature.

### 3.3 Genetic Operators

- **Selection:** Tournament selection (size 3) — probabilistic, preserves diversity
- **Crossover:** Single-point crossover — combines two parents' feature subsets
- **Mutation:** Per-gene bit-flip with probability 0.05 — introduces exploration
- **Elitism:** Best individual from each generation always carries over — prevents regression

### 3.4 Fitness Function (Fast Proxy — NO DNN in GA loop)

For each chromosome mask:
1. Select features: `X_selected = X[:, mask]`
2. Fit `StandardScaler` on GA-train only (no leakage)
3. Train `LogisticRegression` (max_iter=500, lbfgs, multinomial)
4. Return **macro F1-score** on GA-val as fitness

Penalty: if fewer than 5 features selected → fitness = 0.0

### 3.5 Feature Space

- Preprocessing: per-fold PCA on raw EEG (95% variance, fit on train only)
- Eye: NaN impute with training column means
- Fused: [EEG-PCA | Eye] concatenation — ~57 features per fold

---

## 4. Results

### 4.1 Per-Fold Results

| Fold | Baseline Acc | GA Acc | Delta | Features Selected |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 45.26% | 46.31% | +1.05% | 36/57 |
| 2 | 48.24% | 46.17% | -2.07% | 35/55 |
| 3 | 44.72% | 43.57% | -1.15% | 36/53 |
| 4 | 45.02% | 48.08% | +3.06% | 45/56 |
| 5 | 47.27% | 47.72% | +0.45% | 38/56 |


### 4.2 Summary

| Method | Accuracy | F1-Score | Avg Features |
|:---|:---:|:---:|:---:|
| **Baseline (all features)** | 46.10% ± 1.39% | 44.18% ± 2.77% | ~55 |
| **GA Feature Selection** | 46.37% ± 1.59% | 45.18% ± 3.11% | ~38 |
| **Delta** | +0.27% | +1.00% | — |

---

## 5. Key Insights

### Feature Stability Across Folds

| Stability | Count |
|---|---|
| Selected in **all 5 folds** | 9 features |
| **Never** selected | 1 features |
| Average selected per fold | 38.0 |

### Top 10 Most Frequently Selected Features

| Feature Index | Selection Frequency |
|---|---|
| 52 | 5/5 folds |
| 50 | 5/5 folds |
| 49 | 5/5 folds |
| 19 | 5/5 folds |
| 47 | 5/5 folds |
| 20 | 5/5 folds |
| 37 | 5/5 folds |
| 29 | 5/5 folds |
| 24 | 5/5 folds |
| 15 | 4/5 folds |

### Interpretation

Features selected in all 5 folds represent the most **cross-subject stable**
information in the fused space. Features never selected are likely redundant
or noise-dominant given the PCA-compressed EEG and raw Eye signals.

The GA's selection patterns across folds can guide future work:
- If EEG-PCA indices dominate the stable set → EEG carries more consistent signal
- If Eye feature indices dominate → Eye tracking is more cross-subject stable
- Mixed stable set → both modalities contribute, but only specific dimensions

---

## 6. Outputs

| File | Description |
|---|---|
| `ga_feature_selection.py` | Full GA pipeline code |
| `ga_results.csv` | Per-fold DNN test metrics |
| `ga_selected_features.csv` | Feature masks per fold + selection counts |
| `ga_vs_baseline.csv` | Side-by-side per-fold comparison |
| `plots/ga_features_per_fold.png` | Features selected per fold bar chart |
| `plots/ga_accuracy_comparison.png` | Baseline vs GA accuracy |
| `plots/ga_feature_heatmap.png` | Feature selection heatmap + frequency |
| `plots/ga_convergence.png` | GA convergence per fold |
