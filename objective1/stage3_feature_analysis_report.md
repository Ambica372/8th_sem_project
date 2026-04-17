# Stage 3 Feature Analysis Report
## Physiological Emotion Recognition Pipeline — SEED-IV Dataset

---

## A. Objective

Feature analysis is performed **before model training** to answer three critical questions:

1. **Do the features actually discriminate between emotions?** — If features show no statistical difference across emotion classes, training a model on them will not work regardless of model complexity.
2. **Are any features redundant?** — Highly correlated features waste model capacity and can cause instability in linear models.
3. **What structure does the data have?** — Dimensionality reduction reveals whether the four emotion classes form separable clusters, and whether subject-level variation dominates over emotion-level variation.

This analysis uses **only training data** from the 5 cross-validation folds (15,400 deduplicated windows from subjects 1–14, sessions 1–3).

---

## B. Statistical Testing

### What are ANOVA and Kruskal-Wallis?

**One-Way ANOVA (Analysis of Variance)** tests whether the mean of a feature differs significantly across the four emotion groups (neutral, sad, fear, happy). It assumes the data within each group is roughly normally distributed.

**Kruskal-Wallis** is the non-parametric equivalent of ANOVA. It makes no distributional assumption and instead compares the *rank distributions* of groups. It is more robust to outliers and skewed data.

Both tests output a **p-value**. A p-value < 0.05 means there is statistically significant evidence that the feature differs across emotions. Running **both tests together** provides cross-validation: a feature significant in both is a more reliable discriminator.

### Results Summary

| Modality | Total Features | ANOVA Significant | KW Significant | Both Significant |
|----------|---------------|-------------------|----------------|-----------------|
| EEG      | 310           | 296 / 310 (95.5%) | 303 / 310 (97.7%) | **291 / 310 (93.9%)** |
| Eye      | 31            | 31 / 31 (100%)    | 31 / 31 (100%) | **31 / 31 (100%)** |

> **Interpretation:** An overwhelming majority of features are statistically discriminative. The data is rich with emotion-relevant information. This strongly motivates using these features for classification.

### Per-Fold Statistics (EEG)

| Fold | ANOVA Significant | KW Significant |
|------|------------------|----------------|
| 1    | 299 / 310        | 305 / 310      |
| 2    | 307 / 310        | 309 / 310      |
| 3    | 293 / 310        | 304 / 310      |
| 4    | 304 / 310        | 306 / 310      |
| 5    | 296 / 310        | 308 / 310      |

The slight variation across folds confirms the splits are consistent — no fold is an outlier.

### Top 10 Most Discriminative EEG Features

These features showed p ≈ 0 on BOTH tests across ALL 5 folds:

| Feature  | Mapped To             | ANOVA p | KW p | Sig Folds |
|----------|-----------------------|---------|------|-----------|
| eeg_74   | Channel 14, theta     | ~0      | ~0   | 5/5       |
| eeg_119  | Channel 23, delta     | ~0      | ~0   | 5/5       |
| eeg_124  | Channel 24, delta     | ~0      | ~0   | 5/5       |
| eeg_79   | Channel 15, delta     | ~0      | ~0   | 5/5       |
| eeg_118  | Channel 23, theta (LDS) | ~0   | ~0   | 5/5       |
| eeg_84   | Channel 16, delta     | ~0      | ~0   | 5/5       |
| eeg_129  | Channel 25, delta     | ~0      | ~0   | 5/5       |
| eeg_73   | Channel 14, delta     | ~0      | ~0   | 5/5       |
| eeg_164  | Channel 32, delta     | ~0      | ~0   | 5/5       |
| eeg_108  | Channel 21, delta     | ~0      | ~0   | 5/5       |

> **Note:** Feature index mapping: `channel = feat_idx // 5`, `band = feat_idx % 5` (0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma). The top features cluster around **delta and theta bands**, and channels in the **frontal-temporal region** of the scalp.

### Top 10 Most Discriminative Eye Features

All 31 eye features were significant in every fold. The most discriminative (lowest p-value):

| Feature  | Interpretation              |
|----------|-----------------------------|
| eye_0    | Likely fixation duration mean |
| eye_1    | Likely fixation count / rate  |
| eye_4    | Likely blink duration / rate  |
| eye_5    | Related to eye_4 (r ≈ 0.9998) |
| eye_8    | Likely saccade amplitude      |
| eye_9    | Related to eye_8 (r ≈ 0.9997) |
| eye_10   | Pupil diameter mean           |
| eye_25   | Gaze event statistic          |
| eye_23   | Saccade velocity statistic    |
| eye_17   | Fixation spread / dispersion  |

> **Assumption:** The SEED-IV eye feature vector (31-D) is not explicitly documented per-column. The interpretations above are inferred from the known SEED-IV eye feature composition (blink, fixation, saccade, pupil per trial).

---

## C. Key Findings

### EEG

- **93.9% of EEG features are jointly significant** — the 310-dimensional DE feature space is highly informative.
- The most discriminative features are concentrated in the **delta (0–4 Hz)** and **theta (4–8 Hz)** bands, consistent with neuroscience literature showing these bands relate to emotional processing and frontal asymmetry.
- **Fear** stands out as the lowest-activation emotion across all bands and channels, making it the most distinguishable class.
- **Happy** shows the strongest gamma-band activation — consistent with high cognitive engagement during positive emotional states.
- **Only 19 features are non-significant** on average — these are likely near-zero variance features or channels with poor signal quality.

### Eye Movement

- **100% of eye features are significant** — every single eye movement feature carries emotion-relevant information.
- This is a stronger result than EEG and suggests **eye-tracking may be a highly competitive modality** for this task.
- Fixation and blink-related features dominate the top-10 list, consistent with the known relationship between emotional valence and visual attention patterns.

---

## D. Correlation Insights

### EEG Correlation Structure

| Metric | Value |
|--------|-------|
| Feature pairs with \|r\| > 0.90 | **474 pairs** |
| Highest correlation | eeg_1 ↔ eeg_6 (r = 0.987) |

**Pattern:** eeg_1 and eeg_6 are separated by exactly 5 indices (= 1 frequency band step). This confirms that **adjacent channels in the same frequency band are highly correlated** — this is expected because nearby scalp electrodes measure overlapping neural signals.

Clusters of highly correlated features appear at regular intervals corresponding to the 5-band structure. This has implications for Stage 4:
- **PCA or feature selection** can reduce the 310-D space significantly without losing information.
- Models prone to multicollinearity (e.g., linear regression) will benefit most from this reduction.

### Eye Correlation Structure

| Metric | Value |
|--------|-------|
| Feature pairs with \|r\| > 0.90 | **14 pairs** |
| Highest correlation | eye_4 ↔ eye_5 (r = 0.9998) |
| Second highest | eye_8 ↔ eye_9 (r = 0.9997) |

**Pattern:** Near-perfect correlations (r ≈ 1.0) between pairs (eye_4, eye_5) and (eye_8, eye_9) suggest these are **duplicate or derived measurements** (e.g., mean and median of the same quantity). In Stage 4, one of each pair can be dropped without information loss.

---

## E. Visualization Insights

### EEG PCA 2D — Colored by Emotion

- PC1 captures **55.7%** of variance, PC2 captures **7.1%** — the first component alone dominates.
- Emotion classes are **not cleanly separated in 2D PCA space** — the classes overlap, particularly neutral/sad/happy.
- However, some outlier points (especially along the PC1 axis) suggest **between-subject variance** pulling data away from the cluster.
- **Only 24 components** are needed to explain 95% of EEG variance — confirming significant redundancy in the 310-D space.

### EEG t-SNE 2D — Colored by Emotion vs. Subject

The t-SNE plot reveals a critical finding: the **data forms clusters primarily by subject**, not by emotion.
- Left panel (colored by emotion): clusters contain mixed emotion labels.
- Right panel (colored by subject): the same clusters are **nearly pure-subject** — each island corresponds to one subject.

> **Implication:** EEG signals are highly **subject-specific**. This motivates subject-level splitting (our Stage 2 design) and explains why cross-subject emotion recognition is a hard problem. A model must learn emotion-relevant features that generalize *across* individuals.

### EEG UMAP 2D — Strongest Evidence

UMAP reveals even clearer subject-level clustering than t-SNE. Each compact island corresponds to a single subject's data points, with multiple emotions mixed within each island. This confirms:
1. EEG manifold structure is dominated by **individual physiological fingerprints**.
2. Emotion separability exists but requires models that can factor out subject-specific variance.

### Eye UMAP 2D

In contrast to EEG, the Eye UMAP shows **diffuse, overlapping clusters** with no clear subject separation. This suggests eye-movement features are:
- More **subject-independent** than EEG features.
- More **emotion-agnostic** at a global level (mixed colours throughout the embedding).

This is a complex finding: statistically, all eye features are significant, but geometrically they do not form clean clusters — suggesting the emotion signal in eye data is **distributed** across features rather than concentrated in a few dominant dimensions.

---

## F. Neuroscientific Interpretation

### Frequency Band Analysis

| Band  | Frequency | Neutral | Sad  | Fear | Happy | Interpretation |
|-------|-----------|---------|------|------|-------|----------------|
| Delta | 0–4 Hz    | 0.270   | 0.176 | 0.081 | 0.234 | Deepest emotional processing; Fear markedly suppressed |
| Theta | 4–8 Hz    | 0.220   | 0.131 | 0.052 | 0.226 | Frontal theta linked to emotional regulation |
| Alpha | 8–13 Hz   | 0.247   | 0.202 | 0.033 | 0.259 | Alpha asymmetry is a classical emotion marker |
| Beta  | 13–30 Hz  | 0.136   | 0.170 | 0.002 | 0.262 | Beta increases with arousal; Happy is high-arousal |
| Gamma | 30–45 Hz  | 0.108   | 0.125 | 0.012 | 0.317 | Gamma peaks for Happy — cognitive engagement elevated |

**Key neuroscientific observations:**
- **Fear** shows the lowest activation across all bands. This may reflect a suppression of background EEG due to heightened vigilance and startle responses.
- **Happy** has the highest beta and gamma power, consistent with its classification as a high-arousal positive state — cognitive engagement amplifies high-frequency oscillations.
- **Alpha asymmetry** (different left/right frontal activity) is a well-validated EEG emotion marker. The alpha values here show emotion-dependent patterns consistent with this.
- **Neutral and Happy** are often confusable (similar delta, theta, alpha values) — this explains why classifiers may struggle to separate these two classes.

### Brain Region Analysis

The channel-wise plot shows emotion-dependent activation patterns across all 62 channels. Several channels (approximately 28–35 and 38–45) show pronounced separation between fear and the other emotions. These correspond approximately to the **temporal and parietal** scalp regions, which are known to be involved in:
- Emotional memory (amygdala connectivity reflected in temporal EEG)
- Attentional processing tied to emotional salience

---

## G. Conclusion

### Are Emotions Separable?

**Yes — statistically, strongly so.** 291/310 EEG features and 31/31 eye features show significant differences across emotion classes. The dataset contains substantial emotion-discriminative information.

**Geometrically, the separation is harder.** PCA and t-SNE show that subject-level variation strongly overlaps with emotion-level variation in EEG. Models will need either:
- Subject-independent feature extraction, or
- Cross-subject generalization training (as implemented in Stage 2's subject-level splits)

### Which Modality is Stronger?

| Property | EEG | Eye |
|----------|-----|-----|
| % features significant | 93.9% | **100%** |
| Redundancy (high-corr pairs) | 474 pairs | 14 pairs |
| PCA 95% components | 24 / 310 | **11 / 31** |
| Subject clustering (UMAP) | Very strong | Weak |
| Emotion clustering (UMAP) | Diffuse | Diffuse |

**Interpretation:**
- **Eye features** are more efficient: fewer features needed (31), less redundancy, 100% discrimination rate, and lower subject-specific bias.
- **EEG features** contain more raw signal but require more components and have higher redundancy. However, EEG's richness (310-D, temporal resolution) may capture emotion patterns not visible in eye movement alone.
- **A multimodal approach** (EEG + Eye) is expected to outperform either modality alone — which is the motivation for Stage 4.

### Recommendations for Stage 4 (Model Training)

1. **PCA preprocessing**: Reduce EEG from 310-D to 24-D (95% variance) for linear models.
2. **Feature selection**: The top 50 EEG features (all significant in ≥ 4/5 folds) form a compact, high-quality subset.
3. **Eye redundancy removal**: Drop eye_5 (≈ eye_4) and eye_9 (≈ eye_8) to eliminate near-duplicate features.
4. **Subject normalization challenge**: The strong subject-level clustering suggests adding subject adversarial training or domain adaptation may improve cross-subject performance.
5. **Focus on delta and theta bands**: Top EEG discriminators consistently fall in these bands — band-specific models may outperform full-spectrum models.

---

## Output Files Summary

| File | Contents |
|------|----------|
| `eeg_feature_significance.csv` | Per-feature ANOVA and KW p-values, significance flags, fold counts |
| `eye_feature_significance.csv` | Same for 31 eye features |
| `eeg_corr_matrix.csv` | 310×310 Pearson correlation matrix (EEG) |
| `eye_corr_matrix.csv` | 31×31 Pearson correlation matrix (Eye) |
| `eeg_high_corr_pairs.csv` | 474 EEG feature pairs with \|r\| > 0.90 |
| `eye_high_corr_pairs.csv` | 14 Eye feature pairs with \|r\| > 0.90 |
| `eeg_band_mean_per_emotion.csv` | Mean EEG value per frequency band per emotion |
| `plots/pca2d_eeg.png` | EEG PCA 2D scatter (emotion + subject coloring) |
| `plots/pca3d_eeg.png` | EEG PCA 3D scatter |
| `plots/tsne2d_eeg.png` | EEG t-SNE 2D scatter — reveals subject clustering |
| `plots/umap2d_eeg.png` | EEG UMAP 2D — strongest subject-cluster evidence |
| `plots/pca_scree_eeg.png` | EEG cumulative variance — 24 components for 95% |
| `plots/pca2d_eye.png` | Eye PCA 2D scatter |
| `plots/tsne2d_eye.png` | Eye t-SNE 2D |
| `plots/umap2d_eye.png` | Eye UMAP 2D — diffuse, less subject-structured |
| `plots/pca_scree_eye.png` | Eye cumulative variance — 11 components for 95% |
| `plots/corr_heatmap_eeg.png` | EEG correlation heatmap (every 5th feature) |
| `plots/corr_heatmap_eye.png` | Eye correlation heatmap (full 31×31) |
| `plots/eeg_band_channel_agg.png` | Band-wise bar + channel-wise line plot per emotion |
| `plots/eeg_heatmap_ch_band.png` | 62×5 heatmap per emotion class |
| `plots/significance_eeg.png` | Top-30 EEG features: fold count + -log10(p) |
| `plots/significance_eye.png` | All 31 Eye features: significance bars |
| `plots/eye_top10_distributions.png` | Histograms of top-10 Eye features per emotion |
| `plots/eye_boxplots.png` | Box plots of top-6 Eye features per emotion |
