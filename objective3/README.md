# Explainable AI for Multimodal Emotion Recognition

**Objective 3 — Post-hoc Feature Attribution Pipeline**  
Provides model-agnostic and gradient-based explanations for all five deep learning models trained in Objective 2, revealing which physiological features drive predictions and how EEG vs. eye-tracking modalities contribute to emotion classification.

---

## Overview

Understanding *why* a model predicts a given emotion class is as important as prediction accuracy, especially in affective computing where physiological signals carry subject-specific noise. This pipeline implements three complementary XAI methods applied to the same test subset used in Objective 2 (Fold 1, Subject 0 held out):

| Goal | Method Used |
|---|---|
| Global feature importance | SHAP summary plots |
| Input-output gradient attribution | Gradient saliency maps |
| Internal attention analysis | Attention weight visualisation |
| Modality contribution | EEG vs. Eye mean SHAP importance |

**No models are retrained.** All model checkpoints (`best_fold1.pth`) are loaded directly from `objective2-final/results/groupkfold_laso/`.

---

## XAI Methods

### A — SHAP (SHapley Additive exPlanations)

SHAP assigns each input feature a contribution score consistent with game-theoretic Shapley values, satisfying **local accuracy**, **missingness**, and **consistency** axioms.

- **MLP / DNN / Attention / Hybrid**: `shap.GradientExplainer`
  - Efficient gradient-based approximation for PyTorch models
  - Background dataset: 100 training samples
  - Test dataset: 100 held-out samples (subject 0)

- **Decision Fusion**: `shap.KernelExplainer`
  - Model-agnostic SHAP via perturbation sampling
  - Required because Decision Fusion has two separate input streams; a single-input numpy wrapper is used: `output = model(x[:, :29], x[:, 29:])`
  - Background: 100 samples, nsamples=100

**Outputs**: Beeswarm summary plots showing feature impact distribution across all 4 emotion classes.

### B — Gradient Saliency

Computes the absolute gradient of the total model output with respect to the input tensor:

```
saliency[i] = |∂(sum of outputs) / ∂ input[i]|
```

Averaged across all test samples to obtain a global feature importance ranking. This method is fast, model-agnostic in principle, and directly reflects how sensitively the model responds to each input dimension.

**Outputs**: Horizontal bar charts showing the top-20 most salient features per model.

### C — Attention Weight Visualisation

Applicable only to the **Attention** and **Hybrid** models, which contain explicit sigmoid gating layers.

- **Attention model**: Extracts `sigmoid(Linear(input))` weights averaged across the test batch — shows per-feature input-space importance
- **Hybrid model**: Extracts learned weight matrix of the hidden attention layer, reshaped to an 8×16 heatmap

**Outputs**: Bar plots and heatmaps of normalised attention weights.

---

## Models Explained

| Model | SHAP Method | Gradient | Attention Map |
|---|---|---|---|
| MLP | GradientExplainer | ✅ | ✗ |
| DNN | GradientExplainer | ✅ | ✗ |
| Attention | GradientExplainer | ✅ | ✅ |
| Hybrid | GradientExplainer | ✅ | ✅ |
| Decision Fusion | KernelExplainer | ✅ | ✗ |

---

## Output Structure

```
objective3/
│
├── shap/
│   ├── mlp_shap_summary.png               # SHAP beeswarm — MLP
│   ├── dnn_shap_summary.png               # SHAP beeswarm — DNN
│   ├── attention_shap_summary.png         # SHAP beeswarm — Attention
│   ├── hybrid_shap_summary.png            # SHAP beeswarm — Hybrid
│   └── decision_fusion_shap_summary.png   # SHAP beeswarm — Decision Fusion
│
├── gradients/
│   ├── mlp_gradients.png                  # Top-20 salient features — MLP
│   ├── dnn_gradients.png                  # Top-20 salient features — DNN
│   ├── attention_gradients.png            # Top-20 salient features — Attention
│   ├── hybrid_gradients.png               # Top-20 salient features — Hybrid
│   └── decision_fusion_gradients.png      # Top-20 salient features — Decision Fusion
│
├── attention_maps/
│   ├── attention_attention_weights.png    # Input attention weights (Attention model)
│   └── hybrid_attention_weights.png       # Hidden attention heatmap (Hybrid model)
│
└── reports/
    └── xai_summary.txt                    # Modality impact quantification per model
```

---

## Key Findings

### 1. Eye Features Dominate in Most Models

Across MLP, Attention, and Hybrid models, **Eye tracking features (Eye_1–Eye_29) show systematically higher SHAP importance** than EEG features:

| Model | EEG Mean Importance | Eye Mean Importance | Dominant Modality |
|---|---|---|---|
| MLP | 0.0453 | 0.0645 | **Eye** |
| DNN | 0.0228 | 0.0211 | **EEG** *(marginal)* |
| Attention | 0.0436 | 0.0597 | **Eye** |
| Hybrid | 0.0442 | 0.0580 | **Eye** |
| Decision Fusion | — | — | Separate streams |

### 2. DNN Is the Exception — Relies More on EEG

The DNN shows a marginal but consistent preference for EEG features (0.0228 vs. 0.0211). This suggests that deeper stacking of linear layers with batch normalisation is more capable of extracting discriminative signal from the noisier EEG space.

### 3. Attention Mechanisms Successfully Gate Relevant Features

The Attention model's learned input-space weight map shows **non-uniform feature attention**, with elevated weights concentrated on a subset of Eye and EEG features rather than distributing weight evenly. This confirms that the sigmoid gating is acting as a meaningful filter, not collapsing to uniform weights.

### 4. Decision Fusion Maintains Modality Independence

Because Decision Fusion processes EEG and Eye features through entirely separate network branches, SHAP values from KernelExplainer reveal the **independent contribution of each modality** without entanglement. This makes it the most interpretable architecture for modality attribution.

### 5. High-Importance Features Across Models

Top-ranked features (by SHAP and gradient saliency) consistently include:
- **Eye**: Gaze dispersion, fixation duration, pupil diameter changes — correlated with arousal and valence
- **EEG**: Beta and gamma band features — associated with cognitive load and emotional arousal

---

## Interpretation: Why Eye Features Dominate

### Eye Signals Are More Stable Across Subjects

Eye-tracking features such as fixation patterns, blink rate, and pupil dilation are **robustly correlated with emotional state** across individuals. Unlike EEG, eye signals are less susceptible to:
- Electrode placement variability
- Individual alpha rhythm differences
- Muscular artefacts and channel drift

In a Leave-One-Subject-Out (LASO) evaluation, the model must generalise to an unseen person. Eye features offer more **transferable** discriminative patterns.

### EEG Is Rich But Noisy Across Subjects

EEG signals are highly individualised. Spectral patterns (e.g., theta asymmetry, gamma coherence) can encode emotion effectively *within* a subject, but differ substantially *across* subjects. In a cross-subject LASO setting, the model must rely on inter-individually consistent patterns, which are weaker in EEG and more salient in eye signals.

This explains:
- Why most models learn to emphasise Eye features under LASO conditions
- Why DNN — the least biased by attention gating — still finds *some* useful EEG signal
- Why Decision Fusion achieves the best cross-subject accuracy: it lets each modality contribute independently rather than competing in a shared feature space

---

## How to Run

### Prerequisites

```bash
pip install shap
```

SHAP 0.51.0+ is required. All other dependencies are shared with Objective 2.

### Execute XAI Pipeline

```bash
python objective3_xai_pipeline.py
```

The script will:
1. Auto-load and clean the data from `stage4_pipeline/processed_data/`
2. Reconstruct the Fold 1 train/test split (same as Objective 2)
3. Apply per-fold StandardScaler (fit on train, transform test)
4. Load model checkpoints from `objective2-final/results/groupkfold_laso/`
5. Compute SHAP, gradient saliency, and attention maps
6. Save all plots and the summary report to `objective3/`

> **Estimated runtime**: ~5–8 minutes (KernelExplainer for Decision Fusion is the bottleneck).

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Test-set only | XAI applied only to held-out test data — training data not used for attribution |
| Fold 1 as reference | Consistent with model checkpoint availability; representative of LASO performance |
| Background = 100 train samples | Sufficient for stable SHAP estimates without prohibitive computation |
| Test subset = 100 samples | Balances explanation quality with runtime |
| GradientExplainer for PyTorch | Native gradient integration; more efficient than KernelExplainer for differentiable models |
| KernelExplainer for Decision Fusion | Required due to multi-input architecture incompatibility with GradientExplainer wrapper |

---

## Dependencies

```
torch >= 2.0
shap >= 0.51.0
scikit-learn >= 1.3
numpy, pandas, matplotlib, seaborn
```
