# -*- coding: utf-8 -*-
"""
=============================================================================
Objective 3 Extension — Genetic Algorithm (GA) Feature Selection
=============================================================================

Hypothesis:
    Subject variability exists at the FEATURE level — some features are
    consistently noisy or modality-specific, and selecting stable subsets
    across subject groups may improve or stabilize performance.

Design:
    • Binary chromosome GA: each gene = 1 (keep) or 0 (drop) for a feature
    • Population = 20 chromosomes, 15 generations
    • Tournament selection (size=3), crossover (0.8), mutation (0.05)
    • Fitness proxy: Logistic Regression on a GA-train/val split
    • Final evaluation: DeepDNN (identical to Objective 2) on GA-selected features
    • Baseline: all features (same preprocessing, no selection)

Data source: objective1/eeg_features.csv, objective1/eye_features.csv
             (same CSVs as PSO corrected pipeline)
Preprocessing: per-fold PCA on EEG (95% var) + NaN impute on Eye + joint scale
Splitting    : GroupKFold(n_splits=5) — subject-wise, no leakage
Metrics      : Accuracy, Precision, Recall, F1-score (all macro)

Outputs (saved in objective3/):
    ga_results.csv              — per-fold DNN test metrics
    ga_selected_features.csv   — feature masks per fold + selection counts
    ga_vs_baseline.csv         — side-by-side comparison per fold
    plots/ga_features_per_fold.png
    plots/ga_accuracy_comparison.png
    plots/ga_feature_heatmap.png
    ga_report.md
=============================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

# Windows console UTF-8 fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OBJ3_DIR = os.path.dirname(os.path.abspath(__file__))
OBJ1_DIR = os.path.join(os.path.dirname(OBJ3_DIR), "objective1")
PLOT_DIR  = os.path.join(OBJ3_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EEG_CSV = os.path.join(OBJ1_DIR, "eeg_features.csv")
EYE_CSV = os.path.join(OBJ1_DIR, "eye_features.csv")

# ---------------------------------------------------------------------------
# Hyper-parameters — DNN identical to Objective 2
# ---------------------------------------------------------------------------
N_SPLITS     = 5
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 1e-4
VAL_FRAC     = 0.1
PCA_VAR      = 0.95
RANDOM_STATE = 42

META_COLS = ["subject_id", "session_id", "trial_id", "window_id",
             "emotion_label"]

# ---------------------------------------------------------------------------
# GA Hyper-parameters
# ---------------------------------------------------------------------------
GA_POP_SIZE     = 20
GA_N_GEN        = 15
GA_CROSSOVER    = 0.8
GA_MUTATION     = 0.05
GA_TOURN_SIZE   = 3
GA_SPLIT_FRAC   = 0.15   # GA internal val split from fold-train
MIN_FEATURES    = 5      # minimum features a chromosome must select

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# =============================================================================
# MODEL ARCHITECTURE — DeepDNN (UNCHANGED from Objective 2)
# =============================================================================

class DeepDNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3   = nn.Linear(128, 64)
        self.out   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.out(F.relu(self.fc3(x)))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load raw EEG + Eye CSVs (same source as PSO corrected pipeline)."""
    print("=" * 65)
    print(" [DATA] Loading from objective1/ CSVs")
    print("=" * 65)

    eeg_df = pd.read_csv(EEG_CSV)
    eye_df = pd.read_csv(EYE_CSV)

    assert len(eeg_df) == len(eye_df), \
        f"Row mismatch: EEG={len(eeg_df)}, Eye={len(eye_df)}"

    eeg_feat_cols = [c for c in eeg_df.columns if c not in META_COLS]
    eye_feat_cols = [c for c in eye_df.columns if c not in META_COLS]

    X_eeg = eeg_df[eeg_feat_cols].values.astype(np.float64)
    X_eye = eye_df[eye_feat_cols].values.astype(np.float64)
    y        = eeg_df["emotion_label"].values.astype(np.int64)
    subjects = eeg_df["subject_id"].values.astype(np.int64)

    # Remove corrupted rows (NaN/Inf in either modality)
    bad = (np.isnan(X_eeg).any(axis=1) | np.isinf(X_eeg).any(axis=1) |
           np.isnan(X_eye).any(axis=1) | np.isinf(X_eye).any(axis=1))
    X_eeg = X_eeg[~bad];  X_eye = X_eye[~bad]
    y     = y[~bad];       subjects = subjects[~bad]

    print(f"  Raw EEG : {X_eeg.shape}  Eye: {X_eye.shape}")
    print(f"  Removed {bad.sum()} bad rows → {len(y)} clean samples")
    u, c = np.unique(y, return_counts=True)
    print(f"  Classes : {dict(zip(u.tolist(), c.tolist()))}")
    sys.stdout.flush()
    return X_eeg, X_eye, y, subjects, eye_feat_cols


# =============================================================================
# PER-FOLD PREPROCESSING (identical to PSO corrected pipeline)
# =============================================================================

def preprocess_fold(X_eeg_tr, X_eye_tr, X_eeg_te, X_eye_te):
    """
    Per-fold: PCA on EEG (train only) + NaN impute Eye (train mean).
    Returns raw (unscaled) fused arrays + feature name info.
    Scaling happens inside fitness / final eval to avoid leakage.
    """
    # EEG: per-fold PCA (fit on train only)
    pca = PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)
    eeg_tr_pca = pca.fit_transform(X_eeg_tr)
    eeg_te_pca = pca.transform(X_eeg_te)
    n_eeg = eeg_tr_pca.shape[1]

    # Eye: NaN impute with training column means
    eye_mean = np.nanmean(X_eye_tr, axis=0)
    eye_mean = np.nan_to_num(eye_mean, nan=0.0)
    eye_tr_c = X_eye_tr.copy();  eye_te_c = X_eye_te.copy()
    for col in range(X_eye_tr.shape[1]):
        eye_tr_c[np.isnan(eye_tr_c[:, col]), col] = eye_mean[col]
        eye_te_c[np.isnan(eye_te_c[:, col]), col] = eye_mean[col]

    # Concatenate: [EEG-PCA | Eye]  (equal weights, full features)
    X_tr = np.hstack([eeg_tr_pca, eye_tr_c])
    X_te = np.hstack([eeg_te_pca, eye_te_c])

    return X_tr, X_te, n_eeg


def scale_and_select(X_tr, X_te, mask):
    """
    Apply boolean feature mask then fit StandardScaler on train only.
    Returns scaled (X_tr_sel, X_te_sel).
    """
    X_tr_s = X_tr[:, mask]
    X_te_s = X_te[:, mask]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_s)
    X_te_s = sc.transform(X_te_s)
    return X_tr_s, X_te_s


# =============================================================================
# GA FITNESS FUNCTION (Logistic Regression proxy — NO DNN in GA loop)
# =============================================================================

def ga_fitness(mask, X_ga_tr, X_ga_vl, y_ga_tr, y_ga_vl):
    """
    Evaluate a binary feature mask using Logistic Regression.

    Steps:
      1. If too few features selected → return penalty score
      2. Scale (fit on GA-train only — NO LEAKAGE)
      3. Train LogReg, get F1-macro on GA-val

    Returns: float fitness score (higher = better)
    """
    n_selected = mask.sum()
    if n_selected < MIN_FEATURES:
        return 0.0   # penalty for degenerate chromosomes

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_ga_tr[:, mask])
    Xvl = sc.transform(X_ga_vl[:, mask])

    lr = LogisticRegression(
        max_iter=500, solver="lbfgs", multi_class="multinomial",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    lr.fit(Xtr, y_ga_tr)
    preds = lr.predict(Xvl)
    return f1_score(y_ga_vl, preds, average="macro", zero_division=0)


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

class GeneticAlgorithm:
    """
    Binary GA for feature selection.

    Chromosome: binary vector of length n_features
        1 = keep feature, 0 = drop feature

    Operators:
        Selection  : Tournament (size=GA_TOURN_SIZE)
        Crossover  : Single-point (rate=GA_CROSSOVER)
        Mutation   : Bit-flip per gene (rate=GA_MUTATION per gene)

    Elitism: best chromosome always survives to next generation.
    """

    def __init__(self, n_features, pop_size=GA_POP_SIZE,
                 n_gen=GA_N_GEN, crossover=GA_CROSSOVER,
                 mutation=GA_MUTATION, tourn_size=GA_TOURN_SIZE,
                 seed=RANDOM_STATE):
        self.n_features  = n_features
        self.pop_size    = pop_size
        self.n_gen       = n_gen
        self.crossover   = crossover
        self.mutation    = mutation
        self.tourn_size  = tourn_size
        self.rng         = np.random.default_rng(seed)

    def _init_population(self):
        """Random binary population, each row is a chromosome."""
        pop = self.rng.integers(0, 2, (self.pop_size, self.n_features),
                                dtype=bool)
        # Guarantee at least MIN_FEATURES active in each chromosome
        for i in range(self.pop_size):
            if pop[i].sum() < MIN_FEATURES:
                idxs = self.rng.choice(self.n_features, MIN_FEATURES,
                                       replace=False)
                pop[i, idxs] = True
        return pop

    def _tournament_select(self, population, fitnesses):
        """Select one individual via tournament."""
        idxs = self.rng.choice(self.pop_size, self.tourn_size, replace=False)
        best = idxs[np.argmax(fitnesses[idxs])]
        return population[best].copy()

    def _crossover(self, parent1, parent2):
        """Single-point crossover. Returns two offspring."""
        if self.rng.random() < self.crossover:
            pt = self.rng.integers(1, self.n_features)
            c1 = np.concatenate([parent1[:pt], parent2[pt:]])
            c2 = np.concatenate([parent2[:pt], parent1[pt:]])
        else:
            c1, c2 = parent1.copy(), parent2.copy()
        return c1, c2

    def _mutate(self, chromosome):
        """Bit-flip mutation per gene with probability GA_MUTATION."""
        flip = self.rng.random(self.n_features) < self.mutation
        chromosome[flip] = ~chromosome[flip]
        # Repair: ensure at least MIN_FEATURES are selected
        if chromosome.sum() < MIN_FEATURES:
            off_idxs = np.where(~chromosome)[0]
            choose = self.rng.choice(
                off_idxs,
                min(MIN_FEATURES - chromosome.sum(), len(off_idxs)),
                replace=False
            )
            chromosome[choose] = True
        return chromosome

    def run(self, fitness_fn, fold_k, verbose=True):
        """
        Run GA for n_gen generations.

        Args:
            fitness_fn: callable(mask) → float
            fold_k    : fold index (for logging)
        Returns:
            best_mask (bool array), best_fitness, history (list of best per gen)
        """
        population = self._init_population()
        fitnesses  = np.array([fitness_fn(ind) for ind in population])

        best_idx     = np.argmax(fitnesses)
        best_mask    = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        history      = [float(best_fitness)]

        if verbose:
            print(f"    [GA  ] Gen  0/{self.n_gen}  "
                  f"best_f1={best_fitness:.4f}  "
                  f"n_feat={best_mask.sum()}")
            sys.stdout.flush()

        for gen in range(1, self.n_gen + 1):
            new_pop = []

            # Elitism: keep the best from previous generation
            new_pop.append(best_mask.copy())

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            population = np.array(new_pop[:self.pop_size])
            fitnesses  = np.array([fitness_fn(ind) for ind in population])

            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_mask    = population[gen_best_idx].copy()

            history.append(float(best_fitness))

            if verbose and (gen % 3 == 0 or gen == 1 or gen == self.n_gen):
                print(f"    [GA  ] Gen {gen:2d}/{self.n_gen}  "
                      f"best_f1={best_fitness:.4f}  "
                      f"n_feat={best_mask.sum()}")
                sys.stdout.flush()

        return best_mask, best_fitness, history


# =============================================================================
# DNN TRAINING & EVALUATION
# =============================================================================

def train_dnn(X_tr, y_tr, input_dim, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    model     = DeepDNN(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    Xt, Xv, yt, yv = train_test_split(
        X_tr, y_tr, test_size=VAL_FRAC,
        stratify=y_tr, random_state=seed
    )
    Xt_t = torch.FloatTensor(Xt);  yt_t = torch.LongTensor(yt)
    Xv_t = torch.FloatTensor(Xv);  yv_t = torch.LongTensor(yv)

    best_acc, best_state = 0.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(yt_t))
        for i in range(0, len(yt_t), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(Xt_t[idx]), yt_t[idx])
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va = accuracy_score(yv_t.numpy(),
                                torch.argmax(model(Xv_t), 1).numpy())
        if va > best_acc:
            best_acc   = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_dnn(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.FloatTensor(X_te)), 1).numpy()
    return (
        accuracy_score(y_te, preds),
        precision_score(y_te, preds, average="macro", zero_division=0),
        recall_score(y_te, preds, average="macro", zero_division=0),
        f1_score(y_te, preds, average="macro", zero_division=0),
    )


# =============================================================================
# BASELINE: all features (mirrors Obj2 DNN)
# =============================================================================

def run_baseline_fold(X_tr, X_te, y_tr, y_te):
    """All features + joint StandardScaler + DeepDNN."""
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    model = train_dnn(Xtr, y_tr, Xtr.shape[1])
    return eval_dnn(model, Xte, y_te)


# =============================================================================
# GA FOLD: feature selection + final DNN evaluation
# =============================================================================

def run_ga_fold(X_tr, X_te, y_tr, y_te, fold_k):
    """
    GA feature selection for one fold.

    Pipeline (NO LEAKAGE):
      1. Split fold-train into GA-train/GA-val (15%, stratified)
      2. Run GA (LogReg proxy fitness on GA-train/val)
      3. Apply best mask to FULL fold-train + fold-test
      4. Scale (fit on fold-train only)
      5. Train & evaluate DeepDNN on selected features

    Returns: (acc, prec, rec, f1, best_mask, n_selected, history)
    """
    n_features = X_tr.shape[1]

    # Step 1 — GA internal split (from raw preprocessed train data)
    (X_ga_tr, X_ga_vl,
     y_ga_tr, y_ga_vl) = train_test_split(
        X_tr, y_tr,
        test_size=GA_SPLIT_FRAC, stratify=y_tr,
        random_state=RANDOM_STATE
    )

    # Step 2 — GA with LogReg proxy
    def fitness_fn(mask):
        return ga_fitness(mask, X_ga_tr, X_ga_vl, y_ga_tr, y_ga_vl)

    print(f"\n  [Fold {fold_k}] Running GA "
          f"(pop={GA_POP_SIZE}, gen={GA_N_GEN}, "
          f"cr={GA_CROSSOVER}, mr={GA_MUTATION})...")
    sys.stdout.flush()

    ga = GeneticAlgorithm(n_features=n_features, seed=RANDOM_STATE + fold_k)
    best_mask, best_fit, history = ga.run(fitness_fn, fold_k)
    n_sel = int(best_mask.sum())

    print(f"  [Fold {fold_k}] GA done: "
          f"{n_sel}/{n_features} features selected  "
          f"proxy_f1={best_fit:.4f}")
    sys.stdout.flush()

    # Steps 3-5 — apply mask to full fold data, scale, train DNN
    X_tr_sel, X_te_sel = scale_and_select(X_tr, X_te, best_mask)

    print(f"  [Fold {fold_k}] Training DNN "
          f"({EPOCHS} epochs, dim={X_tr_sel.shape[1]})...")
    sys.stdout.flush()
    model = train_dnn(X_tr_sel, y_tr, X_tr_sel.shape[1])
    acc, prec, rec, f1 = eval_dnn(model, X_te_sel, y_te)

    print(f"  [Fold {fold_k}] DNN TEST: acc={acc:.4f}  "
          f"prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    sys.stdout.flush()

    return acc, prec, rec, f1, best_mask, n_sel, history


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_features_per_fold(fold_n_features, total_features, out_dir):
    """Bar chart: how many features GA selected per fold vs total."""
    folds = [f"Fold {k}" for k in range(1, len(fold_n_features) + 1)]
    sel   = fold_n_features
    kept  = [total_features - s for s in sel]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(folds))
    ax.bar(x, sel,  label="Selected", color="#55A868", alpha=0.85)
    ax.bar(x, kept, bottom=sel, label="Dropped", color="#C44E52", alpha=0.5)
    ax.axhline(total_features, ls="--", color="gray", lw=1,
               label=f"Total ({total_features})")
    ax.set_xticks(x); ax.set_xticklabels(folds)
    ax.set_ylabel("Number of Features")
    ax.set_title("GA: Features Selected per Fold", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    for i, s in enumerate(sel):
        ax.text(i, s / 2, str(s), ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ga_features_per_fold.png"), dpi=150)
    plt.close()
    print("  Saved: plots/ga_features_per_fold.png")


def plot_accuracy_comparison(baseline_df, ga_df, out_dir):
    """Per-fold + mean accuracy: Baseline vs GA."""
    ba = baseline_df["accuracy"].values * 100
    ga = ga_df["accuracy"].values * 100
    x  = np.arange(N_SPLITS)
    bw = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Objective 3 Extension — GA Feature Selection",
                 fontsize=14, fontweight="bold")

    b1 = axes[0].bar(x - bw/2, ba, bw, label="Baseline (all features)",
                     color="#4C72B0", alpha=0.85)
    b2 = axes[0].bar(x + bw/2, ga, bw, label="GA Selected features",
                     color="#55A868", alpha=0.85)
    axes[0].axhline(25, ls="--", color="gray", lw=1, label="Chance (25%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Fold {i+1}" for i in x])
    axes[0].set_xlabel("Fold"); axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Per-Fold Accuracy"); axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 75)
    for bar in b1:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for bar in b2:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}", ha="center", fontsize=8)

    means = [ba.mean(), ga.mean()]
    stds  = [ba.std(),  ga.std()]
    axes[1].bar(["Baseline\n(all feats)", "GA Selection\n(subset)"],
                means, color=["#4C72B0", "#55A868"], alpha=0.85,
                yerr=stds, capsize=8, error_kw={"lw": 2})
    axes[1].axhline(25, ls="--", color="gray", lw=1)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m + s + 1.0, f"{m:.1f}%", ha="center",
                     fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Mean Test Accuracy (%)")
    axes[1].set_title("Overall Mean ± Std"); axes[1].set_ylim(0, 75)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ga_accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: plots/ga_accuracy_comparison.png")


def plot_feature_heatmap(masks, total_features, out_dir):
    """
    Heatmap of feature selection across folds.
    Row = fold, Col = feature index. Value = selected (1) or not (0).
    Also shows selection frequency per feature.
    """
    mask_matrix = np.array([m.astype(int) for m in masks])  # (5, n_feat)
    freq        = mask_matrix.mean(axis=0)                   # (n_feat,)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("GA Feature Selection Heatmap", fontsize=13,
                 fontweight="bold")

    sns.heatmap(mask_matrix, ax=axes[0], cmap="YlGn",
                linewidths=0.3, linecolor="gray",
                xticklabels=False, yticklabels=[f"Fold {i+1}" for i in range(N_SPLITS)],
                cbar_kws={"label": "Selected (1=yes)"})
    axes[0].set_xlabel("Feature Index")
    axes[0].set_title("Per-Fold Feature Selection (green = selected)")

    axes[1].bar(np.arange(total_features), freq,
                color="#55A868", alpha=0.8, width=1.0)
    axes[1].axhline(0.6, ls="--", color="red", lw=1,
                    label="60% threshold (stable)")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Selection\nFrequency")
    axes[1].set_title("Feature Selection Frequency Across Folds")
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(-0.5, total_features - 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ga_feature_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: plots/ga_feature_heatmap.png")


def plot_ga_convergence(histories, out_dir):
    """GA convergence: best fitness per generation per fold."""
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, hist in enumerate(histories, 1):
        ax.plot(range(len(hist)), [h * 100 for h in hist],
                marker="o", ms=3, lw=1.8,
                color=colors[k - 1], label=f"Fold {k}")
    ax.set_xlabel("Generation"); ax.set_ylabel("Best Proxy F1 (%)")
    ax.set_title("GA Convergence per Fold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ga_convergence.png"), dpi=150)
    plt.close()
    print("  Saved: plots/ga_convergence.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()

    print("\n" + "=" * 65)
    print("  OBJECTIVE 3 EXTENSION — GA FEATURE SELECTION")
    print("  Fitness proxy: Logistic Regression (no DNN in GA loop)")
    print("  Final eval   : DeepDNN (same as Objective 2)")
    print("=" * 65)
    print(f"  GA: pop={GA_POP_SIZE}  gen={GA_N_GEN}  "
          f"cr={GA_CROSSOVER}  mr={GA_MUTATION}  tourn={GA_TOURN_SIZE}")
    print(f"  DNN: epochs={EPOCHS}  lr={LR}  batch={BATCH_SIZE}")
    print()
    sys.stdout.flush()

    # 1. Load raw data
    X_eeg_raw, X_eye_raw, y, subjects, eye_cols = load_data()

    # 2. GroupKFold
    gkf = GroupKFold(n_splits=N_SPLITS)

    baseline_rows = []
    ga_rows       = []
    mask_rows     = []
    histories     = []
    all_masks     = []
    total_feats   = None

    # 3. Cross-validation loop
    for fold_k, (tr_idx, te_idx) in enumerate(
            gkf.split(X_eeg_raw, y, groups=subjects), start=1):

        tr_subj = np.unique(subjects[tr_idx])
        te_subj = np.unique(subjects[te_idx])
        overlap = set(tr_subj.tolist()) & set(te_subj.tolist())
        assert len(overlap) == 0, f"LEAKAGE fold {fold_k}: {overlap}"

        print(f"\n{'='*65}")
        print(f"  FOLD {fold_k}/{N_SPLITS}")
        print(f"  Train subjects: {sorted(tr_subj.tolist())}")
        print(f"  Test  subjects: {sorted(te_subj.tolist())}")
        print(f"  Train: {tr_idx.sum()} samples  |  Test: {te_idx.sum()} samples")
        sys.stdout.flush()

        Xe_tr = X_eeg_raw[tr_idx];  Xe_te = X_eeg_raw[te_idx]
        Ey_tr = X_eye_raw[tr_idx];  Ey_te = X_eye_raw[te_idx]
        y_tr  = y[tr_idx];          y_te  = y[te_idx]

        # Per-fold preprocessing (PCA + impute) — NO leakage
        X_tr, X_te, n_eeg = preprocess_fold(Xe_tr, Ey_tr, Xe_te, Ey_te)
        if total_feats is None:
            total_feats = X_tr.shape[1]
        print(f"  PCA → {n_eeg} EEG components + {X_tr.shape[1]-n_eeg} "
              f"Eye features = {X_tr.shape[1]} total")
        sys.stdout.flush()

        # --- BASELINE (all features) ---
        print(f"\n  [Fold {fold_k}] BASELINE (all {X_tr.shape[1]} features + DNN)...")
        sys.stdout.flush()
        b_acc, b_prec, b_rec, b_f1 = run_baseline_fold(X_tr, X_te, y_tr, y_te)
        print(f"  [Fold {fold_k}] BASELINE: acc={b_acc:.4f}  "
              f"prec={b_prec:.4f}  rec={b_rec:.4f}  f1={b_f1:.4f}")
        sys.stdout.flush()
        baseline_rows.append({
            "fold": fold_k, "method": "Baseline",
            "accuracy": round(b_acc, 4), "precision": round(b_prec, 4),
            "recall": round(b_rec, 4), "f1": round(b_f1, 4),
            "n_features": X_tr.shape[1]
        })

        # --- GA FEATURE SELECTION ---
        g_acc, g_prec, g_rec, g_f1, best_mask, n_sel, hist = run_ga_fold(
            X_tr, X_te, y_tr, y_te, fold_k
        )
        ga_rows.append({
            "fold": fold_k, "method": "GA Selection",
            "accuracy": round(g_acc, 4), "precision": round(g_prec, 4),
            "recall": round(g_rec, 4), "f1": round(g_f1, 4),
            "n_features": n_sel
        })
        all_masks.append(best_mask)
        histories.append(hist)

        # Record which features were selected
        mask_row = {"fold": fold_k, "n_selected": n_sel,
                    "total_features": X_tr.shape[1],
                    "selection_rate": round(n_sel / X_tr.shape[1], 4)}
        for fi, val in enumerate(best_mask):
            mask_row[f"feat_{fi}"] = int(val)
        mask_rows.append(mask_row)

    # 4. DataFrames
    baseline_df = pd.DataFrame(baseline_rows)
    ga_df       = pd.DataFrame(ga_rows)
    masks_df    = pd.DataFrame(mask_rows)

    # Align mask sizes (folds may differ by 1-2 due to variable PCA dims)
    min_feats = min(m.shape[0] for m in all_masks)
    trimmed   = [m[:min_feats] for m in all_masks]

    # 5. Save CSVs
    ga_df.to_csv(os.path.join(OBJ3_DIR, "ga_results.csv"), index=False)
    masks_df.to_csv(os.path.join(OBJ3_DIR, "ga_selected_features.csv"),
                    index=False)
    pd.concat([baseline_df, ga_df], ignore_index=True).to_csv(
        os.path.join(OBJ3_DIR, "ga_vs_baseline.csv"), index=False)

    print(f"\n  Saved: ga_results.csv")
    print(f"  Saved: ga_selected_features.csv")
    print(f"  Saved: ga_vs_baseline.csv")

    # 6. Feature selection frequency
    sel_counts = np.array([m.astype(int) for m in trimmed]).sum(axis=0)
    stable_feats = (sel_counts == N_SPLITS).sum()
    unstable_feats = (sel_counts == 0).sum()
    print(f"\n  Feature stability:")
    print(f"    Selected in ALL 5 folds : {stable_feats} features")
    print(f"    Never selected          : {unstable_feats} features")
    print(f"    Avg selected/fold       : "
          f"{np.mean([r['n_selected'] for r in mask_rows]):.1f}")

    # 7. Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Method':<22} {'Acc (mean +/- std)':<24} {'F1 (mean +/- std)'}")
    print(f"  {'-'*62}")
    for lbl, df in [("Baseline (all)", baseline_df),
                    ("GA Selection",   ga_df)]:
        accs = df["accuracy"].values
        f1s  = df["f1"].values
        nf   = df["n_features"].values
        print(f"  {lbl:<22} "
              f"{accs.mean()*100:>6.2f}% +/- {accs.std()*100:>4.2f}%    "
              f"{f1s.mean()*100:>6.2f}% +/- {f1s.std()*100:>4.2f}%  "
              f"  (~{nf.mean():.0f} feats)")

    delta_acc = (ga_df["accuracy"].mean() - baseline_df["accuracy"].mean()) * 100
    delta_f1  = (ga_df["f1"].mean()       - baseline_df["f1"].mean())       * 100
    print(f"\n  Delta Accuracy : {delta_acc:+.2f}%")
    print(f"  Delta F1       : {delta_f1:+.2f}%")

    # Sanity checks
    bl_accs = baseline_df["accuracy"].values
    ga_accs = ga_df["accuracy"].values
    masks_differ = not all(
        np.array_equal(all_masks[0][:min_feats],
                       all_masks[i][:min_feats])
        for i in range(1, len(all_masks))
    )
    print(f"\n  [Sanity] Baseline matches ~46% : "
          f"{bl_accs.mean()*100:.2f}%  "
          f"({'OK' if 0.42 < bl_accs.mean() < 0.52 else 'CHECK'})")
    print(f"  [Sanity] Feature masks differ : "
          f"{'OK — vary per fold' if masks_differ else 'WARN — all same'}")
    print(f"  [Sanity] No subject overlap   : OK (enforced by GroupKFold)")

    # 8. Plots
    print("\n  [Plots]")
    fold_n_features = [r["n_selected"] for r in mask_rows]
    plot_features_per_fold(fold_n_features, total_feats, PLOT_DIR)
    plot_accuracy_comparison(baseline_df, ga_df, PLOT_DIR)
    plot_feature_heatmap(trimmed, min_feats, PLOT_DIR)
    plot_ga_convergence(histories, PLOT_DIR)

    # 9. Auto-generate ga_report.md
    write_ga_report(baseline_df, ga_df, masks_df, trimmed,
                    stable_feats, unstable_feats, delta_acc, delta_f1,
                    elapsed)

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\n" + "=" * 65)
    print("  GA FEATURE SELECTION COMPLETE — NO SUBJECT LEAKAGE")
    print(f"  All outputs: {OBJ3_DIR}")
    print("=" * 65)
    sys.stdout.flush()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def write_ga_report(baseline_df, ga_df, masks_df, trimmed,
                    stable_feats, unstable_feats, delta_acc, delta_f1,
                    elapsed):
    """Auto-generate ga_report.md from actual results."""

    bl_accs = baseline_df["accuracy"].values
    ga_accs = ga_df["accuracy"].values
    bl_f1s  = baseline_df["f1"].values
    ga_f1s  = ga_df["f1"].values

    per_fold_rows = ""
    for _, br in baseline_df.iterrows():
        gr = ga_df[ga_df["fold"] == br["fold"]].iloc[0]
        mr = masks_df[masks_df["fold"] == br["fold"]].iloc[0]
        delta = (gr["accuracy"] - br["accuracy"]) * 100
        per_fold_rows += (
            f"| {int(br['fold'])} | {br['accuracy']*100:.2f}% | "
            f"{gr['accuracy']*100:.2f}% | {delta:+.2f}% | "
            f"{int(mr['n_selected'])}/{int(mr['total_features'])} |\n"
        )

    min_feats = min(m.shape[0] for m in trimmed)
    sel_counts = np.array([m.astype(int) for m in trimmed]).sum(axis=0)
    top_feats = np.argsort(sel_counts)[::-1][:10].tolist()
    top_freq  = sel_counts[top_feats].tolist()
    top_table = "\n".join(
        f"| {fi} | {fr}/5 folds |"
        for fi, fr in zip(top_feats, top_freq)
    )

    report = f"""# Objective 3 Extension: GA-Based Feature Selection

> **Run date:** April 2026  
> **Runtime:** {elapsed/60:.1f} minutes  
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
there are $2^{{57}}$ possible subsets — exhaustive search is impossible.

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
| Population size | {GA_POP_SIZE} |
| Generations | {GA_N_GEN} |
| Crossover rate | {GA_CROSSOVER} |
| Mutation rate (per gene) | {GA_MUTATION} |
| Tournament size | {GA_TOURN_SIZE} |
| Minimum features | {MIN_FEATURES} |
| Elitism | Yes (best always survives) |

### 3.2 Chromosome (Individual)

A binary vector of length = total fused features.
`1` = include feature, `0` = drop feature.

### 3.3 Genetic Operators

- **Selection:** Tournament selection (size 3) — probabilistic, preserves diversity
- **Crossover:** Single-point crossover — combines two parents' feature subsets
- **Mutation:** Per-gene bit-flip with probability {GA_MUTATION} — introduces exploration
- **Elitism:** Best individual from each generation always carries over — prevents regression

### 3.4 Fitness Function (Fast Proxy — NO DNN in GA loop)

For each chromosome mask:
1. Select features: `X_selected = X[:, mask]`
2. Fit `StandardScaler` on GA-train only (no leakage)
3. Train `LogisticRegression` (max_iter=500, lbfgs, multinomial)
4. Return **macro F1-score** on GA-val as fitness

Penalty: if fewer than {MIN_FEATURES} features selected → fitness = 0.0

### 3.5 Feature Space

- Preprocessing: per-fold PCA on raw EEG (95% variance, fit on train only)
- Eye: NaN impute with training column means
- Fused: [EEG-PCA | Eye] concatenation — ~57 features per fold

---

## 4. Results

### 4.1 Per-Fold Results

| Fold | Baseline Acc | GA Acc | Delta | Features Selected |
|:---:|:---:|:---:|:---:|:---:|
{per_fold_rows}

### 4.2 Summary

| Method | Accuracy | F1-Score | Avg Features |
|:---|:---:|:---:|:---:|
| **Baseline (all features)** | {bl_accs.mean()*100:.2f}% ± {bl_accs.std()*100:.2f}% | {bl_f1s.mean()*100:.2f}% ± {bl_f1s.std()*100:.2f}% | ~{baseline_df['n_features'].mean():.0f} |
| **GA Feature Selection** | {ga_accs.mean()*100:.2f}% ± {ga_accs.std()*100:.2f}% | {ga_f1s.mean()*100:.2f}% ± {ga_f1s.std()*100:.2f}% | ~{ga_df['n_features'].mean():.0f} |
| **Delta** | {delta_acc:+.2f}% | {delta_f1:+.2f}% | — |

---

## 5. Key Insights

### Feature Stability Across Folds

| Stability | Count |
|---|---|
| Selected in **all 5 folds** | {stable_feats} features |
| **Never** selected | {unstable_feats} features |
| Average selected per fold | {ga_df['n_features'].mean():.1f} |

### Top 10 Most Frequently Selected Features

| Feature Index | Selection Frequency |
|---|---|
{top_table}

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
"""

    report_path = os.path.join(OBJ3_DIR, "ga_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: ga_report.md")


if __name__ == "__main__":
    main()
