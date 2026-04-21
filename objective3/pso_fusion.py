# -*- coding: utf-8 -*-
"""
=============================================================================
Objective 3 — PSO-Based Adaptive Fusion (CORRECTED v2)
=============================================================================

Root cause of v1 degradation (35% baseline vs Obj2's 46%):
  1. WRONG FUSION: element-wise sum (w1*EEG + w2*Eye → 29D) is
     mathematically invalid — EEG and Eye feature dimensions have no
     semantic alignment. This discards 50% of information.
  2. WRONG DATA SOURCE: X_eeg_pca.npy was produced by a GLOBAL PCA
     applied once across all subjects before cross-validation. This
     violates the per-fold PCA protocol Obj2 used, which retains 95%
     variance per fold and avoids leakage in PCA fitting.

CORRECTED DESIGN:
  • Load raw features from the same CSVs Objective 2 uses
  • Run per-fold PCA on EEG (fit on train only, 95% variance) — identical
    to Objective 2's preprocess_fold()
  • Eye NaN imputation with training column means (no leakage)
  • PSO searches (w1, w2) using WEIGHTED CONCATENATION:
        X_fused = [w1 * X_eeg_pca  |  w2 * X_eye]   → variable-D
        (preserves ALL features from both modalities)
  • Baseline: w1 = w2 = 0.5 → standard concatenation ≡ Objective 2
  • Final model: DeepDNN (UNCHANGED architecture from Objective 2)

Data source: objective1/eeg_features.csv, objective1/eye_features.csv
Splitting  : GroupKFold(n_splits=5) — identical to Objective 2
Metrics    : Accuracy, Precision, Recall, F1 (macro)
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
# Paths  (reads from same CSVs as Objective 2 — guaranteed equivalence)
# ---------------------------------------------------------------------------
OBJ3_DIR = os.path.dirname(os.path.abspath(__file__))
OBJ2_DIR = os.path.join(os.path.dirname(OBJ3_DIR), "objective2")
OBJ1_DIR = os.path.join(os.path.dirname(OBJ3_DIR), "objective1")
PLOT_DIR = os.path.join(OBJ3_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

EEG_CSV = os.path.join(OBJ1_DIR, "eeg_features.csv")
EYE_CSV = os.path.join(OBJ1_DIR, "eye_features.csv")

# ---------------------------------------------------------------------------
# Hyper-parameters — IDENTICAL to Objective 2
# ---------------------------------------------------------------------------
N_SPLITS     = 5
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 1e-4
VAL_FRAC     = 0.1
PCA_VAR      = 0.95          # retain 95% variance — same as Obj2
RANDOM_STATE = 42

META_COLS = ["subject_id", "session_id", "trial_id", "window_id",
             "emotion_label"]

# ---------------------------------------------------------------------------
# PSO Parameters
# ---------------------------------------------------------------------------
PSO_N_PARTICLES  = 10
PSO_N_ITERATIONS = 10
PSO_INERTIA      = 0.7
PSO_C1           = 1.5
PSO_C2           = 1.5

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
# DATA LOADING — from CSVs (same as Objective 2)
# =============================================================================

def load_data():
    """
    Load raw EEG and Eye features from Objective 1 CSVs.
    This mirrors Objective 2's load_data() exactly:
      - preserves subject_id
      - removes NaN/Inf rows consistently
      - returns aligned raw feature arrays for per-fold preprocessing
    """
    print("=" * 65)
    print(" [DATA] Loading from objective1/ CSVs")
    print("=" * 65)

    eeg_df = pd.read_csv(EEG_CSV)
    eye_df = pd.read_csv(EYE_CSV)

    assert len(eeg_df) == len(eye_df), \
        f"Row count mismatch: EEG={len(eeg_df)}, Eye={len(eye_df)}"

    eeg_feat_cols = [c for c in eeg_df.columns if c not in META_COLS]
    eye_feat_cols = [c for c in eye_df.columns if c not in META_COLS]

    X_eeg_raw = eeg_df[eeg_feat_cols].values.astype(np.float64)
    X_eye_raw = eye_df[eye_feat_cols].values.astype(np.float64)
    y         = eeg_df["emotion_label"].values.astype(np.int64)
    subjects  = eeg_df["subject_id"].values.astype(np.int64)

    print(f"  Raw EEG : {X_eeg_raw.shape}")
    print(f"  Raw Eye : {X_eye_raw.shape}")
    print(f"  Subjects: {sorted(np.unique(subjects).tolist())}")

    # Remove rows with NaN/Inf in either modality (matches Obj2)
    bad = (np.isnan(X_eeg_raw).any(axis=1) | np.isinf(X_eeg_raw).any(axis=1)
         | np.isnan(X_eye_raw).any(axis=1) | np.isinf(X_eye_raw).any(axis=1))
    n_bad = bad.sum()
    X_eeg_raw = X_eeg_raw[~bad]
    X_eye_raw = X_eye_raw[~bad]
    y         = y[~bad]
    subjects  = subjects[~bad]

    print(f"  Removed {n_bad} corrupted rows → {len(y)} clean samples")
    u, c = np.unique(y, return_counts=True)
    print(f"  Classes : {dict(zip(u.tolist(), c.tolist()))}")
    sys.stdout.flush()
    return X_eeg_raw, X_eye_raw, y, subjects


# =============================================================================
# PER-FOLD PREPROCESSING — identical to Objective 2's preprocess_fold()
# =============================================================================

def preprocess_fold(X_eeg_tr, X_eye_tr, X_eeg_te, X_eye_te):
    """
    Per-fold preprocessing (NO LEAKAGE):
      1. PCA on EEG train only → transform test  (95% variance retained)
      2. Eye NaN imputation with train column means
      3. Scale fused = joint StandardScaler fit on train only

    Returns dict with fused_tr, fused_te, eeg_pca_tr, eye_clean_tr
    and dimensions (for weighted fusion).

    This is Objective 2's preprocessing EXACTLY — ensures baseline ≡ Obj2.
    """
    # --- EEG: per-fold PCA (fit on train only) ---
    pca = PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)
    eeg_tr_pca = pca.fit_transform(X_eeg_tr)
    eeg_te_pca = pca.transform(X_eeg_te)

    # --- Eye: impute NaN with training column means (NO LEAKAGE) ---
    eye_train_mean = np.nanmean(X_eye_tr, axis=0)
    eye_train_mean = np.nan_to_num(eye_train_mean, nan=0.0)
    eye_tr_c = X_eye_tr.copy()
    eye_te_c = X_eye_te.copy()
    for col in range(X_eye_tr.shape[1]):
        eye_tr_c[np.isnan(eye_tr_c[:, col]), col] = eye_train_mean[col]
        eye_te_c[np.isnan(eye_te_c[:, col]), col] = eye_train_mean[col]

    return {
        "eeg_tr": eeg_tr_pca, "eeg_te": eeg_te_pca,
        "eye_tr": eye_tr_c,   "eye_te": eye_te_c,
        "n_eeg": eeg_tr_pca.shape[1],
        "n_eye": eye_tr_c.shape[1],
    }


def weighted_concat_fuse(eeg, eye, w1, w2):
    """
    Weighted concatenation:  [w1 * EEG_pca  |  w2 * Eye]

    All features (EEG-PCA + Eye) are preserved.
    PSO weights scale each modality block BEFORE joint scaling,
    controlling relative influence of each modality in the fused space.

    When w1 = w2 = 0.5  →  standard concatenation (baseline / Obj2).
    """
    total = w1 + w2
    w1_n, w2_n = (w1/total, w2/total) if total > 0 else (0.5, 0.5)
    return np.hstack([w1_n * eeg, w2_n * eye])


def apply_fusion_and_scale(eeg_tr, eye_tr, eeg_te, eye_te, w1, w2):
    """
    Fuse with given weights then apply joint StandardScaler.
    Scaler is fit on training data only — NO LEAKAGE.
    """
    X_tr = weighted_concat_fuse(eeg_tr, eye_tr, w1, w2)
    X_te = weighted_concat_fuse(eeg_te, eye_te, w1, w2)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr, X_te


# =============================================================================
# FAST PSO FITNESS FUNCTION (Logistic Regression surrogate)
# =============================================================================

def pso_fitness(particle,
                eeg_pso_tr, eye_pso_tr,
                eeg_pso_vl, eye_pso_vl,
                y_pso_tr, y_pso_vl):
    """
    Evaluate fitness of candidate weights (w1, w2) using Logistic Regression.

    Data passed in is already PCA-transformed EEG and imputed Eye
    (from the fold's preprocess_fold output).
    Scaler is fit on PSO-train → applied to PSO-val (no leakage).
    """
    w1, w2 = float(particle[0]), float(particle[1])

    X_tr, X_vl = apply_fusion_and_scale(
        eeg_pso_tr, eye_pso_tr, eeg_pso_vl, eye_pso_vl, w1, w2
    )
    lr = LogisticRegression(
        max_iter=500, solver="lbfgs", multi_class="multinomial",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    lr.fit(X_tr, y_pso_tr)
    return accuracy_score(y_pso_vl, lr.predict(X_vl))


# =============================================================================
# PARTICLE SWARM OPTIMISATION
# =============================================================================

class PSO:
    """
    PSO for 2D weight search [w1, w2] ∈ [0,1]^2.

    Velocity: v = inertia*v + c1*r1*(pbest-pos) + c2*r2*(gbest-pos)
    Positions clipped to [0,1]; normalized to sum=1 at convergence.
    """
    def __init__(self, n_particles=PSO_N_PARTICLES,
                 n_iterations=PSO_N_ITERATIONS,
                 inertia=PSO_INERTIA, c1=PSO_C1, c2=PSO_C2,
                 seed=RANDOM_STATE):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.inertia      = inertia
        self.c1           = c1
        self.c2           = c2
        self.rng          = np.random.default_rng(seed)

    def optimize(self, fitness_fn, fold_k, verbose=True):
        rng = self.rng
        positions  = rng.uniform(0, 1, (self.n_particles, 2))
        velocities = rng.uniform(-0.1, 0.1, (self.n_particles, 2))

        pbest_pos = positions.copy()
        pbest_fit = np.array([fitness_fn(p) for p in positions])

        gbest_idx = np.argmax(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]
        history   = [(0, float(gbest_fit))]

        if verbose:
            print(f"    [PSO] Init  gbest={gbest_fit:.4f}"
                  f"  w=[{gbest_pos[0]:.3f}, {gbest_pos[1]:.3f}]")
            sys.stdout.flush()

        for it in range(1, self.n_iterations + 1):
            r1 = rng.uniform(0, 1, (self.n_particles, 2))
            r2 = rng.uniform(0, 1, (self.n_particles, 2))
            velocities = (self.inertia * velocities
                          + self.c1 * r1 * (pbest_pos - positions)
                          + self.c2 * r2 * (gbest_pos  - positions))
            positions  = np.clip(positions + velocities, 0.0, 1.0)

            fitnesses = np.array([fitness_fn(p) for p in positions])
            improved  = fitnesses > pbest_fit
            pbest_pos[improved] = positions[improved]
            pbest_fit[improved] = fitnesses[improved]

            best_idx = np.argmax(pbest_fit)
            if pbest_fit[best_idx] > gbest_fit:
                gbest_fit = pbest_fit[best_idx]
                gbest_pos = pbest_pos[best_idx].copy()

            history.append((it, float(gbest_fit)))
            if verbose and (it % 2 == 0 or it == 1):
                print(f"    [PSO] Iter {it:2d}/{self.n_iterations}"
                      f"  gbest={gbest_fit:.4f}"
                      f"  w=[{gbest_pos[0]:.3f}, {gbest_pos[1]:.3f}]")
                sys.stdout.flush()

        total = gbest_pos[0] + gbest_pos[1]
        if total > 0:
            gbest_pos = gbest_pos / total
        return gbest_pos, gbest_fit, history


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
# BASELINE — equal weights (mirrors Objective 2 DNN exactly)
# =============================================================================

def run_baseline_fold(data, y_tr, y_te):
    """
    Baseline: [0.5*EEG_pca | 0.5*Eye] → joint scale → DNN.
    Identical to Objective 2's DNN pipeline.
    """
    X_tr, X_te = apply_fusion_and_scale(
        data["eeg_tr"], data["eye_tr"],
        data["eeg_te"], data["eye_te"],
        w1=1.0, w2=1.0   # normalizes to 0.5 / 0.5
    )
    model = train_dnn(X_tr, y_tr, X_tr.shape[1])
    return eval_dnn(model, X_te, y_te)


# =============================================================================
# PSO FUSION — per fold
# =============================================================================

def run_pso_fold(data, y_tr, y_te, fold_k):
    """
    PSO-optimised weighted concatenation for one fold.

    Pipeline (NO LEAKAGE AT ANY STEP):
      1. Split preprocessed fold-train into PSO-train / PSO-val (15%)
      2. PSO searches (w1, w2) using LogReg proxy on PSO-train/val
         (scaler fit inside each fitness call on PSO-train only)
      3. Apply best weights to FULL fold-train & fold-test
         + fit joint StandardScaler on fold-train only
      4. Train final DNN on weighted-fused 58D features
      5. Evaluate ONCE on fold-test
    """
    eeg_tr = data["eeg_tr"];  eye_tr = data["eye_tr"]
    eeg_te = data["eeg_te"];  eye_te = data["eye_te"]

    # Step 1 — internal PSO split (from preprocessed train data)
    (Xe_pso_tr, Xe_pso_vl,
     Ey_pso_tr, Ey_pso_vl,
     y_pso_tr,  y_pso_vl) = train_test_split(
        eeg_tr, eye_tr, y_tr,
        test_size=0.15, stratify=y_tr, random_state=RANDOM_STATE
    )

    # Step 2 — PSO with LogReg proxy
    def fitness_fn(particle):
        return pso_fitness(particle,
                           Xe_pso_tr, Ey_pso_tr,
                           Xe_pso_vl, Ey_pso_vl,
                           y_pso_tr,  y_pso_vl)

    print(f"\n  [Fold {fold_k}] Running PSO "
          f"({PSO_N_PARTICLES}P × {PSO_N_ITERATIONS}I)...")
    sys.stdout.flush()
    pso = PSO(seed=RANDOM_STATE + fold_k)
    best_pos, best_fit, history = pso.optimize(fitness_fn, fold_k)
    w1, w2 = float(best_pos[0]), float(best_pos[1])

    print(f"  [Fold {fold_k}] PSO done: w1={w1:.4f}(EEG)  "
          f"w2={w2:.4f}(Eye)  proxy_acc={best_fit:.4f}")
    sys.stdout.flush()

    # Step 3 — apply best weights to full fold data
    X_tr, X_te = apply_fusion_and_scale(eeg_tr, eye_tr, eeg_te, eye_te,
                                        w1, w2)

    # Steps 4-5 — final DNN training + test evaluation
    print(f"  [Fold {fold_k}] Training DNN "
          f"({EPOCHS} epochs, dim={X_tr.shape[1]})...")
    sys.stdout.flush()
    model = train_dnn(X_tr, y_tr, X_tr.shape[1])
    acc, prec, rec, f1 = eval_dnn(model, X_te, y_te)

    print(f"  [Fold {fold_k}] DNN TEST: acc={acc:.4f}  "
          f"prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")
    sys.stdout.flush()
    return acc, prec, rec, f1, w1, w2, history


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_accuracy_comparison(baseline_df, pso_df, obj2_dnn_acc, out_dir):
    """Per-fold + mean accuracy comparison, including Obj2 reference line."""
    ba = baseline_df["accuracy"].values * 100
    pa = pso_df["accuracy"].values * 100
    x  = np.arange(N_SPLITS)
    bw = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Objective 3 — Baseline vs PSO Adaptive Fusion",
                 fontsize=14, fontweight="bold")

    b1 = axes[0].bar(x - bw/2, ba, bw, label="Baseline (equal weights)",
                     color="#4C72B0", alpha=0.85)
    b2 = axes[0].bar(x + bw/2, pa, bw, label="PSO Fusion (opt. weights)",
                     color="#DD8452", alpha=0.85)
    axes[0].axhline(25, ls="--", color="gray", lw=1, label="Chance (25%)")
    axes[0].axhline(obj2_dnn_acc * 100, ls="-.", color="green", lw=1.5,
                    label=f"Obj2 DNN ref ({obj2_dnn_acc*100:.1f}%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Fold {i+1}" for i in x])
    axes[0].set_xlabel("Fold"); axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Per-Fold Test Accuracy")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0, 75)
    for bar in b1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for bar in b2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}", ha="center", fontsize=8)

    means = [ba.mean(), pa.mean()]
    stds  = [ba.std(),  pa.std()]
    bars  = axes[1].bar(["Baseline\n(equal)", "PSO Fusion\n(opt.)"],
                        means, color=["#4C72B0", "#DD8452"], alpha=0.85,
                        yerr=stds, capsize=8, error_kw={"lw": 2})
    axes[1].axhline(25, ls="--", color="gray", lw=1)
    axes[1].axhline(obj2_dnn_acc * 100, ls="-.", color="green", lw=1.5,
                    label=f"Obj2 DNN ({obj2_dnn_acc*100:.1f}%)")
    axes[1].legend(fontsize=9)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m + s + 1.0, f"{m:.1f}%", ha="center",
                     fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Mean Test Accuracy (%)")
    axes[1].set_title("Overall Mean ± Std"); axes[1].set_ylim(0, 75)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: plots/accuracy_comparison.png")


def plot_weight_distribution(weights_df, out_dir):
    folds = weights_df["fold"].values
    w1s   = weights_df["w1_eeg"].values
    w2s   = weights_df["w2_eye"].values
    x     = np.arange(len(folds))
    bw    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PSO Optimal Fusion Weights per Fold",
                 fontsize=14, fontweight="bold")

    axes[0].bar(x - bw/2, w1s, bw, label="w1 (EEG)", color="#4C72B0", alpha=0.85)
    axes[0].bar(x + bw/2, w2s, bw, label="w2 (Eye)", color="#55A868", alpha=0.85)
    axes[0].axhline(0.5, ls="--", color="gray", lw=1, label="Equal (0.5)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Fold {f}" for f in folds])
    axes[0].set_ylabel("Normalized Weight"); axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Weight per Modality per Fold"); axes[0].legend(fontsize=9)
    for xi, (a, b) in enumerate(zip(w1s, w2s)):
        axes[0].text(xi - bw/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        axes[0].text(xi + bw/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)

    sc = axes[1].scatter(w1s, w2s, c=folds, cmap="viridis", s=180,
                         edgecolors="black", zorder=5, linewidths=0.8)
    for i, f in enumerate(folds):
        axes[1].annotate(f"F{f}", (w1s[i], w2s[i]),
                         textcoords="offset points", xytext=(8, 4), fontsize=10)
    plt.colorbar(sc, ax=axes[1], label="Fold")
    axes[1].set_xlabel("w1 (EEG)"); axes[1].set_ylabel("w2 (Eye)")
    axes[1].set_title("Optimal Weights in 2D Space")
    axes[1].set_xlim(-0.05, 1.05); axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.5)
    axes[1].axvline(0.5, ls="--", color="gray", lw=0.8, alpha=0.5)
    axes[1].plot([0, 1], [1, 0], ls=":", color="red", lw=1, label="w1+w2=1")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "weight_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: plots/weight_distribution.png")


def plot_pso_convergence(histories, out_dir):
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, hist in enumerate(histories, 1):
        iters = [h[0] for h in hist]
        fits  = [h[1] * 100 for h in hist]
        ax.plot(iters, fits, marker="o", ms=4, lw=1.8,
                color=colors[k - 1], label=f"Fold {k}")
    ax.axhline(25, ls="--", color="gray", lw=1, label="Chance (25%)")
    ax.set_xlabel("PSO Iteration"); ax.set_ylabel("Best Proxy Accuracy (%)")
    ax.set_title("PSO Convergence per Fold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pso_convergence.png"), dpi=150)
    plt.close()
    print("  Saved: plots/pso_convergence.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    OBJ2_DNN_REF = 0.4681   # Objective 2 DNN mean accuracy reference

    print("\n" + "=" * 65)
    print("  OBJECTIVE 3 (CORRECTED v2) — PSO ADAPTIVE FUSION")
    print("  Fusion   : [w1*EEG_pca | w2*Eye] → weighted concatenation")
    print("  Data src : objective1/ CSVs  (same as Objective 2)")
    print("  Preproc  : per-fold PCA (95% var) + NaN impute + joint scale")
    print("=" * 65)
    print(f"  Folds={N_SPLITS}  Epochs={EPOCHS}  PCA_var={PCA_VAR}")
    print(f"  PSO: {PSO_N_PARTICLES}P × {PSO_N_ITERATIONS}I  "
          f"inertia={PSO_INERTIA}  c1={PSO_C1}  c2={PSO_C2}")
    print()
    sys.stdout.flush()

    # 1. Load raw data from CSVs (same as Obj2)
    X_eeg_raw, X_eye_raw, y, subjects = load_data()

    # 2. GroupKFold (MANDATORY — mirrors Obj2)
    gkf = GroupKFold(n_splits=N_SPLITS)

    baseline_rows = []
    pso_rows      = []
    weights_rows  = []
    histories     = []

    # 3. Cross-validation loop
    for fold_k, (tr_idx, te_idx) in enumerate(
            gkf.split(X_eeg_raw, y, groups=subjects), start=1):

        tr_subj = np.unique(subjects[tr_idx])
        te_subj = np.unique(subjects[te_idx])
        overlap = set(tr_subj.tolist()) & set(te_subj.tolist())

        print(f"\n{'='*65}")
        print(f"  FOLD {fold_k}/{N_SPLITS}")
        print(f"  Train subjects: {sorted(tr_subj.tolist())}")
        print(f"  Test  subjects: {sorted(te_subj.tolist())}")
        print(f"  Overlap (MUST be empty): {overlap}")
        assert len(overlap) == 0, f"LEAKAGE in fold {fold_k}: {overlap}"

        Xe_tr = X_eeg_raw[tr_idx];  Xe_te = X_eeg_raw[te_idx]
        Ey_tr = X_eye_raw[tr_idx];  Ey_te = X_eye_raw[te_idx]
        y_tr  = y[tr_idx];          y_te  = y[te_idx]
        print(f"  Train: {len(y_tr)} samples  |  Test: {len(y_te)} samples")
        sys.stdout.flush()

        # Per-fold preprocessing (PCA + impute) — identical to Obj2
        data = preprocess_fold(Xe_tr, Ey_tr, Xe_te, Ey_te)
        print(f"  PCA → {data['n_eeg']} EEG components  |  "
              f"{data['n_eye']} Eye features")
        sys.stdout.flush()

        # --- BASELINE (equal-weight fusion → Obj2-equivalent) ---
        print(f"\n  [Fold {fold_k}] BASELINE (equal weights + DNN)...")
        sys.stdout.flush()
        b_acc, b_prec, b_rec, b_f1 = run_baseline_fold(data, y_tr, y_te)
        print(f"  [Fold {fold_k}] BASELINE: acc={b_acc:.4f}  "
              f"prec={b_prec:.4f}  rec={b_rec:.4f}  f1={b_f1:.4f}")
        sys.stdout.flush()
        baseline_rows.append({
            "fold": fold_k, "method": "Baseline",
            "accuracy": round(b_acc, 4), "precision": round(b_prec, 4),
            "recall": round(b_rec, 4), "f1": round(b_f1, 4)
        })

        # --- PSO FUSION ---
        p_acc, p_prec, p_rec, p_f1, w1, w2, hist = run_pso_fold(
            data, y_tr, y_te, fold_k
        )
        pso_rows.append({
            "fold": fold_k, "method": "PSO",
            "accuracy": round(p_acc, 4), "precision": round(p_prec, 4),
            "recall": round(p_rec, 4), "f1": round(p_f1, 4)
        })
        weights_rows.append({
            "fold": fold_k,
            "w1_eeg": round(w1, 4), "w2_eye": round(w2, 4)
        })
        histories.append(hist)

    # 4. DataFrames
    baseline_df = pd.DataFrame(baseline_rows)
    pso_df      = pd.DataFrame(pso_rows)
    weights_df  = pd.DataFrame(weights_rows)

    # 5. Save all CSVs
    pso_df.to_csv(os.path.join(OBJ3_DIR, "pso_results.csv"), index=False)
    weights_df.to_csv(os.path.join(OBJ3_DIR, "pso_weights.csv"), index=False)
    pd.concat([baseline_df, pso_df], ignore_index=True).to_csv(
        os.path.join(OBJ3_DIR, "baseline_vs_pso.csv"), index=False)

    summary_rows = []
    for lbl, df in [("Baseline", baseline_df), ("PSO Fusion", pso_df)]:
        summary_rows.append({
            "method":         lbl,
            "mean_accuracy":  round(df["accuracy"].mean(), 4),
            "std_accuracy":   round(df["accuracy"].std(),  4),
            "mean_precision": round(df["precision"].mean(), 4),
            "std_precision":  round(df["precision"].std(),  4),
            "mean_recall":    round(df["recall"].mean(),    4),
            "std_recall":     round(df["recall"].std(),     4),
            "mean_f1":        round(df["f1"].mean(),        4),
            "std_f1":         round(df["f1"].std(),         4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OBJ3_DIR, "comparison_summary.csv"), index=False)
    summary_df.to_csv(
        os.path.join(OBJ3_DIR, "objective3_corrected_results.csv"), index=False)

    print(f"\n  Saved: pso_results.csv")
    print(f"  Saved: pso_weights.csv")
    print(f"  Saved: baseline_vs_pso.csv")
    print(f"  Saved: comparison_summary.csv")
    print(f"  Saved: objective3_corrected_results.csv")

    # 6. Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  Objective 2 DNN reference: {OBJ2_DNN_REF*100:.2f}% (mean, 5-fold)")
    print(f"  {'Method':<22} {'Acc (mean +/- std)':<24} {'F1 (mean +/- std)'}")
    print(f"  {'-'*62}")
    for lbl, df in [("Baseline", baseline_df), ("PSO Fusion", pso_df)]:
        accs = df["accuracy"].values
        f1s  = df["f1"].values
        print(f"  {lbl:<22} "
              f"{accs.mean()*100:>6.2f}% +/- {accs.std()*100:>4.2f}%    "
              f"{f1s.mean()*100:>6.2f}% +/- {f1s.std()*100:>4.2f}%")

    delta_acc = (pso_df["accuracy"].mean() - baseline_df["accuracy"].mean()) * 100
    delta_f1  = (pso_df["f1"].mean()       - baseline_df["f1"].mean())       * 100
    print(f"\n  Delta Accuracy : {delta_acc:+.2f}%")
    print(f"  Delta F1       : {delta_f1:+.2f}%")

    print(f"\n  PSO Optimal Weights per Fold:")
    print(f"  {'Fold':<6} {'w1 (EEG)':<12} {'w2 (Eye)'}")
    for _, r in weights_df.iterrows():
        dom = ("← EEG dom." if r.w1_eeg > 0.55 else
               "← Eye dom." if r.w2_eye > 0.55 else "← balanced")
        print(f"  {int(r.fold):<6} {r.w1_eeg:<12.4f} {r.w2_eye:.4f}  {dom}")

    # 7. Sanity checks
    pso_accs  = pso_df["accuracy"].values
    bl_accs   = baseline_df["accuracy"].values
    spread    = pso_accs.max() - pso_accs.min()
    w1_spread = weights_df["w1_eeg"].max() - weights_df["w1_eeg"].min()
    same      = np.allclose(bl_accs, pso_accs, atol=1e-4)

    print(f"\n  [Sanity] PSO acc spread     = {spread:.4f}  "
          f"({'OK' if spread > 0.005 else 'WARN'})")
    print(f"  [Sanity] Weight w1 spread   = {w1_spread:.4f}  "
          f"({'OK — varies' if w1_spread > 0.01 else 'WARN — converged same'})")
    print(f"  [Sanity] Baseline vs PSO    : "
          f"{'FAIL — identical!' if same else 'OK — differ'}")
    bl_vs_obj2 = abs(bl_accs.mean() - OBJ2_DNN_REF) * 100
    print(f"  [Sanity] Baseline vs Obj2   : {bl_accs.mean()*100:.2f}% "
          f"(gap={bl_vs_obj2:.2f}%, {'OK' if bl_vs_obj2 < 3.0 else 'CHECK'})")

    # 8. Plots
    print("\n  [Plots]")
    plot_accuracy_comparison(baseline_df, pso_df, OBJ2_DNN_REF, PLOT_DIR)
    plot_weight_distribution(weights_df, PLOT_DIR)
    plot_pso_convergence(histories, PLOT_DIR)

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\n" + "=" * 65)
    print("  OBJECTIVE 3 CORRECTED v2 COMPLETE — NO SUBJECT LEAKAGE")
    print(f"  All outputs: {OBJ3_DIR}")
    print("=" * 65)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
