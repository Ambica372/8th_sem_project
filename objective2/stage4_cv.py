# -*- coding: utf-8 -*-
"""
=============================================================================
Objective 2 — Subject-Aware GroupKFold Cross-Validation Pipeline
=============================================================================

Data source  : objective1/eeg_features.csv + objective1/eye_features.csv
Splitting    : GroupKFold(n_splits=5) — groups = subject_id
               NO subject appears in both train and test within any fold!
Preprocessing: Per-fold PCA (EEG), per-fold StandardScaler (fit on train)
Models       : MLP, DNN, Attention, Hybrid, Decision Fusion (unchanged)
Metrics      : Accuracy, Precision, Recall, F1  (macro average)
Outputs      : cv_fold_results.csv, cv_summary_results.csv, charts
=============================================================================
"""

import os
import sys
import json
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

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# Windows console UTF-8 fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OBJ2_DIR   = os.path.dirname(os.path.abspath(__file__))          # objective2/
OBJ1_DIR   = os.path.join(os.path.dirname(OBJ2_DIR), "objective1")
OUTPUT_DIR = os.path.join(OBJ2_DIR, "stage4_cv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EEG_CSV = os.path.join(OBJ1_DIR, "eeg_features.csv")
EYE_CSV = os.path.join(OBJ1_DIR, "eye_features.csv")

# ---------------------------------------------------------------------------
# Hyper-parameters (UNCHANGED from original)
# ---------------------------------------------------------------------------
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 1e-4
N_SPLITS     = 5
VAL_FRAC     = 0.1      # internal val split from training data
PCA_VAR      = 0.95     # retain 95% variance for EEG PCA
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Feature/meta column constants
META_COLS = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# =============================================================================
# MODEL ARCHITECTURES — identical to original run_models.py
# =============================================================================

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, 128)
        self.bn1      = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2      = nn.Linear(128, 64)
        self.bn2      = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.out      = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.out(x)


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


class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        attn = torch.sigmoid(self.attention_weights(x))
        return self.out(F.relu(self.fc1(x * attn)))


class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1               = nn.Linear(input_dim, 128)
        self.attention_weights = nn.Linear(128, 128)
        self.fc2               = nn.Linear(128, 64)
        self.out               = nn.Linear(64, num_classes)

    def forward(self, x):
        x    = F.relu(self.fc1(x))
        attn = torch.sigmoid(self.attention_weights(x))
        return self.out(F.relu(self.fc2(x * attn)))


class DecisionFusion(nn.Module):
    def __init__(self, eeg_dim, eye_dim, num_classes=4):
        super().__init__()
        self.eeg_fc  = nn.Linear(eeg_dim, 64)
        self.eeg_out = nn.Linear(64, num_classes)
        self.eye_fc  = nn.Linear(eye_dim, 32)
        self.eye_out = nn.Linear(32, num_classes)

    def forward(self, eeg_x, eye_x):
        return (self.eeg_out(F.relu(self.eeg_fc(eeg_x)))
                + self.eye_out(F.relu(self.eye_fc(eye_x)))) / 2.0


# =============================================================================
# DATA LOADING  (from CSVs — preserves subject_id)
# =============================================================================

def load_data():
    """
    Load EEG and Eye CSVs from objective1/.
    Extract features, labels, and subject IDs.
    Clean NaN/Inf rows consistently across both modalities.
    Returns: X_eeg_raw, X_eye_raw, y, subjects  (all numpy arrays, aligned)
    """
    print("[1] Loading data from CSVs ...")
    print(f"    EEG: {EEG_CSV}")
    print(f"    Eye: {EYE_CSV}")

    eeg_df = pd.read_csv(EEG_CSV)
    eye_df = pd.read_csv(EYE_CSV)

    assert len(eeg_df) == len(eye_df), \
        f"Row count mismatch: EEG={len(eeg_df)}, Eye={len(eye_df)}"

    # Extract feature columns
    eeg_feat_cols = [c for c in eeg_df.columns if c not in META_COLS]
    eye_feat_cols = [c for c in eye_df.columns if c not in META_COLS]

    X_eeg_raw = eeg_df[eeg_feat_cols].values.astype(np.float64)
    X_eye_raw = eye_df[eye_feat_cols].values.astype(np.float64)
    y         = eeg_df["emotion_label"].values.astype(np.int64)
    subjects  = eeg_df["subject_id"].values.astype(np.int64)

    print(f"    Raw shapes -> EEG: {X_eeg_raw.shape} | Eye: {X_eye_raw.shape}")
    print(f"    Subjects: {sorted(np.unique(subjects).tolist())}")

    # Remove NaN / Inf rows consistently across both modalities
    bad = (np.isnan(X_eeg_raw).any(axis=1) | np.isinf(X_eeg_raw).any(axis=1)
         | np.isnan(X_eye_raw).any(axis=1) | np.isinf(X_eye_raw).any(axis=1))

    n_removed = bad.sum()
    X_eeg_raw = X_eeg_raw[~bad]
    X_eye_raw = X_eye_raw[~bad]
    y         = y[~bad]
    subjects  = subjects[~bad]

    print(f"    Removed {n_removed} corrupted rows -> {len(y)} clean samples")

    # Validate
    for name, arr in [("X_eeg", X_eeg_raw), ("X_eye", X_eye_raw)]:
        assert not np.isnan(arr).any() and not np.isinf(arr).any(), \
            f"NaN/Inf still present in {name} after cleaning"

    unique, counts = np.unique(y, return_counts=True)
    print(f"    Classes: {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"    Samples per subject:")
    for s in sorted(np.unique(subjects)):
        print(f"      Subject {s:2d}: {(subjects == s).sum()} samples")

    return X_eeg_raw, X_eye_raw, y, subjects


# =============================================================================
# PER-FOLD PREPROCESSING  (NO LEAKAGE)
# =============================================================================

def preprocess_fold(X_eeg_tr, X_eye_tr, X_eeg_te, X_eye_te):
    """
    Per-fold preprocessing:
      1. PCA on training EEG only -> transform test
      2. Eye imputation with train column means
      3. Fuse EEG-PCA + Eye
      4. Scale fused (train only)
      5. Scale EEG-PCA and Eye separately for Decision Fusion
    Returns dict of preprocessed arrays and dimensions.
    """
    # --- PCA on EEG (fit on train only) ---
    pca = PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)
    eeg_tr_pca = pca.fit_transform(X_eeg_tr)
    eeg_te_pca = pca.transform(X_eeg_te)

    # --- Eye imputation (train column means) ---
    eye_train_mean = np.nanmean(X_eye_tr, axis=0)
    X_eye_tr_clean = X_eye_tr.copy()
    X_eye_te_clean = X_eye_te.copy()
    for col in range(X_eye_tr.shape[1]):
        X_eye_tr_clean[np.isnan(X_eye_tr_clean[:, col]), col] = eye_train_mean[col]
        X_eye_te_clean[np.isnan(X_eye_te_clean[:, col]), col] = eye_train_mean[col]

    # --- Fuse ---
    fused_tr = np.hstack([eeg_tr_pca, X_eye_tr_clean])
    fused_te = np.hstack([eeg_te_pca, X_eye_te_clean])

    # --- Scale fused (fit on train only) ---
    fused_sc = StandardScaler()
    fused_tr = fused_sc.fit_transform(fused_tr)
    fused_te = fused_sc.transform(fused_te)

    # --- Scale EEG-PCA and Eye separately for Decision Fusion ---
    eeg_sc = StandardScaler()
    eeg_tr_s = eeg_sc.fit_transform(eeg_tr_pca)
    eeg_te_s = eeg_sc.transform(eeg_te_pca)

    eye_sc = StandardScaler()
    eye_tr_s = eye_sc.fit_transform(X_eye_tr_clean)
    eye_te_s = eye_sc.transform(X_eye_te_clean)

    # Sanity checks
    for tag, arr in [("fused_tr", fused_tr), ("fused_te", fused_te),
                      ("eeg_tr", eeg_tr_s), ("eye_tr", eye_tr_s)]:
        assert not np.isnan(arr).any(), f"NaN in {tag}"
        assert not np.isinf(arr).any(), f"Inf in {tag}"

    return {
        "fused_tr": fused_tr, "fused_te": fused_te,
        "eeg_tr": eeg_tr_s,   "eeg_te": eeg_te_s,
        "eye_tr": eye_tr_s,   "eye_te": eye_te_s,
        "fused_dim": fused_tr.shape[1],
        "eeg_dim": eeg_tr_s.shape[1],
        "eye_dim": eye_tr_s.shape[1],
        "n_pca": eeg_tr_pca.shape[1],
    }


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_eval_fold(model_name, model, fold_k,
                    Xtr, ytr, Xte, yte,
                    model_dir,
                    X2tr=None, X2te=None):
    """
    Train on (Xtr, ytr) with internal val split.
    Model selection uses VALIDATION accuracy.
    Final evaluation on (Xte, yte) — never seen during training.

    Returns: acc, prec, rec, f1  (macro)
    """
    ensure_dir(model_dir)

    # --- Internal validation split from training data -----------------------
    if X2tr is not None:
        (Xtr_t, Xval_t, X2tr_t, X2val_t,
         ytr_t, yval_t) = train_test_split(
            Xtr, X2tr, ytr,
            test_size=VAL_FRAC, stratify=ytr, random_state=RANDOM_STATE
        )
    else:
        Xtr_t, Xval_t, ytr_t, yval_t = train_test_split(
            Xtr, ytr,
            test_size=VAL_FRAC, stratify=ytr, random_state=RANDOM_STATE
        )
        X2tr_t = X2val_t = None

    is_dec = (X2tr is not None)

    # Convert to tensors
    def to_t(arr):
        return torch.FloatTensor(arr)

    Xtr_pt  = to_t(Xtr_t);   Xval_pt = to_t(Xval_t)
    Xte_pt  = to_t(Xte)
    ytr_pt  = torch.LongTensor(ytr_t)
    yval_pt = torch.LongTensor(yval_t)
    yte_pt  = torch.LongTensor(yte)

    if is_dec:
        X2tr_pt  = to_t(X2tr_t);   X2val_pt = to_t(X2val_t)
        X2te_pt  = to_t(X2te)

    print(f"\n  [{model_name}] Fold {fold_k} | "
          f"train={len(ytr_t)} val={len(yval_t)} test={len(yte)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc  = 0.0
    best_ckpt     = os.path.join(model_dir, f"best_fold{fold_k}.pth")

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr_pt))
        ep_loss, n_batches = 0.0, 0

        for i in range(0, len(ytr_pt), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()

            out = (model(Xtr_pt[idx], X2tr_pt[idx]) if is_dec
                   else model(Xtr_pt[idx]))
            loss = criterion(out, ytr_pt[idx])

            if torch.isnan(loss):
                print(f"  [WARN] NaN loss at epoch {ep+1}, stopping.")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss  += loss.item()
            n_batches += 1

        if n_batches == 0:
            break

        # --- Validate on internal val set (NOT the test fold) ---------------
        model.eval()
        with torch.no_grad():
            val_out = (model(Xval_pt, X2val_pt) if is_dec else model(Xval_pt))
            _, val_pred = torch.max(val_out, 1)
            val_acc = accuracy_score(yval_pt.numpy(), val_pred.numpy())

        # --- Model selection: save checkpoint when val improves -------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    Epoch [{ep+1:2d}/{EPOCHS}] "
                  f"loss={ep_loss/n_batches:.4f}  val_acc={val_acc:.4f}")

    # --- Load best checkpoint -----------------------------------------------
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, weights_only=True))

    # --- Evaluate ONLY on test fold (first and only time) -------------------
    model.eval()
    with torch.no_grad():
        test_out = (model(Xte_pt, X2te_pt) if is_dec else model(Xte_pt))
        _, test_pred = torch.max(test_out, 1)

    y_true = yte_pt.numpy()
    y_pred = test_pred.numpy()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"    -> TEST  acc={acc:.4f}  prec={prec:.4f}  "
          f"rec={rec:.4f}  f1={f1:.4f}")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} — Fold {fold_k}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"cm_fold{fold_k}.png"), dpi=100)
    plt.close()

    # Save classification report
    report = classification_report(y_true, y_pred, zero_division=0)
    with open(os.path.join(model_dir, f"report_fold{fold_k}.txt"), "w") as f:
        f.write(f"Fold: {fold_k}\n")
        f.write(f"Test Acc: {acc:.4f}\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n\n")
        f.write(report)

    return acc, prec, rec, f1


# =============================================================================
# PLOT UTILITIES
# =============================================================================

def plot_cv_results(summary_df, out_dir):
    """Bar chart — mean accuracy +/- std for each model."""
    models = summary_df["model"].tolist()
    means  = summary_df["mean_accuracy"].values * 100
    stds   = summary_df["std_accuracy"].values  * 100

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("5-Fold Subject-Level GroupKFold CV — Objective 2",
                 fontsize=13, fontweight="bold")

    # Accuracy bar chart with error bars
    bars = axes[0].barh(models, means, xerr=stds, color=colors,
                        alpha=0.85, capsize=5,
                        error_kw={"ecolor": "black", "lw": 1.5})
    axes[0].axvline(25, color="gray", ls="--", lw=1, label="Chance (25%)")
    axes[0].set_xlabel("Mean Test Accuracy (%)", fontsize=11)
    axes[0].set_title("Accuracy (mean +/- std)", fontsize=11)
    axes[0].set_xlim(0, 100)
    for bar, val in zip(bars, means):
        axes[0].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)
    axes[0].legend()

    # F1 bar chart
    f1_means = summary_df["mean_f1"].values * 100
    f1_stds  = summary_df["std_f1"].values * 100
    bars2 = axes[1].barh(models, f1_means, xerr=f1_stds, color=colors,
                          alpha=0.85, capsize=5,
                          error_kw={"ecolor": "black", "lw": 1.5})
    axes[1].axvline(25, color="gray", ls="--", lw=1, label="Chance (25%)")
    axes[1].set_xlabel("Mean Macro F1-Score (%)", fontsize=11)
    axes[1].set_title("F1-Score (mean +/- std)", fontsize=11)
    axes[1].set_xlim(0, 100)
    for bar, val in zip(bars2, f1_means):
        axes[1].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", fontsize=9)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_performance_chart.png"), dpi=130)
    plt.close()
    print(f"  Saved: cv_performance_chart.png")


def plot_fold_variance(fold_df, out_dir):
    """Box / strip plot showing per-fold accuracy spread for each model."""
    models = fold_df["model"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_model = [fold_df[fold_df["model"] == m]["accuracy"].values * 100
                     for m in models]
    bp = ax.boxplot(data_by_model, patch_artist=True, widths=0.4,
                    medianprops={"color": "black", "lw": 2},
                    showfliers=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Test Accuracy per Fold (%)", fontsize=11)
    ax.set_title("Fold-Level Accuracy Distribution per Model\n"
                 "(Subject-Level GroupKFold CV)", fontsize=12, fontweight="bold")
    ax.axhline(25, color="gray", ls="--", lw=1, label="Chance (25%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_fold_variance.png"), dpi=130)
    plt.close()
    print(f"  Saved: cv_fold_variance.png")


# =============================================================================
# MAIN — SUBJECT-AWARE GroupKFold CROSS-VALIDATION
# =============================================================================

def main():
    print("=" * 65)
    print(" OBJECTIVE 2 — SUBJECT-AWARE GroupKFold CROSS-VALIDATION")
    print("=" * 65)
    print(f" Output dir : {OUTPUT_DIR}")
    print(f" Folds      : {N_SPLITS} | Epochs: {EPOCHS}"
          f" | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f" Val split  : {VAL_FRAC*100:.0f}% of training fold (stratified)")
    print(f" PCA retain : {PCA_VAR*100:.0f}% variance")
    print(f" Metrics    : Accuracy, Precision, Recall, F1  (macro)")
    print()

    # ------------------------------------------------------------------
    # 1. Load data (from CSVs with subject_id)
    # ------------------------------------------------------------------
    X_eeg_raw, X_eye_raw, y, subjects = load_data()
    n_samples = len(y)

    # ------------------------------------------------------------------
    # 2. Define GroupKFold splitter (groups = subjects)
    # ------------------------------------------------------------------
    gkf = GroupKFold(n_splits=N_SPLITS)

    # Model factory — fresh model for every fold (dim set per-fold after PCA)
    model_names = ["MLP", "DNN", "Attention", "Hybrid", "Decision Fusion"]

    # Collect per-fold results
    all_fold_rows = []
    per_model = {n: {"acc": [], "prec": [], "rec": [], "f1": []}
                 for n in model_names}

    # ------------------------------------------------------------------
    # 3. Cross-validation loop (SUBJECT-AWARE)
    # ------------------------------------------------------------------
    for fold_k, (train_idx, test_idx) in enumerate(
            gkf.split(X_eeg_raw, y, groups=subjects), start=1):

        train_subjects = np.unique(subjects[train_idx])
        test_subjects  = np.unique(subjects[test_idx])
        overlap        = set(train_subjects.tolist()) & set(test_subjects.tolist())

        print(f"\n{'='*65}")
        print(f" FOLD {fold_k} / {N_SPLITS}")
        print(f"  Train samples : {len(train_idx)}"
              f"  |  Test samples : {len(test_idx)}")
        print(f"  Train subjects: {sorted(train_subjects.tolist())}")
        print(f"  Test  subjects: {sorted(test_subjects.tolist())}")
        print(f"  Subject overlap (MUST be empty): {overlap}")
        assert len(overlap) == 0, \
            f"SUBJECT LEAKAGE in fold {fold_k}: {overlap}"

        # --- Class distribution per fold ---
        tr_unique, tr_counts = np.unique(y[train_idx], return_counts=True)
        te_unique, te_counts = np.unique(y[test_idx], return_counts=True)
        print(f"  Train class dist: {dict(zip(tr_unique.tolist(), tr_counts.tolist()))}")
        print(f"  Test  class dist: {dict(zip(te_unique.tolist(), te_counts.tolist()))}")

        # ---- Step 1: raw split ------------------------------------------
        eeg_tr_raw = X_eeg_raw[train_idx]
        eeg_te_raw = X_eeg_raw[test_idx]
        eye_tr_raw = X_eye_raw[train_idx]
        eye_te_raw = X_eye_raw[test_idx]
        y_train    = y[train_idx]
        y_test     = y[test_idx]

        # ---- Step 2: per-fold preprocessing (PCA + scaling, NO LEAKAGE) ---
        data = preprocess_fold(eeg_tr_raw, eye_tr_raw, eeg_te_raw, eye_te_raw)

        fused_dim = data["fused_dim"]
        eeg_dim   = data["eeg_dim"]
        eye_dim   = data["eye_dim"]

        print(f"  Preprocessing done: PCA -> {data['n_pca']} components | "
              f"fused_dim={fused_dim} | eeg_dim={eeg_dim} | eye_dim={eye_dim}")

        # ---- Step 3: train / evaluate each model on this fold -----------
        def make_models():
            return [
                ("MLP",             BaselineMLP(fused_dim),   "mlp"),
                ("DNN",             DeepDNN(fused_dim),       "dnn"),
                ("Attention",       AttentionModel(fused_dim), "attention"),
                ("Hybrid",          HybridModel(fused_dim),   "hybrid"),
                ("Decision Fusion", DecisionFusion(eeg_dim, eye_dim), "decision_fusion"),
            ]

        models = make_models()

        for model_name, model, folder in models:
            is_dec = (model_name == "Decision Fusion")
            model_dir = os.path.join(OUTPUT_DIR, folder)
            ensure_dir(model_dir)

            if is_dec:
                acc, prec, rec, f1 = train_eval_fold(
                    model_name, model, fold_k,
                    data["eeg_tr"], y_train, data["eeg_te"], y_test,
                    model_dir,
                    X2tr=data["eye_tr"], X2te=data["eye_te"]
                )
            else:
                acc, prec, rec, f1 = train_eval_fold(
                    model_name, model, fold_k,
                    data["fused_tr"], y_train, data["fused_te"], y_test,
                    model_dir
                )

            # Store fold row
            all_fold_rows.append({
                "fold":      fold_k,
                "model":     model_name,
                "accuracy":  round(acc,  4),
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
            })
            per_model[model_name]["acc"].append(acc)
            per_model[model_name]["prec"].append(prec)
            per_model[model_name]["rec"].append(rec)
            per_model[model_name]["f1"].append(f1)

    # ------------------------------------------------------------------
    # 4. Save fold-wise results CSV
    # ------------------------------------------------------------------
    fold_df = pd.DataFrame(all_fold_rows)
    fold_csv = os.path.join(OUTPUT_DIR, "cv_fold_results.csv")
    fold_df.to_csv(fold_csv, index=False)
    print(f"\n  Saved: cv_fold_results.csv")

    # ------------------------------------------------------------------
    # 5. Compute and save summary CSV
    # ------------------------------------------------------------------
    summary_rows = []
    for model_name in model_names:
        acc_arr  = np.array(per_model[model_name]["acc"])
        prec_arr = np.array(per_model[model_name]["prec"])
        rec_arr  = np.array(per_model[model_name]["rec"])
        f1_arr   = np.array(per_model[model_name]["f1"])
        summary_rows.append({
            "model":         model_name,
            "mean_accuracy": round(acc_arr.mean(),  4),
            "std_accuracy":  round(acc_arr.std(),   4),
            "mean_precision":round(prec_arr.mean(), 4),
            "std_precision": round(prec_arr.std(),  4),
            "mean_recall":   round(rec_arr.mean(),  4),
            "std_recall":    round(rec_arr.std(),   4),
            "mean_f1":       round(f1_arr.mean(),   4),
            "std_f1":        round(f1_arr.std(),    4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "cv_summary_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved: cv_summary_results.csv")

    # ------------------------------------------------------------------
    # 6. Print final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" FINAL SUBJECT-LEVEL GroupKFold CV SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<20} {'Acc (mean+/-std)':>18} {'F1 (mean+/-std)':>18}")
    print(f"  {'-'*58}")
    for row in summary_rows:
        print(f"  {row['model']:<20} "
              f"{row['mean_accuracy']*100:>7.2f}% +/- {row['std_accuracy']*100:>5.2f}%"
              f"   {row['mean_f1']*100:>7.2f}% +/- {row['std_f1']*100:>5.2f}%")

    # ------------------------------------------------------------------
    # 7. Verify results differ across folds (sanity check)
    # ------------------------------------------------------------------
    print("\n[Sanity Check] Per-fold accuracy spread (should differ):")
    for model_name in model_names:
        fold_accs = [r["accuracy"] for r in all_fold_rows
                     if r["model"] == model_name]
        spread = max(fold_accs) - min(fold_accs)
        status = "OK" if spread > 0.001 else "WARN — identical across folds"
        print(f"  {model_name:<20} folds={[f'{a:.3f}' for a in fold_accs]}"
              f"  spread={spread:.4f}  [{status}]")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    plot_cv_results(summary_df, OUTPUT_DIR)
    plot_fold_variance(fold_df, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(" PIPELINE COMPLETE — NO SUBJECT LEAKAGE")
    print(f" Outputs in: {OUTPUT_DIR}")
    print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
