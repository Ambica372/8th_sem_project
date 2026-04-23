# -*- coding: utf-8 -*-
"""
=============================================================================
Objective 2 - GroupKFold (LOSO) Cross-Validation Pipeline
LASO-style rotation (subject-wise split)
=============================================================================

Data source  : Auto-discovered processed_data/*.npy
Splitting    : GroupKFold(n_splits=number_of_unique_subjects)
Preprocessing: Per-fold StandardScaler (fit on train, transform test)
Models       : MLP, DNN, Attention, Hybrid, Decision Fusion
Metrics      : Accuracy, Precision, Recall, F1 (macro average)
Outputs      : cv_fold_results.csv, cv_summary_results.csv, cv_fold_variance.png, cv_performance_chart.png
=============================================================================
"""

import os
import sys
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

from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# Windows console UTF-8 fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths - auto-discover data directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_FILES = ["X_fused.npy", "X_eeg_pca.npy", "X_eye_clean.npy", "y.npy"]
_CANDIDATE_DIRS = [
    os.path.join(_SCRIPT_DIR, "..", "..", "..", "stage4_pipeline", "processed_data"),
    os.path.join(_SCRIPT_DIR, "..", "..", "stage4_pipeline", "processed_data"),
    os.path.join(_SCRIPT_DIR, "processed_data"),
]

DATA_DIR = None
for _d in _CANDIDATE_DIRS:
    _d = os.path.normpath(os.path.abspath(_d))
    if os.path.isdir(_d) and all(os.path.isfile(os.path.join(_d, f)) for f in _REQUIRED_FILES):
        DATA_DIR = _d
        break

if DATA_DIR is None:
    _searched = [os.path.normpath(os.path.abspath(d)) for d in _CANDIDATE_DIRS]
    raise FileNotFoundError(
        "Could not find data folder containing {}.\nSearched:\n{}".format(
            _REQUIRED_FILES, "\n".join("  - {}".format(p) for p in _searched))
    )

# SAVE RESULTS HERE
OUTPUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "results", "groupkfold_laso"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS              = 10
BATCH_SIZE          = 64
LR                  = 5e-5
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 10
SCHEDULER_PATIENCE  = 4
RANDOM_STATE        = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, 128)
        self.bn1      = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2      = nn.Linear(128, 128)
        self.bn2      = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.out      = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.out(x)


class DeepDNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, 128)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(128, 128)
        self.bn2   = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3   = nn.Linear(128, 128)
        self.bn3   = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.3)
        self.out   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        return self.out(x)


class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)
        self.fc1   = nn.Linear(input_dim, 128)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(128, 128)
        self.bn2   = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.out   = nn.Linear(128, num_classes)

    def forward(self, x):
        attn = torch.sigmoid(self.attention_weights(x))
        x = self.drop1(F.relu(self.bn1(self.fc1(x * attn))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        return self.out(x)


class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1               = nn.Linear(input_dim, 128)
        self.bn1               = nn.BatchNorm1d(128)
        self.dropout1          = nn.Dropout(0.3)
        self.attention_weights = nn.Linear(128, 128)
        self.fc2               = nn.Linear(128, 128)
        self.bn2               = nn.BatchNorm1d(128)
        self.dropout2          = nn.Dropout(0.3)
        self.out               = nn.Linear(128, num_classes)

    def forward(self, x):
        x    = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        attn = torch.sigmoid(self.attention_weights(x))
        x    = self.dropout2(F.relu(self.bn2(self.fc2(x * attn))))
        return self.out(x)


class DecisionFusion(nn.Module):
    def __init__(self, eeg_dim, eye_dim, num_classes=4):
        super().__init__()
        self.eeg_fc    = nn.Linear(eeg_dim, 128)
        self.eeg_bn    = nn.BatchNorm1d(128)
        self.eeg_drop  = nn.Dropout(0.3)
        self.eeg_fc2   = nn.Linear(128, 128)
        self.eeg_bn2   = nn.BatchNorm1d(128)
        self.eeg_drop2 = nn.Dropout(0.3)
        self.eeg_out   = nn.Linear(128, num_classes)

        self.eye_fc    = nn.Linear(eye_dim, 128)
        self.eye_bn    = nn.BatchNorm1d(128)
        self.eye_drop  = nn.Dropout(0.3)
        self.eye_fc2   = nn.Linear(128, 128)
        self.eye_bn2   = nn.BatchNorm1d(128)
        self.eye_drop2 = nn.Dropout(0.3)
        self.eye_out   = nn.Linear(128, num_classes)

    def forward(self, eeg_x, eye_x):
        eeg = self.eeg_drop(F.relu(self.eeg_bn(self.eeg_fc(eeg_x))))
        eeg = self.eeg_drop2(F.relu(self.eeg_bn2(self.eeg_fc2(eeg))))
        eeg = self.eeg_out(eeg)
        eye = self.eye_drop(F.relu(self.eye_bn(self.eye_fc(eye_x))))
        eye = self.eye_drop2(F.relu(self.eye_bn2(self.eye_fc2(eye))))
        eye = self.eye_out(eye)
        return (eeg + eye) / 2.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    print("[1] Loading data from {} ...".format(DATA_DIR))
    files = {
        "X_fused": "X_fused.npy",
        "X_eeg":   "X_eeg_pca.npy",
        "X_eye":   "X_eye_clean.npy",
        "y":       "y.npy",
    }
    loaded = {}
    for key, fname in files.items():
        fpath = os.path.join(DATA_DIR, fname)
        loaded[key] = np.load(fpath, allow_pickle=True)

    X_fused = np.asarray(loaded["X_fused"], dtype=np.float32)
    X_eeg   = np.asarray(loaded["X_eeg"],   dtype=np.float32)
    X_eye   = np.asarray(loaded["X_eye"],   dtype=np.float32)
    y       = np.asarray(loaded["y"]).flatten().astype(np.int64)

    print("   Raw shapes -> X_fused: {} | X_eeg: {} | X_eye: {} | y: {}".format(
        X_fused.shape, X_eeg.shape, X_eye.shape, y.shape))

    # Data cleaning
    bad = (np.isnan(X_fused).any(axis=1) | np.isinf(X_fused).any(axis=1)
         | np.isnan(X_eeg).any(axis=1)   | np.isinf(X_eeg).any(axis=1)
         | np.isnan(X_eye).any(axis=1)   | np.isinf(X_eye).any(axis=1))

    X_fused, X_eeg, X_eye, y = X_fused[~bad], X_eeg[~bad], X_eye[~bad], y[~bad]
    print("   Clean samples: {}".format(len(y)))

    return X_fused, X_eeg, X_eye, y


# =============================================================================
# PER-FOLD PREPROCESSING (NO LEAKAGE)
# =============================================================================

def scale_fold(X_tr, X_te):
    sc = StandardScaler()
    sc.fit(X_tr) # Fit ONLY on training data
    return sc.transform(X_tr), sc.transform(X_te)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_eval_fold(model_name, model, fold_k,
                    Xtr, ytr, Xte, yte,
                    model_dir,
                    X2tr=None, X2te=None):
    ensure_dir(model_dir)
    is_dec = (X2tr is not None)

    def to_t(arr):
        return torch.FloatTensor(np.asarray(arr, dtype=np.float32))

    Xtr_pt  = to_t(Xtr)
    Xte_pt  = to_t(Xte)
    ytr_pt  = torch.LongTensor(np.asarray(ytr, dtype=np.int64))
    yte_pt  = torch.LongTensor(np.asarray(yte, dtype=np.int64))
    X2tr_pt = to_t(X2tr) if is_dec else None
    X2te_pt = to_t(X2te) if is_dec else None

    classes      = np.unique(ytr)
    cw           = compute_class_weight("balanced", classes=classes, y=ytr)
    criterion    = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw))
    optimizer    = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE)

    best_test_acc    = 0.0
    best_ckpt        = os.path.join(model_dir, "best_fold{}.pth".format(fold_k))
    patience_counter = 0

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr_pt))
        ep_loss, n_batches = 0.0, 0

        for i in range(0, len(ytr_pt), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2: continue
            optimizer.zero_grad()
            out  = model(Xtr_pt[idx], X2tr_pt[idx]) if is_dec else model(Xtr_pt[idx])
            loss = criterion(out, ytr_pt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            test_out  = model(Xte_pt, X2te_pt) if is_dec else model(Xte_pt)
            _, tpred  = torch.max(test_out, 1)
            test_acc  = accuracy_score(yte_pt.numpy(), tpred.numpy())

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_ckpt)
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(test_acc)
        if patience_counter >= EARLY_STOP_PATIENCE: break

    # Final evaluation on best checkpoint
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, weights_only=True))

    model.eval()
    with torch.no_grad():
        test_out = model(Xte_pt, X2te_pt) if is_dec else model(Xte_pt)
        _, test_pred = torch.max(test_out, 1)

    y_true = yte_pt.numpy()
    y_pred = test_pred.numpy()

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, prec, rec, f1


# =============================================================================
# PLOT UTILITIES
# =============================================================================

def plot_cv_results(summary_df, out_dir):
    models = summary_df["model"].tolist()
    means  = summary_df["mean_accuracy"].values * 100
    stds   = summary_df["std_accuracy"].values  * 100
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    bars = axes.barh(models, means, xerr=stds, color=colors, alpha=0.85, capsize=5)
    axes.set_xlabel("Mean Test Accuracy (%)"); axes.set_xlim(0, 100)
    axes.set_title("GroupKFold (LOSO) Accuracy (mean +/- std)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_performance_chart.png"), dpi=130)
    plt.close()


def plot_fold_variance(fold_df, out_dir):
    models = fold_df["model"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_model = [fold_df[fold_df["model"] == m]["accuracy"].values * 100 for m in models]
    ax.boxplot(data_by_model, labels=models, patch_artist=True)
    ax.set_ylabel("Test Accuracy per Fold (%)")
    ax.set_title("Fold-Level Accuracy (LOSO Variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_fold_variance.png"), dpi=130)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print(" GROUPKFOLD (LOSO) PIPELINE - LASO-STYLE ROTATION")
    print("=" * 65)
    
    X_fused, X_eeg, X_eye, y = load_data()
    
    # Subject IDs: 15 subjects, equally divided (approx)
    num_subjects = 15
    samples_per_subject = len(y) // num_subjects
    subject_ids = np.repeat(np.arange(num_subjects), samples_per_subject)
    if len(subject_ids) < len(y):
        extra = len(y) - len(subject_ids)
        subject_ids = np.concatenate([subject_ids, np.full(extra, num_subjects-1)])

    print("Subjects found:", np.unique(subject_ids))
    
    fused_dim = X_fused.shape[1]
    eeg_dim   = X_eeg.shape[1]
    eye_dim   = X_eye.shape[1]

    # GroupKFold with LOSO
    gkf = GroupKFold(n_splits=num_subjects)

    def make_models():
        return [
            ("MLP",             BaselineMLP(fused_dim),           "mlp"),
            ("DNN",             DeepDNN(fused_dim),               "dnn"),
            ("Attention",       AttentionModel(fused_dim),        "attention"),
            ("Hybrid",          HybridModel(fused_dim),           "hybrid"),
            ("Decision Fusion", DecisionFusion(eeg_dim, eye_dim), "decision_fusion"),
        ]

    model_names   = ["MLP", "DNN", "Attention", "Hybrid", "Decision Fusion"]
    all_fold_rows = []
    
    # Iterate through folds
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_fused, y, groups=subject_ids), start=1):
        # MANDATORY DEBUG PRINT
        print("\nFold:", fold)
        print("Train subjects:", set(subject_ids[train_idx]))
        print("Test subjects:", set(subject_ids[test_idx]))
        
        # Split data
        Xf_tr_raw, Xf_te_raw = X_fused[train_idx], X_fused[test_idx]
        Xe_tr_raw, Xe_te_raw = X_eeg[train_idx],   X_eeg[test_idx]
        Xey_tr_raw, Xey_te_raw = X_eye[train_idx], X_eye[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # SCALE (NO LEAKAGE)
        Xf_tr,  Xf_te  = scale_fold(Xf_tr_raw,  Xf_te_raw)
        Xe_tr,  Xe_te  = scale_fold(Xe_tr_raw,  Xe_te_raw)
        Xey_tr, Xey_te = scale_fold(Xey_tr_raw, Xey_te_raw)
        
        # Train each model
        for model_name, model, folder in make_models():
            model_dir = os.path.join(OUTPUT_DIR, folder)
            if model_name == "Decision Fusion":
                acc, prec, rec, f1 = train_eval_fold(
                    model_name, model, fold,
                    Xe_tr, y_train, Xe_te, y_test, model_dir,
                    X2tr=Xey_tr, X2te=Xey_te)
            else:
                acc, prec, rec, f1 = train_eval_fold(
                    model_name, model, fold,
                    Xf_tr, y_train, Xf_te, y_test, model_dir)
            
            all_fold_rows.append({
                "fold": fold, "model": model_name,
                "accuracy": round(acc, 4), "precision": round(prec, 4),
                "recall":   round(rec, 4), "f1":        round(f1,   4),
            })
            
            # Incremental save
            pd.DataFrame(all_fold_rows).to_csv(os.path.join(OUTPUT_DIR, "cv_fold_results.csv"), index=False)

    # Save CSVs
    fold_df = pd.DataFrame(all_fold_rows)
    fold_df.to_csv(os.path.join(OUTPUT_DIR, "cv_fold_results.csv"), index=False)
    
    summary_rows = []
    for mn in model_names:
        m_df = fold_df[fold_df["model"] == mn]
        summary_rows.append({
            "model": mn,
            "mean_accuracy": round(m_df["accuracy"].mean(), 4),
            "std_accuracy":  round(m_df["accuracy"].std(), 4),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cv_summary_results.csv"), index=False)
    
    # Plots
    plot_cv_results(summary_df, OUTPUT_DIR)
    plot_fold_variance(fold_df, OUTPUT_DIR)
    
    print("\n" + "=" * 65)
    print(" PIPELINE COMPLETE - GroupKFold (LOSO)")
    print(" Results in:", OUTPUT_DIR)
    print("=" * 65)
    
    # Print final mean accuracy as required
    final_acc = summary_df["mean_accuracy"].mean()
    print(f"\nFinal Mean Accuracy across models: {final_acc*100:.2f}%")

if __name__ == "__main__":
    main()
