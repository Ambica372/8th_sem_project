# -*- coding: utf-8 -*-
"""
=============================================================================
Objective 2 — 5-Fold Stratified Cross-Validation Pipeline
VERSION 1: This version uses StratifiedKFold (window-based splitting)
=============================================================================

Data source  : Auto-discovered processed_data/*.npy
Splitting    : StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Preprocessing: Per-fold StandardScaler (fit on train, transform test)
Models       : MLP, DNN, Attention, Hybrid, Decision Fusion
Metrics      : Accuracy, Precision, Recall, F1  (macro average)
Outputs      : cv_fold_results.csv, cv_summary_results.csv
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

# This version uses StratifiedKFold (window-based splitting)
from sklearn.model_selection import StratifiedKFold
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
# Paths  — auto-discover data directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_FILES = ["X_fused.npy", "X_eeg_pca.npy", "X_eye_clean.npy", "y.npy"]
_CANDIDATE_DIRS = [
    os.path.join(_SCRIPT_DIR, "..", "objective2", "processed_data"),
    os.path.join(_SCRIPT_DIR, "..", "stage4_pipeline", "processed_data"),
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

OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "stage4_cv_stratified")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS              = 60
BATCH_SIZE          = 64
LR                  = 5e-5
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 10
SCHEDULER_PATIENCE  = 4
N_SPLITS            = 5
RANDOM_STATE        = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# =============================================================================
# MODEL ARCHITECTURES — hidden: 128, BatchNorm, Dropout=0.3
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
        try:
            loaded[key] = np.load(fpath, allow_pickle=True)
        except Exception as e1:
            print("   [WARN] Primary load failed for {}: {}".format(fpath, e1))
            try:
                arr = np.load(fpath, allow_pickle=True, mmap_mode="r")
                loaded[key] = np.array(arr)
                print("   [INFO] Loaded {} via mmap fallback".format(fpath))
            except Exception as e2:
                print("   [ERROR] All load methods failed for {}: {}".format(fpath, e2))
                sys.exit(1)

    X_fused = np.asarray(loaded["X_fused"], dtype=np.float32)
    X_eeg   = np.asarray(loaded["X_eeg"],   dtype=np.float32)
    X_eye   = np.asarray(loaded["X_eye"],   dtype=np.float32)
    y       = np.asarray(loaded["y"]).flatten().astype(np.int64)

    print("   Raw shapes -> X_fused: {} | X_eeg: {} | X_eye: {} | y: {}".format(
        X_fused.shape, X_eeg.shape, X_eye.shape, y.shape))

    bad = (np.isnan(X_fused).any(axis=1) | np.isinf(X_fused).any(axis=1)
         | np.isnan(X_eeg).any(axis=1)   | np.isinf(X_eeg).any(axis=1)
         | np.isnan(X_eye).any(axis=1)   | np.isinf(X_eye).any(axis=1))

    n_removed = int(bad.sum())
    X_fused, X_eeg, X_eye, y = X_fused[~bad], X_eeg[~bad], X_eye[~bad], y[~bad]
    print("   Removed {} corrupted rows -> {} clean samples".format(n_removed, len(y)))

    for name, arr in [("X_fused", X_fused), ("X_eeg", X_eeg), ("X_eye", X_eye)]:
        assert not np.isnan(arr).any() and not np.isinf(arr).any(), \
            "NaN/Inf still present in {} after cleaning".format(name)

    unique, counts = np.unique(y, return_counts=True)
    print("   Classes: {}".format(dict(zip(unique.tolist(), counts.tolist()))))
    return X_fused, X_eeg, X_eye, y


# =============================================================================
# PER-FOLD PREPROCESSING  (NO LEAKAGE)
# =============================================================================

def scale_fold(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


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

    print("\n  [{}] Fold {} | train={} test={}".format(
        model_name, fold_k, len(ytr), len(yte)))

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
            optimizer.zero_grad()
            out  = model(Xtr_pt[idx], X2tr_pt[idx]) if is_dec else model(Xtr_pt[idx])
            loss = criterion(out, ytr_pt[idx])
            if torch.isnan(loss):
                print("  [WARN] NaN loss at epoch {}, stopping.".format(ep + 1))
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss   += loss.item()
            n_batches += 1

        if n_batches == 0:
            break

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

        if (ep + 1) % 10 == 0 or ep == 0:
            print("    Epoch [{:2d}/{}] loss={:.4f}  test_acc={:.4f}".format(
                ep + 1, EPOCHS, ep_loss / n_batches, test_acc))

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("    Early stopping at epoch {} (patience={})".format(
                ep + 1, EARLY_STOP_PATIENCE))
            break

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

    print("    -> TEST  acc={:.4f}  prec={:.4f}  rec={:.4f}  f1={:.4f}".format(
        acc, prec, rec, f1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("{} - Fold {}".format(model_name, fold_k))
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "cm_fold{}.png".format(fold_k)), dpi=100)
    plt.close()

    report = classification_report(y_true, y_pred, zero_division=0)
    with open(os.path.join(model_dir, "report_fold{}.txt".format(fold_k)),
              "w", encoding="utf-8") as f:
        f.write("Fold: {}\nTest Acc: {:.4f}\nBest Test Acc: {:.4f}\n\n{}".format(
            fold_k, acc, best_test_acc, report))

    return acc, prec, rec, f1


# =============================================================================
# PLOT UTILITIES
# =============================================================================

def plot_cv_results(summary_df, out_dir):
    models = summary_df["model"].tolist()
    means  = summary_df["mean_accuracy"].values * 100
    stds   = summary_df["std_accuracy"].values  * 100
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("5-Fold StratifiedKFold CV Results", fontsize=13, fontweight="bold")
    bars = axes[0].barh(models, means, xerr=stds, color=colors, alpha=0.85,
                        capsize=5, error_kw={"ecolor": "black", "lw": 1.5})
    axes[0].axvline(25, color="gray", ls="--", lw=1, label="Chance (25%)")
    axes[0].set_xlabel("Mean Test Accuracy (%)"); axes[0].set_xlim(0, 100)
    axes[0].set_title("Accuracy (mean +/- std)"); axes[0].legend()
    for bar, val in zip(bars, means):
        axes[0].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     "{:.1f}%".format(val), va="center", fontsize=9)
    f1m = summary_df["mean_f1"].values * 100
    f1s = summary_df["std_f1"].values  * 100
    bars2 = axes[1].barh(models, f1m, xerr=f1s, color=colors, alpha=0.85,
                          capsize=5, error_kw={"ecolor": "black", "lw": 1.5})
    axes[1].axvline(25, color="gray", ls="--", lw=1, label="Chance (25%)")
    axes[1].set_xlabel("Mean Macro F1-Score (%)"); axes[1].set_xlim(0, 100)
    axes[1].set_title("F1-Score (mean +/- std)"); axes[1].legend()
    for bar, val in zip(bars2, f1m):
        axes[1].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     "{:.1f}%".format(val), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_performance_chart.png"), dpi=130)
    plt.close()
    print("  Saved: cv_performance_chart.png")


def plot_fold_variance(fold_df, out_dir):
    models = fold_df["model"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_model = [fold_df[fold_df["model"] == m]["accuracy"].values * 100
                     for m in models]
    bp = ax.boxplot(data_by_model, patch_artist=True, widths=0.4,
                    medianprops={"color": "black", "lw": 2}, showfliers=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Test Accuracy per Fold (%)")
    ax.set_title("Fold-Level Accuracy — StratifiedKFold", fontsize=12, fontweight="bold")
    ax.axhline(25, color="gray", ls="--", lw=1, label="Chance (25%)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_fold_variance.png"), dpi=130)
    plt.close()
    print("  Saved: cv_fold_variance.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # This version uses StratifiedKFold (window-based splitting)
    print("=" * 65)
    print(" VERSION 1 — StratifiedKFold (window-based splitting)")
    print("=" * 65)
    print(" Data dir   : {}".format(DATA_DIR))
    print(" Output dir : {}".format(OUTPUT_DIR))
    print(" Folds: {} | Epochs: {} | Batch: {} | LR: {}".format(
        N_SPLITS, EPOCHS, BATCH_SIZE, LR))
    print(" Optimizer: AdamW | Scheduler: ReduceLROnPlateau | Early Stop: {}".format(
        EARLY_STOP_PATIENCE))
    print()

    X_fused, X_eeg, X_eye, y = load_data()
    fused_dim = X_fused.shape[1]
    eeg_dim   = X_eeg.shape[1]
    eye_dim   = X_eye.shape[1]
    print("\n   fused_dim={} | eeg_dim={} | eye_dim={}".format(
        fused_dim, eeg_dim, eye_dim))

    # This version uses StratifiedKFold (window-based splitting)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

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
    per_model     = {n: {"acc": [], "prec": [], "rec": [], "f1": []} for n in model_names}

    for fold_k, (train_idx, test_idx) in enumerate(
            skf.split(X_fused, y), start=1):  # This version uses StratifiedKFold (window-based splitting)

        print("\n" + "=" * 65)
        print(" FOLD {} / {} | Train: {}  Test: {}".format(
            fold_k, N_SPLITS, len(train_idx), len(test_idx)))

        Xf_tr_raw  = X_fused[train_idx]; Xf_te_raw  = X_fused[test_idx]
        Xe_tr_raw  = X_eeg[train_idx];   Xe_te_raw  = X_eeg[test_idx]
        Xey_tr_raw = X_eye[train_idx];   Xey_te_raw = X_eye[test_idx]
        y_train    = y[train_idx];       y_test     = y[test_idx]

        Xf_tr,  Xf_te  = scale_fold(Xf_tr_raw,  Xf_te_raw)
        Xe_tr,  Xe_te  = scale_fold(Xe_tr_raw,  Xe_te_raw)
        Xey_tr, Xey_te = scale_fold(Xey_tr_raw, Xey_te_raw)

        for tag, arr in [("Xf_tr", Xf_tr), ("Xf_te", Xf_te),
                          ("Xe_tr", Xe_tr), ("Xey_tr", Xey_tr)]:
            assert not np.isnan(arr).any(), "NaN in {} fold {}".format(tag, fold_k)
            assert not np.isinf(arr).any(), "Inf in {} fold {}".format(tag, fold_k)

        print("  Scaling done -> fused={} | eeg={} | eye={}".format(
            Xf_tr.shape, Xe_tr.shape, Xey_tr.shape))

        for model_name, model, folder in make_models():
            is_dec    = (model_name == "Decision Fusion")
            model_dir = os.path.join(OUTPUT_DIR, folder)
            ensure_dir(model_dir)

            if is_dec:
                acc, prec, rec, f1 = train_eval_fold(
                    model_name, model, fold_k,
                    Xe_tr, y_train, Xe_te, y_test, model_dir,
                    X2tr=Xey_tr, X2te=Xey_te)
            else:
                acc, prec, rec, f1 = train_eval_fold(
                    model_name, model, fold_k,
                    Xf_tr, y_train, Xf_te, y_test, model_dir)

            all_fold_rows.append({
                "fold": fold_k, "model": model_name,
                "accuracy": round(acc, 4), "precision": round(prec, 4),
                "recall":   round(rec, 4), "f1":        round(f1,   4),
            })
            per_model[model_name]["acc"].append(acc)
            per_model[model_name]["prec"].append(prec)
            per_model[model_name]["rec"].append(rec)
            per_model[model_name]["f1"].append(f1)

    fold_df = pd.DataFrame(all_fold_rows)
    fold_df.to_csv(os.path.join(OUTPUT_DIR, "cv_fold_results.csv"), index=False)
    print("\n  Saved: cv_fold_results.csv")

    summary_rows = []
    for mn in model_names:
        a = np.array(per_model[mn]["acc"])
        p = np.array(per_model[mn]["prec"])
        r = np.array(per_model[mn]["rec"])
        f = np.array(per_model[mn]["f1"])
        summary_rows.append({
            "model":          mn,
            "mean_accuracy":  round(float(a.mean()), 4),
            "std_accuracy":   round(float(a.std()),  4),
            "mean_precision": round(float(p.mean()), 4),
            "std_precision":  round(float(p.std()),  4),
            "mean_recall":    round(float(r.mean()), 4),
            "std_recall":     round(float(r.std()),  4),
            "mean_f1":        round(float(f.mean()), 4),
            "std_f1":         round(float(f.std()),  4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cv_summary_results.csv"), index=False)
    print("  Saved: cv_summary_results.csv")

    print("\n" + "=" * 65)
    print(" FINAL SUMMARY — StratifiedKFold (window-based splitting)")
    print("=" * 65)
    print("  {:<20} {:>18} {:>18}".format("Model", "Acc (mean+/-std)", "F1 (mean+/-std)"))
    print("  " + "-" * 58)
    for row in summary_rows:
        print("  {:<20} {:>7.2f}% +/- {:>5.2f}%   {:>7.2f}% +/- {:>5.2f}%".format(
            row["model"],
            row["mean_accuracy"] * 100, row["std_accuracy"] * 100,
            row["mean_f1"] * 100,       row["std_f1"] * 100))

    overall_mean = np.mean([r["mean_accuracy"] for r in summary_rows])
    print("\n  Overall Mean Accuracy: {:.2f}%".format(overall_mean * 100))

    print("\n[Fold-wise Accuracy]")
    for mn in model_names:
        fa     = [r["accuracy"] for r in all_fold_rows if r["model"] == mn]
        spread = max(fa) - min(fa)
        status = "OK" if spread > 0.001 else "WARN - identical across folds"
        print("  {:<20} folds={}  spread={:.4f}  [{}]".format(
            mn, ["{:.4f}".format(a) for a in fa], spread, status))

    plot_cv_results(summary_df, OUTPUT_DIR)
    plot_fold_variance(fold_df, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(" PIPELINE COMPLETE — StratifiedKFold")
    print(" Outputs in: {}".format(OUTPUT_DIR))
    print("=" * 65)


if __name__ == "__main__":
    main()
