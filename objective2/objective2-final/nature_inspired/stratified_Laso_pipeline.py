# -- coding: utf-8 --
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_REQUIRED_FILES = ["X_fused.npy", "X_eeg_pca.npy", "X_eye_clean.npy", "y.npy"]
_CANDIDATE_DIRS = [
    os.path.join(SCRIPT_DIR, "..", "processed_data"),                        # objective2-final/processed_data  ← actual location
    os.path.join(SCRIPT_DIR, "..", "objective2", "processed_data"),
    os.path.join(SCRIPT_DIR, "..", "stage4_pipeline", "processed_data"),
    os.path.join(SCRIPT_DIR, "processed_data"),
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

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "stage4_cv_stratified")
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
            if len(idx) < 2:
                continue
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


def plot_metric_comparison(summary_df, out_dir):
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1 for all models."""
    models  = summary_df["model"].tolist()
    metrics = ["mean_accuracy", "mean_precision", "mean_recall", "mean_f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-score"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    n_models  = len(models)
    n_metrics = len(metrics)
    x         = np.arange(n_models)
    bar_width  = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = summary_df[metric].values * 100
        offsets = x + (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(offsets, values, width=bar_width, label=label,
                      color=color, alpha=0.87, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    "{:.1f}".format(val), ha="center", va="bottom",
                    fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Evaluation Metrics Comparison — All Models",
                 fontsize=13, fontweight="bold")
    ax.axhline(25, color="gray", ls="--", lw=1, label="Chance (25%)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(out_dir, "evaluation_metrics_bargraph.png")
    plt.savefig(save_path, dpi=130)
    plt.close()
    print("  Saved: evaluation_metrics_bargraph.png")


# =============================================================================
# HYPERPARAMETER OPTIMIZATION — GA, PSO, WOA, GWO
# =============================================================================

# Hyperparameter search space
HP_SPACE = {
    "lr":           (1e-5,  5e-4),
    "dropout":      (0.2,   0.5),
    "hidden_dim":   [64, 128, 256],
    "batch_size":   [32, 64, 128],
    "weight_decay": (1e-6,  1e-3),
}
OPT_EPOCHS    = 15   # reduced epochs during optimisation search
OPT_SUBJECTS  = 5    # only first N subjects used during search
OPT_POP_SIZE  = 8
OPT_ITERS     = 10


def _decode(vec):
    """Map a continuous vector in [0,1]^5 to concrete hyperparameters."""
    lr_min,  lr_max  = HP_SPACE["lr"]
    dr_min,  dr_max  = HP_SPACE["dropout"]
    wd_min,  wd_max  = HP_SPACE["weight_decay"]
    lr          = lr_min  + vec[0] * (lr_max  - lr_min)
    dropout     = dr_min  + vec[1] * (dr_max  - dr_min)
    hidden_dim  = HP_SPACE["hidden_dim"][int(vec[2] * 2.999)]
    batch_size  = HP_SPACE["batch_size"][int(vec[3] * 2.999)]
    weight_decay = wd_min + vec[4] * (wd_max  - wd_min)
    return {"lr": lr, "dropout": dropout, "hidden_dim": hidden_dim,
            "batch_size": batch_size, "weight_decay": weight_decay}


def _clip(vec):
    return np.clip(vec, 0.0, 1.0)


def _quick_model(input_dim, hidden_dim, dropout, num_classes=4):
    """Lightweight 2-layer MLP for fast hyperparameter fitness evaluation."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def evaluate_hyperparams(hp, X_fused, y, subject_ids, fused_dim):
    """
    Quick 2-fold CV on first OPT_SUBJECTS subjects with OPT_EPOCHS.
    Returns mean accuracy (higher = better fitness).
    """
    opt_subjects = sorted(np.unique(subject_ids))[:OPT_SUBJECTS]
    mask = np.isin(subject_ids, opt_subjects)
    Xo, yo = X_fused[mask], y[mask]

    skf_opt = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    accs = []
    for tr_idx, te_idx in skf_opt.split(Xo, yo):
        Xtr_raw, Xte_raw = Xo[tr_idx], Xo[te_idx]
        ytr, yte         = yo[tr_idx], yo[te_idx]
        Xtr, Xte = scale_fold(Xtr_raw, Xte_raw)

        model = _quick_model(fused_dim, hp["hidden_dim"], hp["dropout"])
        classes = np.unique(ytr)
        cw      = compute_class_weight("balanced", classes=classes, y=ytr)
        crit    = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw))
        opt     = optim.AdamW(model.parameters(), lr=hp["lr"],
                              weight_decay=hp["weight_decay"])

        Xt = torch.FloatTensor(Xtr); Xv = torch.FloatTensor(Xte)
        yt = torch.LongTensor(ytr);  yv = torch.LongTensor(yte)
        bs = hp["batch_size"]

        for _ in range(OPT_EPOCHS):
            model.train()
            perm = torch.randperm(len(yt))
            for i in range(0, len(yt), bs):
                idx = perm[i:i+bs]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = crit(model(Xt[idx]), yt[idx])
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

        model.eval()
        with torch.no_grad():
            _, pred = torch.max(model(Xv), 1)
        accs.append(accuracy_score(yv.numpy(), pred.numpy()))

    return float(np.mean(accs)) if accs else 0.0


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------
def run_ga(X_fused, y, subject_ids, fused_dim):
    print("\n[GA] Starting Genetic Algorithm optimisation...")
    rng = np.random.RandomState(42)
    pop = rng.rand(OPT_POP_SIZE, 5)
    results = []

    for gen in range(OPT_ITERS):
        fitness = [evaluate_hyperparams(_decode(p), X_fused, y, subject_ids, fused_dim)
                   for p in pop]
        ranked  = np.argsort(fitness)[::-1]
        elites  = pop[ranked[:2]].copy()
        children = [elites[0], elites[1]]
        while len(children) < OPT_POP_SIZE:
            p1, p2 = pop[rng.choice(ranked[:4])], pop[rng.choice(ranked[:4])]
            cp  = rng.randint(1, 5)
            child = np.concatenate([p1[:cp], p2[cp:]])
            if rng.rand() < 0.2:
                child[rng.randint(5)] += rng.randn() * 0.1
            children.append(_clip(child))
        pop = np.array(children)
        best_hp = _decode(pop[0])
        best_fit = fitness[ranked[0]]
        results.append({"generation": gen+1, "best_accuracy": round(best_fit, 4),
                        **{k: round(v, 6) if isinstance(v, float) else v
                           for k, v in best_hp.items()}})
        print(f"  GA Gen {gen+1}/{OPT_ITERS}  best_acc={best_fit*100:.2f}%")

    best_vec = pop[0]
    best_hp  = _decode(best_vec)
    return best_hp, pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Particle Swarm Optimisation
# ---------------------------------------------------------------------------
def run_pso(X_fused, y, subject_ids, fused_dim):
    print("\n[PSO] Starting Particle Swarm optimisation...")
    rng = np.random.RandomState(43)
    pos = rng.rand(OPT_POP_SIZE, 5)
    vel = rng.randn(OPT_POP_SIZE, 5) * 0.1
    pbest     = pos.copy()
    pbest_fit = np.array([evaluate_hyperparams(_decode(p), X_fused, y, subject_ids, fused_dim)
                           for p in pos])
    gbest     = pbest[np.argmax(pbest_fit)].copy()
    gbest_fit = pbest_fit.max()
    results   = []

    w, c1, c2 = 0.7, 1.5, 1.5
    for it in range(OPT_ITERS):
        r1, r2 = rng.rand(OPT_POP_SIZE, 5), rng.rand(OPT_POP_SIZE, 5)
        vel    = w*vel + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)
        pos    = _clip(pos + vel)
        fits   = [evaluate_hyperparams(_decode(p), X_fused, y, subject_ids, fused_dim)
                  for p in pos]
        for i, f in enumerate(fits):
            if f > pbest_fit[i]:
                pbest_fit[i] = f; pbest[i] = pos[i].copy()
            if f > gbest_fit:
                gbest_fit = f; gbest = pos[i].copy()
        best_hp = _decode(gbest)
        results.append({"iteration": it+1, "best_accuracy": round(gbest_fit, 4),
                        **{k: round(v, 6) if isinstance(v, float) else v
                           for k, v in best_hp.items()}})
        print(f"  PSO Iter {it+1}/{OPT_ITERS}  best_acc={gbest_fit*100:.2f}%")

    return _decode(gbest), pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Whale Optimisation Algorithm
# ---------------------------------------------------------------------------
def run_woa(X_fused, y, subject_ids, fused_dim):
    print("\n[WOA] Starting Whale Optimisation Algorithm...")
    rng = np.random.RandomState(44)
    pos = rng.rand(OPT_POP_SIZE, 5)
    fits = [evaluate_hyperparams(_decode(p), X_fused, y, subject_ids, fused_dim)
            for p in pos]
    best_pos = pos[np.argmax(fits)].copy()
    best_fit = max(fits)
    results  = []

    for it in range(OPT_ITERS):
        a  = 2 - 2 * it / OPT_ITERS        # decreases 2→0
        a2 = -1 - it / OPT_ITERS           # decreases -1→-2
        for i in range(OPT_POP_SIZE):
            r, p_val = rng.rand(), rng.rand()
            A = 2*a*rng.rand(5) - a
            C = 2*rng.rand(5)
            if p_val < 0.5:
                if abs(A).max() < 1:
                    D = abs(C * best_pos - pos[i])
                    pos[i] = _clip(best_pos - A * D)
                else:
                    rand_pos = rng.rand(5)
                    D = abs(C * rand_pos - pos[i])
                    pos[i] = _clip(rand_pos - A * D)
            else:
                b, l = 1.0, rng.uniform(-1, 1)
                D    = abs(best_pos - pos[i])
                pos[i] = _clip(D * np.exp(b*l) * np.cos(2*np.pi*l) + best_pos)

            f = evaluate_hyperparams(_decode(pos[i]), X_fused, y, subject_ids, fused_dim)
            if f > best_fit:
                best_fit = f; best_pos = pos[i].copy()

        best_hp = _decode(best_pos)
        results.append({"iteration": it+1, "best_accuracy": round(best_fit, 4),
                        **{k: round(v, 6) if isinstance(v, float) else v
                           for k, v in best_hp.items()}})
        print(f"  WOA Iter {it+1}/{OPT_ITERS}  best_acc={best_fit*100:.2f}%")

    return _decode(best_pos), pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Grey Wolf Optimiser
# ---------------------------------------------------------------------------
def run_gwo(X_fused, y, subject_ids, fused_dim):
    print("\n[GWO] Starting Grey Wolf Optimisation...")
    rng = np.random.RandomState(45)
    pos = rng.rand(OPT_POP_SIZE, 5)
    fits = [evaluate_hyperparams(_decode(p), X_fused, y, subject_ids, fused_dim)
            for p in pos]
    order   = np.argsort(fits)[::-1]
    alpha, beta, delta = pos[order[0]].copy(), pos[order[1]].copy(), pos[order[2]].copy()
    alpha_fit = fits[order[0]]
    results   = []

    for it in range(OPT_ITERS):
        a = 2 - 2 * it / OPT_ITERS
        for i in range(OPT_POP_SIZE):
            new_pos = np.zeros(5)
            for leader in [alpha, beta, delta]:
                r1, r2 = rng.rand(5), rng.rand(5)
                A = 2*a*r1 - a; C = 2*r2
                D = abs(C * leader - pos[i])
                new_pos += leader - A * D
            pos[i] = _clip(new_pos / 3.0)
            f = evaluate_hyperparams(_decode(pos[i]), X_fused, y, subject_ids, fused_dim)
            fits[i] = f
        order = np.argsort(fits)[::-1]
        alpha, beta, delta = pos[order[0]].copy(), pos[order[1]].copy(), pos[order[2]].copy()
        alpha_fit = fits[order[0]]
        best_hp   = _decode(alpha)
        results.append({"iteration": it+1, "best_accuracy": round(alpha_fit, 4),
                        **{k: round(v, 6) if isinstance(v, float) else v
                           for k, v in best_hp.items()}})
        print(f"  GWO Iter {it+1}/{OPT_ITERS}  best_acc={alpha_fit*100:.2f}%")

    return _decode(alpha), pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Orchestrator — run all 4 and pick best
# ---------------------------------------------------------------------------
def run_optimization(X_fused, y, subject_ids, fused_dim):
    print("\n" + "="*65)
    print(" HYPERPARAMETER OPTIMISATION (GA / PSO / WOA / GWO)")
    print(" Using first {} subjects | {} epochs | {} pop | {} iters".format(
        OPT_SUBJECTS, OPT_EPOCHS, OPT_POP_SIZE, OPT_ITERS))
    print("="*65)

    ga_hp,  ga_df  = run_ga(X_fused,  y, subject_ids, fused_dim)
    pso_hp, pso_df = run_pso(X_fused, y, subject_ids, fused_dim)
    woa_hp, woa_df = run_woa(X_fused, y, subject_ids, fused_dim)
    gwo_hp, gwo_df = run_gwo(X_fused, y, subject_ids, fused_dim)

    # Save individual result CSVs
    ga_df.to_csv(os.path.join(OUTPUT_DIR,  "ga_results.csv"),  index=False)
    pso_df.to_csv(os.path.join(OUTPUT_DIR, "pso_results.csv"), index=False)
    woa_df.to_csv(os.path.join(OUTPUT_DIR, "woa_results.csv"), index=False)
    gwo_df.to_csv(os.path.join(OUTPUT_DIR, "gwo_results.csv"), index=False)
    print("\n  Saved: ga_results.csv, pso_results.csv, woa_results.csv, gwo_results.csv")

    # Evaluate each best on opt subset to pick winner
    scores = {}
    for name, hp in [("GA", ga_hp), ("PSO", pso_hp), ("WOA", woa_hp), ("GWO", gwo_hp)]:
        scores[name] = evaluate_hyperparams(hp, X_fused, y, subject_ids, fused_dim)

    best_opt = max(scores, key=scores.get)
    best_hp  = {"GA": ga_hp, "PSO": pso_hp, "WOA": woa_hp, "GWO": gwo_hp}[best_opt]

    # Save best hyperparameters CSV
    best_row = {"optimizer": best_opt, "opt_accuracy": round(scores[best_opt], 4)}
    best_row.update({k: round(v, 8) if isinstance(v, float) else v for k, v in best_hp.items()})
    pd.DataFrame([best_row]).to_csv(
        os.path.join(OUTPUT_DIR, "best_hyperparameters.csv"), index=False)
    print("  Saved: best_hyperparameters.csv")

    print("\n" + "="*65)
    print(" OPTIMISATION SUMMARY")
    print("="*65)
    for name, sc in scores.items():
        marker = "  <-- BEST" if name == best_opt else ""
        print("  {:5s}  opt_acc={:.2f}%{}".format(name, sc*100, marker))
    print("\n  Best Optimizer  : {}".format(best_opt))
    print("  Best Hyperparams:")
    for k, v in best_hp.items():
        print("    {:15s}: {}".format(k, v))
    print("="*65)

    return best_hp, best_opt


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
    # Rebuild subject IDs directly (ignore broken groups.npy)
    num_subjects = 15

    samples_per_subject = len(y) // num_subjects

    subject_ids = np.repeat(
        np.arange(num_subjects),
        samples_per_subject
    )

    # Fix remainder
    if len(subject_ids) < len(y):
        extra = len(y) - len(subject_ids)

        subject_ids = np.concatenate(
            [
                subject_ids,
                np.full(extra, num_subjects-1)
            ]
        )

    print("Subjects:", np.unique(subject_ids))
    fused_dim = X_fused.shape[1]
    eeg_dim   = X_eeg.shape[1]
    eye_dim   = X_eye.shape[1]
    print("\n   fused_dim={} | eeg_dim={} | eye_dim={}".format(
        fused_dim, eeg_dim, eye_dim))

    # ------------------------------------------------------------------
    # PHASE 1: Hyperparameter Optimisation (does NOT alter main pipeline)
    # ------------------------------------------------------------------
    best_hp, best_opt_name = run_optimization(
        X_fused, y, subject_ids, fused_dim)

    # This version uses StratifiedKFold (window-based splitting)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def make_models():
        return [
            ("MLP",             BaselineMLP(fused_dim),           "mlp"),
            ("DNN",             DeepDNN(fused_dim),               "dnn"),
            ("Attention",       AttentionModel(fused_dim),        "attention"),
            ("Attention+DNN",   HybridModel(fused_dim),           "hybrid"),
            ("Decision Fusion", DecisionFusion(eeg_dim, eye_dim), "decision_fusion"),
        ]

    model_names   = ["MLP", "DNN", "Attention", "Attention+DNN", "Decision Fusion"]
    all_fold_rows = []
    per_model     = {n: {"acc": [], "prec": [], "rec": [], "f1": []} for n in model_names}

    trial_accuracies = []

    for trial_subject in np.unique(subject_ids):

        print("\n"+"="*70)
        print(f"LOSO TRIAL {trial_subject+1}/15")
        print(f"Test Subject = {trial_subject}")
        print("="*70)

        all_fold_rows = []
        per_model = {
            n: {"acc": [], "prec": [], "rec": [], "f1": []}
            for n in model_names
        }

        loso_test_idx = np.where(
            subject_ids == trial_subject
        )[0]

        loso_train_idx = np.where(
            subject_ids != trial_subject
        )[0]

        X_fused_train = X_fused[loso_train_idx]
        X_eeg_train   = X_eeg[loso_train_idx]
        X_eye_train   = X_eye[loso_train_idx]
        y_train_all   = y[loso_train_idx]

    for fold_k, (train_idx, test_idx) in enumerate(
            skf.split(X_fused_train, y_train_all), start=1):
        print("\n" + "=" * 65)
        print(" FOLD {} / {} | Train: {}  Test: {}".format(
            fold_k, N_SPLITS, len(train_idx), len(test_idx)))

        Xf_tr_raw = X_fused_train[train_idx]
        Xf_te_raw = X_fused_train[test_idx]
        Xe_tr_raw = X_eeg_train[train_idx]
        Xe_te_raw = X_eeg_train[test_idx]
        Xey_tr_raw = X_eye_train[train_idx]
        Xey_te_raw = X_eye_train[test_idx]
        y_train = y_train_all[train_idx]
        y_test  = y_train_all[test_idx]

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
            trial_acc = np.mean(per_model["Attention+DNN"]["acc"])
            trial_accuracies.append(trial_acc)
            print(f"Trial {trial_subject+1} Accuracy: {trial_acc*100:.2f}%")

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
    plot_metric_comparison(summary_df, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # FINAL REPORT — best optimizer + pipeline summary metrics
    # ------------------------------------------------------------------
    overall_acc  = np.mean([r["mean_accuracy"]  for r in summary_rows])
    overall_prec = np.mean([r["mean_precision"] for r in summary_rows])
    overall_rec  = np.mean([r["mean_recall"]    for r in summary_rows])
    overall_f1   = np.mean([r["mean_f1"]        for r in summary_rows])

    print("\n" + "="*65)
    print(" FINAL REPORT")
    print("="*65)
    print("  Best Optimizer       : {}".format(best_opt_name))
    print("  Best Hyperparameters :")
    for k, v in best_hp.items():
        print("    {:15s}: {}".format(k, v))
    print()
    print("  Final Pipeline Metrics (mean across all folds/subjects):")
    print("    Accuracy  : {:.4f}  ({:.2f}%)".format(overall_acc,  overall_acc  * 100))
    print("    Precision : {:.4f}  ({:.2f}%)".format(overall_prec, overall_prec * 100))
    print("    Recall    : {:.4f}  ({:.2f}%)".format(overall_rec,  overall_rec  * 100))
    print("    F1-score  : {:.4f}  ({:.2f}%)".format(overall_f1,   overall_f1   * 100))
    print("="*65)

    print("\n" + "=" * 65)
    print(" PIPELINE COMPLETE — StratifiedKFold")
    print(" Outputs in: {}".format(OUTPUT_DIR))
    print("=" * 65)


if __name__ == "__main__":
    main()
