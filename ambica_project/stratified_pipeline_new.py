# -*- coding: utf-8 -*-
"""
=============================================================================
Objective 2 — LOSO + Inner StratifiedKFold Pipeline
=============================================================================
Outer loop : LOSO  — hold out ONE subject as test set
Inner loop : StratifiedKFold on remaining subjects (training / validation only)
Final eval : ONLY on the held-out LOSO subject — ONE result per (subject, model)

Outputs : subject_results.csv, summary_results.csv,
          cv_fold_variance.png, cv_performance_chart.png
=============================================================================
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

import io
import datetime

# Windows: reconfigure stdout to UTF-8 without double-wrapping
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # older Python fallback — skip


# ---------------------------------------------------------------------------
# Tee Logger — mirrors every print() to terminal AND a .log file
# ---------------------------------------------------------------------------
class _Tee:
    """Write to both the original stdout and a log file simultaneously."""
    def __init__(self, log_path, original_stdout):
        self._file   = open(log_path, "w", encoding="utf-8", buffering=1)
        self._stdout = original_stdout

    def write(self, msg):
        self._stdout.write(msg)
        self._stdout.flush()       # force immediate terminal output
        self._file.write(msg)
        self._file.flush()         # force immediate log write

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Forward everything else (e.g. fileno) to the real stdout
    def __getattr__(self, name):
        return getattr(self._stdout, name)


def _start_logging(log_dir):
    """Redirect stdout through _Tee and return the tee object."""
    os.makedirs(log_dir, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, "run_{}.log".format(ts))
    tee      = _Tee(log_path, sys.stdout)
    sys.stdout = tee
    print("[Logger] All output is being saved to: {}".format(log_path))
    return tee

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_REQUIRED = ["X_fused.npy", "X_eeg_pca.npy", "X_eye_clean.npy", "y.npy"]
_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "processed_data"),                                    # built by build_npy_from_csv.py (primary)
    os.path.join(_SCRIPT_DIR, "..", "..", "..", "stage4_pipeline", "processed_data"),
    os.path.join(_SCRIPT_DIR, "..", "..", "stage4_pipeline", "processed_data"),
]

DATA_DIR = None
for _d in _CANDIDATES:
    _d = os.path.normpath(os.path.abspath(_d))
    if os.path.isdir(_d) and all(os.path.isfile(os.path.join(_d, f)) for f in _REQUIRED):
        DATA_DIR = _d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Cannot find data folder with {}.\nSearched:\n{}".format(
            _REQUIRED, "\n".join("  " + os.path.normpath(os.path.abspath(d))
                                 for d in _CANDIDATES)))

OUTPUT_DIR = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "results", "stratified_loso"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters  (UNCHANGED)
# ---------------------------------------------------------------------------
EPOCHS              = 60
BATCH_SIZE          = 64
LR                  = 5e-5
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 10
SCHEDULER_PATIENCE  = 4
INNER_FOLDS         = 5
NUM_SUBJECTS        = 15
RANDOM_STATE        = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# =============================================================================
# MODEL ARCHITECTURES  (UNCHANGED)
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
    loaded = {}
    for key, fname in [("X_fused", "X_fused.npy"), ("X_eeg", "X_eeg_pca.npy"),
                       ("X_eye", "X_eye_clean.npy"), ("y", "y.npy")]:
        loaded[key] = np.load(os.path.join(DATA_DIR, fname), allow_pickle=True)

    X_fused = np.asarray(loaded["X_fused"], dtype=np.float32)
    X_eeg   = np.asarray(loaded["X_eeg"],   dtype=np.float32)
    X_eye   = np.asarray(loaded["X_eye"],   dtype=np.float32)
    y       = np.asarray(loaded["y"]).flatten().astype(np.int64)

    bad = (np.isnan(X_fused).any(axis=1) | np.isinf(X_fused).any(axis=1)
         | np.isnan(X_eeg).any(axis=1)   | np.isinf(X_eeg).any(axis=1)
         | np.isnan(X_eye).any(axis=1)   | np.isinf(X_eye).any(axis=1))

    X_fused, X_eeg, X_eye, y = X_fused[~bad], X_eeg[~bad], X_eye[~bad], y[~bad]
    print("   Clean samples: {}  |  Classes: {}".format(len(y), np.unique(y)))
    return X_fused, X_eeg, X_eye, y


# =============================================================================
# SCALING  — fit on train split ONLY, transform both (no leakage)
# =============================================================================

def scale_fold(X_tr, X_te):
    sc = StandardScaler()
    sc.fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te)


# =============================================================================
# INNER TRAINING  — StratifiedKFold on training data only
# LOSO test subject is NEVER seen here.
# Returns the model with the best inner-validation weights.
# =============================================================================

def train_with_inner_cv(model, model_name, subj,
                        Xf_train, Xe_train, Xey_train, y_train,
                        model_dir):
    ensure_dir(model_dir)
    is_dec = (model_name == "Decision Fusion")

    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    best_val_acc  = -1.0
    best_ckpt     = os.path.join(model_dir, "best_subj{}.pt".format(subj))

    # ---- Inner StratifiedKFold — training / validation only ----
    for inner_k, (tr_idx, val_idx) in enumerate(
            skf.split(Xf_train, y_train), start=1):

        # Slice inner train / val
        Xf_itr,  Xf_ival  = Xf_train[tr_idx],  Xf_train[val_idx]
        Xe_itr,  Xe_ival  = Xe_train[tr_idx],   Xe_train[val_idx]
        Xey_itr, Xey_ival = Xey_train[tr_idx],  Xey_train[val_idx]
        y_itr,   y_ival   = y_train[tr_idx],    y_train[val_idx]

        # Scale using inner training statistics only
        Xf_itr,  Xf_ival  = scale_fold(Xf_itr,  Xf_ival)
        Xe_itr,  Xe_ival  = scale_fold(Xe_itr,  Xe_ival)
        Xey_itr, Xey_ival = scale_fold(Xey_itr, Xey_ival)

        X_itr  = Xe_itr  if is_dec else Xf_itr
        X_ival = Xe_ival if is_dec else Xf_ival
        X2_itr  = Xey_itr  if is_dec else None
        X2_ival = Xey_ival if is_dec else None

        def t(a):
            return torch.FloatTensor(np.asarray(a, dtype=np.float32))

        Xtr_pt  = t(X_itr);   Xval_pt  = t(X_ival)
        ytr_pt  = torch.LongTensor(y_itr.astype(np.int64))
        yval_pt = torch.LongTensor(y_ival.astype(np.int64))
        X2tr_pt  = t(X2_itr)  if is_dec else None
        X2val_pt = t(X2_ival) if is_dec else None

        classes   = np.unique(y_itr)
        cw        = compute_class_weight("balanced", classes=classes, y=y_itr)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw))
        optimizer = optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE)

        patience_ctr  = 0
        fold_best_val = 0.0
        fold_ckpt = os.path.join(
            model_dir, "tmp_subj{}_fold{}.pt".format(subj, inner_k))

        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(ytr_pt))
            for i in range(0, len(ytr_pt), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                out  = (model(Xtr_pt[idx], X2tr_pt[idx]) if is_dec
                        else model(Xtr_pt[idx]))
                loss = criterion(out, ytr_pt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validate on inner val (NOT the LOSO test subject)
            model.eval()
            with torch.no_grad():
                val_out  = (model(Xval_pt, X2val_pt) if is_dec
                            else model(Xval_pt))
                _, vpred = torch.max(val_out, 1)
                val_acc  = accuracy_score(yval_pt.numpy(), vpred.numpy())

            if val_acc > fold_best_val:
                fold_best_val = val_acc
                torch.save(model.state_dict(), fold_ckpt)
                patience_ctr  = 0
            else:
                patience_ctr += 1

            scheduler.step(val_acc)
            if patience_ctr >= EARLY_STOP_PATIENCE:
                break

        # Keep the checkpoint if it's the best across all inner folds
        if fold_best_val > best_val_acc:
            best_val_acc = fold_best_val
            if os.path.exists(fold_ckpt):
                shutil.copy2(fold_ckpt, best_ckpt)

        # Remove temporary fold checkpoint
        if os.path.exists(fold_ckpt):
            os.remove(fold_ckpt)

    # Load the best inner-fold weights
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, weights_only=True))

    return model


# =============================================================================
# LOSO TEST EVALUATION — called ONCE per (subject, model)
# LOSO test subject ONLY. No fold data here.
# =============================================================================

def evaluate_loso(model, model_name,
                  Xf_test, Xe_test, Xey_test, y_test):
    is_dec = (model_name == "Decision Fusion")

    def t(a):
        return torch.FloatTensor(np.asarray(a, dtype=np.float32))

    Xf_pt  = t(Xf_test)
    Xe_pt  = t(Xe_test)
    Xey_pt = t(Xey_test)
    yte_pt = torch.LongTensor(y_test.astype(np.int64))

    model.eval()
    with torch.no_grad():
        out      = model(Xe_pt, Xey_pt) if is_dec else model(Xf_pt)
        _, pred  = torch.max(out, 1)

    y_true = yte_pt.numpy()
    y_pred = pred.numpy()

    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


# =============================================================================
# PLOTS
# =============================================================================

def plot_summary(summary_df, out_dir):
    models = summary_df["model"].tolist()
    means  = summary_df["mean_accuracy"].values * 100
    stds   = summary_df["std_accuracy"].values  * 100
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(models, means, xerr=stds, color=colors[:len(models)],
            alpha=0.85, capsize=5)
    ax.set_xlabel("Mean LOSO Test Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("LOSO + Inner StratifiedKFold — Mean Accuracy ± Std across Subjects")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_performance_chart.png"), dpi=130)
    plt.close()


def plot_variance(subject_df, out_dir):
    models = subject_df["model"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [subject_df[subject_df["model"] == m]["accuracy"].values * 100
            for m in models]
    ax.boxplot(data, labels=models, patch_artist=True)
    ax.set_ylabel("LOSO Test Accuracy per Subject (%)")
    ax.set_title("Subject-Level Accuracy Variance (LOSO)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_fold_variance.png"), dpi=130)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Start logging — all print() output goes to terminal AND a .log file
    _logger = _start_logging(OUTPUT_DIR)

    print("=" * 65)
    print(" LOSO + INNER StratifiedKFold PIPELINE")
    print(" Evaluation uses LOSO (subject-independent validation)")
    print("=" * 65)

    X_fused, X_eeg, X_eye, y = load_data()

    # Build subject ID array
    sps = len(y) // NUM_SUBJECTS
    subject_ids = np.repeat(np.arange(NUM_SUBJECTS), sps)
    if len(subject_ids) < len(y):
        subject_ids = np.concatenate(
            [subject_ids, np.full(len(y) - len(subject_ids), NUM_SUBJECTS - 1)])

    fused_dim = X_fused.shape[1]
    eeg_dim   = X_eeg.shape[1]
    eye_dim   = X_eye.shape[1]

    model_names = ["MLP", "DNN", "Attention", "Hybrid", "Decision Fusion"]

    def make_models():
        return [
            ("MLP",             BaselineMLP(fused_dim),           "mlp"),
            ("DNN",             DeepDNN(fused_dim),               "dnn"),
            ("Attention",       AttentionModel(fused_dim),        "attention"),
            ("Hybrid",          HybridModel(fused_dim),           "hybrid"),
            ("Decision Fusion", DecisionFusion(eeg_dim, eye_dim), "decision_fusion"),
        ]

    # ------------------------------------------------------------------
    # GLOBAL result containers — OUTSIDE LOSO loop (Issue 5 fix)
    # ------------------------------------------------------------------
    all_subject_rows = []   # one row per (subject, model) — Issue 4 fix

    # ==================================================================
    # OUTER LOOP: LOSO — hold out ONE subject at a time  (Issue 1 fix)
    # ==================================================================
    for subj in np.unique(subject_ids):

        test_mask  = (subject_ids == subj)
        train_mask = ~test_mask

        # LOSO test set — held-out subject, NEVER touched during training
        Xf_test  = X_fused[test_mask]
        Xe_test  = X_eeg[test_mask]
        Xey_test = X_eye[test_mask]
        y_test   = y[test_mask]

        # LOSO training pool — all other subjects
        Xf_train  = X_fused[train_mask]
        Xe_train  = X_eeg[train_mask]
        Xey_train = X_eye[train_mask]
        y_train   = y[train_mask]

        print("\n" + "=" * 70)
        print("  LOSO TRIAL {:02d}/{}  |  Test subject={}  |  "
              "Train={} samples  Test={} samples".format(
                  int(subj) + 1, NUM_SUBJECTS, subj,
                  len(y_train), len(y_test)))
        print("=" * 70)

        # Scale using LOSO training stats only (no test leakage)
        Xf_tr_s,  Xf_te_s  = scale_fold(Xf_train,  Xf_test)
        Xe_tr_s,  Xe_te_s  = scale_fold(Xe_train,  Xe_test)
        Xey_tr_s, Xey_te_s = scale_fold(Xey_train, Xey_test)

        # ==============================================================
        # Per-model: inner StratifiedKFold training → LOSO evaluation
        # (Issue 2 + 3 fix — both loops are INSIDE the subject loop)
        # ==============================================================
        for model_name, model, folder in make_models():
            model_dir = os.path.join(OUTPUT_DIR, folder)

            print("  Training {} for subject {:02d} ...".format(
                model_name, int(subj)))

            # INNER LOOP: StratifiedKFold on training data only (Issue 3)
            model = train_with_inner_cv(
                model, model_name, subj,
                Xf_tr_s, Xe_tr_s, Xey_tr_s, y_train,
                model_dir)

            # FINAL EVAL: ONCE on LOSO test subject (Issue 4 + 6 fix)
            acc, prec, rec, f1 = evaluate_loso(
                model, model_name,
                Xf_te_s, Xe_te_s, Xey_te_s, y_test)

            print("    Subject {:02d}  {}  ->  Accuracy: {:.4f}".format(
                int(subj), model_name, acc))

            # Append ONE row per (subject, model) — no duplicates
            all_subject_rows.append({
                "subject":   int(subj),
                "model":     model_name,
                "accuracy":  round(acc,  4),
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
            })

            # Incremental save after each model completes
            pd.DataFrame(all_subject_rows).to_csv(
                os.path.join(OUTPUT_DIR, "subject_results.csv"), index=False)

    # ==================================================================
    # OUTPUT A — subject_results.csv  (one row per subject per model)
    # ==================================================================
    subject_df = pd.DataFrame(all_subject_rows)
    subject_df.to_csv(os.path.join(OUTPUT_DIR, "subject_results.csv"), index=False)

    print("\n\n" + "=" * 65)
    print("  Subject | Model | Accuracy")
    print("=" * 65)
    pivot = subject_df.pivot(index="subject", columns="model", values="accuracy")
    print(pivot.to_string())

    # ==================================================================
    # OUTPUT B — summary_results.csv  (mean ± std across subjects)
    # ==================================================================
    summary_rows = []
    for mn in model_names:
        m_df = subject_df[subject_df["model"] == mn]
        summary_rows.append({
            "model":          mn,
            "mean_accuracy":  round(m_df["accuracy"].mean(),  4),
            "std_accuracy":   round(m_df["accuracy"].std(),   4),
            "mean_precision": round(m_df["precision"].mean(), 4),
            "mean_recall":    round(m_df["recall"].mean(),    4),
            "mean_f1":        round(m_df["f1"].mean(),        4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)

    print("\n\n" + "=" * 65)
    print("  Model | Mean Accuracy | Std")
    print("=" * 65)
    for row in summary_rows:
        print("  {:20s}  Mean: {:.4f}   Std: {:.4f}".format(
            row["model"], row["mean_accuracy"], row["std_accuracy"]))

    # ==================================================================
    # OUTPUT C — mandatory statement
    # ==================================================================
    print("\nEvaluation uses LOSO (subject-independent validation)")

    plot_summary(summary_df, OUTPUT_DIR)
    plot_variance(subject_df, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE -- Results in:", OUTPUT_DIR)
    print("=" * 65)

    # Flush and close log file
    log_path = _logger._file.name
    _logger.flush()
    _logger.close()
    sys.stdout = _logger._stdout   # restore original stdout
    print("\n[Logger] Run log saved to: {}".format(log_path))


if __name__ == "__main__":
    main()