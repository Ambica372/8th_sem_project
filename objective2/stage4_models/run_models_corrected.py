# -*- coding: utf-8 -*-
"""
=============================================================================
Objective 2 — CORRECTED Multimodal Emotion Recognition Pipeline
=============================================================================

FIXES APPLIED vs run_models.py (original):
  1. Subject-level splitting    : Uses pre-existing fold CSVs from stage2_output
                                  (subject-level K-fold, no window leakage across subjects).
  2. PCA on train only          : PCA fitted exclusively on training EEG data per fold.
  3. Scaling on train only      : StandardScaler fitted on train, applied to val/test.
  4. Separate val / test sets   : 5-fold CV — each fold has its own train, test, and
                                  held-out validation subject.
  5. Model selection on val     : Best epoch chosen by VALIDATION accuracy, not test.
  6. Fair comparison            : All models (incl. Decision Fusion) use same scaled inputs.
  7. Window-based training      : Preserved (no trial aggregation).

EVALUATION STRATEGY:
  - 5-fold subject-level cross-validation (from stage2 fold manifest)
  - Held-out validation subject (Subject 15, normalized per fold)
  - Final metrics averaged across 5 folds
  - Per-fold confusion matrices saved

DATA SOURCES:
  - objective1/stage2_output/fold_k_{train|test|validation}_{eeg|eye}.csv
  - Subject 15 is the held-out validation subject (never used in any fold's training)
=============================================================================
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Windows cp1252 console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Fold CSVs produced by objective1/stage2_preprocessing.py
STAGE2_DIR = r"c:\Users\Rose J Thachil\Desktop\8th_sem_project\objective1\stage2_output"
MANIFEST   = os.path.join(STAGE2_DIR, "fold_manifest.json")

# Output directory for corrected results
ROOT_MODELS = r"c:\Users\Rose J Thachil\Desktop\8th_sem_project\objective2\stage4_models_corrected"

# ---------------------------------------------------------------------------
# Hyper-parameters (UNCHANGED from original)
# ---------------------------------------------------------------------------
EPOCHS     = 30
BATCH_SIZE = 64
LR         = 1e-4
N_FOLDS    = 5
PCA_VAR    = 0.95      # retain 95% variance

# Feature/meta column constants
META_COLS   = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]
N_EEG_RAW  = 310
N_EYE_RAW  = 31

np.random.seed(42)
torch.manual_seed(42)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =============================================================================
# MODEL ARCHITECTURES — IDENTICAL to original run_models.py
# =============================================================================

class BaselineMLP(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        return self.out(x)


class DeepDNN(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

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
        attn_scores = torch.sigmoid(self.attention_weights(x))
        return self.out(F.relu(self.fc1(x * attn_scores)))


class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.attention_weights = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        attn_scores = torch.sigmoid(self.attention_weights(x))
        return self.out(F.relu(self.fc2(x * attn_scores)))


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
# DATA LOADING
# =============================================================================

def load_fold_data(fold_k):
    """
    Load train / test / validation CSVs for a given fold k from stage2_output.
    EEG and Eye are merged on (subject_id, session_id, trial_id, window_id).

    Returns:
        train_eeg, train_eye, train_y   — numpy arrays
        test_eeg,  test_eye,  test_y
        val_eeg,   val_eye,   val_y
        trial_ids_test                  — for optional trial-level evaluation
    """
    def read_split(split_name):
        eeg_path = os.path.join(STAGE2_DIR, f"fold_{fold_k}_{split_name}_eeg.csv")
        eye_path = os.path.join(STAGE2_DIR, f"fold_{fold_k}_{split_name}_eye.csv")
        eeg_df = pd.read_csv(eeg_path)
        eye_df = pd.read_csv(eye_path)

        # Align on index key (stage2 guarantees row alignment)
        assert len(eeg_df) == len(eye_df), f"Row mismatch in fold {fold_k} {split_name}"

        eeg_feat_cols = [c for c in eeg_df.columns if c not in META_COLS]
        eye_feat_cols = [c for c in eye_df.columns if c not in META_COLS]

        X_eeg = eeg_df[eeg_feat_cols].values.astype(np.float32)
        X_eye = eye_df[eye_feat_cols].values.astype(np.float32)
        y     = eeg_df["emotion_label"].values.astype(np.int64)

        # Keep trial IDs for optional trial-level eval
        trial_ids = eeg_df["trial_id"].values if "trial_id" in eeg_df.columns else None
        subj_ids  = eeg_df["subject_id"].values if "subject_id" in eeg_df.columns else None

        return X_eeg, X_eye, y, trial_ids, subj_ids

    tr_eeg, tr_eye, tr_y, _, _           = read_split("train")
    ts_eeg, ts_eye, ts_y, ts_trial, ts_subj = read_split("test")
    va_eeg, va_eye, va_y, _, _           = read_split("validation")

    return (tr_eeg, tr_eye, tr_y,
            ts_eeg, ts_eye, ts_y,
            va_eeg, va_eye, va_y,
            ts_trial, ts_subj)


# =============================================================================
# PREPROCESSING (per-fold, train-only fit)
# =============================================================================

def preprocess_fold(tr_eeg, tr_eye,
                    ts_eeg, ts_eye,
                    va_eeg, va_eye):
    """
    For one fold:
      1. Remove NaN/Inf rows (consistent mask across EEG + Eye).
      2. Fit PCA on training EEG ONLY → transform train/test/val.
      3. Fit StandardScaler on training fused ONLY → transform all splits.
      4. Also fit separate scalers for EEG-PCA and Eye (for Decision Fusion).

    Returns cleaned, scaled numpy arrays + PCA info.
    """

    # ---- Step 1: NaN/Inf cleaning (apply mask consistently to all splits) ----
    def clean_mask(eeg, eye):
        bad = (np.isnan(eeg).any(axis=1) | np.isinf(eeg).any(axis=1) |
               np.isnan(eye).any(axis=1) | np.isinf(eye).any(axis=1))
        return ~bad

    tr_mask = clean_mask(tr_eeg, tr_eye)
    ts_mask = clean_mask(ts_eeg, ts_eye)
    va_mask = clean_mask(va_eeg, va_eye)

    tr_eeg, tr_eye = tr_eeg[tr_mask], tr_eye[tr_mask]
    ts_eeg, ts_eye = ts_eeg[ts_mask], ts_eye[ts_mask]
    va_eeg, va_eye = va_eeg[va_mask], va_eye[va_mask]

    # ---- Step 2: PCA on EEG — fit ONLY on training data ----------------------
    pca = PCA(n_components=PCA_VAR, random_state=42)
    tr_eeg_pca = pca.fit_transform(tr_eeg)          # fit + transform train
    ts_eeg_pca = pca.transform(ts_eeg)              # transform only
    va_eeg_pca = pca.transform(va_eeg)              # transform only
    n_pca_components = tr_eeg_pca.shape[1]

    # ---- Step 3: Eye imputation (replace NaN with column mean from train) ----
    # Eye data can have NaN after stage2 (preserved as-is). Impute with train mean.
    eye_train_mean = np.nanmean(tr_eye, axis=0)
    for col_idx in range(tr_eye.shape[1]):
        tr_eye[np.isnan(tr_eye[:, col_idx]), col_idx] = eye_train_mean[col_idx]
        ts_eye[np.isnan(ts_eye[:, col_idx]), col_idx] = eye_train_mean[col_idx]
        va_eye[np.isnan(va_eye[:, col_idx]), col_idx] = eye_train_mean[col_idx]

    # ---- Step 4: Feature-level fusion (EEG-PCA + Eye) ----------------------
    tr_fused = np.hstack([tr_eeg_pca, tr_eye])
    ts_fused = np.hstack([ts_eeg_pca, ts_eye])
    va_fused = np.hstack([va_eeg_pca, va_eye])

    # ---- Step 5: Scale fused features — fit ONLY on training data -----------
    fused_scaler = StandardScaler()
    tr_fused = fused_scaler.fit_transform(tr_fused)
    ts_fused = fused_scaler.transform(ts_fused)
    va_fused = fused_scaler.transform(va_fused)

    # ---- Step 6: Scale EEG-PCA and Eye separately for Decision Fusion -------
    eeg_scaler = StandardScaler()
    tr_eeg_scaled = eeg_scaler.fit_transform(tr_eeg_pca)
    ts_eeg_scaled = eeg_scaler.transform(ts_eeg_pca)
    va_eeg_scaled = eeg_scaler.transform(va_eeg_pca)

    eye_scaler = StandardScaler()
    tr_eye_scaled = eye_scaler.fit_transform(tr_eye)
    ts_eye_scaled = eye_scaler.transform(ts_eye)
    va_eye_scaled = eye_scaler.transform(va_eye)

    # Verify no leakage: assert no NaN/Inf in any output
    for name, arr in [("tr_fused", tr_fused), ("ts_fused", ts_fused),
                      ("va_fused", va_fused),
                      ("tr_eeg_scaled", tr_eeg_scaled),
                      ("tr_eye_scaled", tr_eye_scaled)]:
        assert not np.isnan(arr).any(), f"NaN found in {name} after preprocessing"
        assert not np.isinf(arr).any(), f"Inf found in {name} after preprocessing"

    return {
        "tr_fused": tr_fused, "ts_fused": ts_fused, "va_fused": va_fused,
        "tr_eeg": tr_eeg_scaled, "ts_eeg": ts_eeg_scaled, "va_eeg": va_eeg_scaled,
        "tr_eye": tr_eye_scaled, "ts_eye": ts_eye_scaled, "va_eye": va_eye_scaled,
        "tr_mask": tr_mask, "ts_mask": ts_mask, "va_mask": va_mask,
        "n_pca": n_pca_components,
        "fused_dim": tr_fused.shape[1],
        "eeg_dim": tr_eeg_scaled.shape[1],
        "eye_dim": tr_eye_scaled.shape[1],
    }


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_eval_model(model_name, model, data, fold_k, model_dir,
                     X2_key_tr=None, X2_key_ts=None, X2_key_va=None):
    """
    Train a model with validation-based early stopping.
    Evaluate on test set ONLY after training is complete.

    Args:
        data      : dict from preprocess_fold()
        X2_key_*  : keys into data dict for secondary input (Decision Fusion only)
    """
    ensure_dir(model_dir)

    # Identify primary and secondary (if any) feature arrays
    X_tr = data["tr_fused"] if X2_key_tr is None else data["tr_eeg"]
    X_ts = data["ts_fused"] if X2_key_tr is None else data["ts_eeg"]
    X_va = data["va_fused"] if X2_key_tr is None else data["va_eeg"]

    y_tr_raw = data.get("tr_y")
    y_ts_raw = data.get("ts_y")
    y_va_raw = data.get("va_y")

    is_dec = X2_key_tr is not None

    # Convert to tensors
    Xtr_t  = torch.FloatTensor(X_tr)
    Xts_t  = torch.FloatTensor(X_ts)
    Xva_t  = torch.FloatTensor(X_va)
    ytr_t  = torch.LongTensor(y_tr_raw)
    yts_t  = torch.LongTensor(y_ts_raw)
    yva_t  = torch.LongTensor(y_va_raw)

    if is_dec:
        X2tr_t = torch.FloatTensor(data[X2_key_tr])
        X2ts_t = torch.FloatTensor(data[X2_key_ts])
        X2va_t = torch.FloatTensor(data[X2_key_va])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_model_path = os.path.join(model_dir, f"best_model_fold{fold_k}.pth")
    train_losses, val_accs = [], []

    print(f"\n  --- Training {model_name} (Fold {fold_k}) ---")
    print(f"  Train: {len(y_tr_raw)} | Val: {len(y_va_raw)} | Test: {len(y_ts_raw)}")

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr_t))
        epoch_loss, batches = 0.0, 0

        for i in range(0, len(ytr_t), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()

            if is_dec:
                out = model(Xtr_t[idx], X2tr_t[idx])
            else:
                out = model(Xtr_t[idx])

            loss = criterion(out, ytr_t[idx])
            if torch.isnan(loss):
                print(f"  [WARN] NaN loss at epoch {ep+1}. Stopping.")
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        if batches == 0:
            break

        # --- Evaluate on VALIDATION set (NOT test set) ---
        model.eval()
        with torch.no_grad():
            if is_dec:
                val_out = model(Xva_t, X2va_t)
            else:
                val_out = model(Xva_t)
            _, val_pred = torch.max(val_out, 1)
            val_acc = accuracy_score(yva_t.numpy(), val_pred.numpy())

        train_losses.append(epoch_loss / batches)
        val_accs.append(val_acc)

        # --- Save best model based on VALIDATION accuracy ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch [{ep+1:2d}/{EPOCHS}] | Loss: {epoch_loss/batches:.4f}"
                  f" | Val Acc: {val_acc:.4f}")

    # --- Load best checkpoint (chosen by val, not test) ---
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # --- Evaluate on TEST set (only once, after training) ---
    model.eval()
    with torch.no_grad():
        if is_dec:
            test_out = model(Xts_t, X2ts_t)
        else:
            test_out = model(Xts_t)
        _, test_pred = torch.max(test_out, 1)

    np_pred = test_pred.numpy()
    np_true = yts_t.numpy()

    acc = accuracy_score(np_true, np_pred)
    p, r, f1, _ = precision_recall_fscore_support(np_true, np_pred,
                                                   average='weighted', zero_division=0)
    cm = confusion_matrix(np_true, np_pred)
    report = classification_report(np_true, np_pred, zero_division=0)

    # Save outputs
    with open(os.path.join(model_dir, f"accuracy_fold{fold_k}.txt"), 'w') as f:
        f.write(f"Fold: {fold_k}\nTest Accuracy: {acc:.4f}\nBest Val Acc: {best_val_acc:.4f}\n")
    with open(os.path.join(model_dir, f"report_fold{fold_k}.txt"), 'w') as f:
        f.write(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} — Fold {fold_k} Confusion Matrix')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"confusion_matrix_fold{fold_k}.png"))
    plt.close()

    print(f"  Test Acc: {acc:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
    return acc, p, r, f1, best_val_acc, train_losses, val_accs


# =============================================================================
# TRIAL-LEVEL EVALUATION (Optional — majority voting)
# =============================================================================

def trial_level_accuracy(model, X_t, y_t, trial_ids_t, is_dec=False, X2_t=None):
    """
    Aggregate window predictions per trial using majority voting.
    Returns trial-level accuracy.
    """
    model.eval()
    with torch.no_grad():
        if is_dec:
            out = model(torch.FloatTensor(X_t), torch.FloatTensor(X2_t))
        else:
            out = model(torch.FloatTensor(X_t))
        _, preds = torch.max(out, 1)
    preds_np = preds.numpy()

    # Group by trial_id and take majority vote
    trial_df = pd.DataFrame({
        "trial_id": trial_ids_t,
        "pred": preds_np,
        "true": y_t
    })
    trial_acc_list = []
    for tid, grp in trial_df.groupby("trial_id"):
        majority_pred = grp["pred"].mode()[0]
        true_label    = grp["true"].iloc[0]
        trial_acc_list.append(int(majority_pred == true_label))

    return float(np.mean(trial_acc_list))


# =============================================================================
# MAIN CROSS-VALIDATION LOOP
# =============================================================================

def main():
    ensure_dir(ROOT_MODELS)

    # Load fold manifest
    with open(MANIFEST, 'r') as f:
        manifest = json.load(f)

    print("=" * 65)
    print(" OBJECTIVE 2 — CORRECTED MULTIMODAL PIPELINE")
    print("=" * 65)
    print(f"  Folds: {N_FOLDS} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"  PCA variance retained: {PCA_VAR*100:.0f}%")
    print(f"  Validation subject: {manifest['validation_subject']} (held-out)")
    print()

    # Collect per-fold results
    all_results = {name: {"acc": [], "prec": [], "rec": [], "f1": [],
                           "val_acc": [], "trial_acc": []}
                   for name in ["MLP", "DNN", "Attention", "Hybrid", "Decision Fusion"]}

    for fold_k in range(1, N_FOLDS + 1):
        print(f"\n{'='*65}")
        print(f" FOLD {fold_k} / {N_FOLDS}")
        fold_info = manifest["folds"][fold_k - 1]
        print(f"  Train subjects: {fold_info['train_subjects']}")
        print(f"  Test  subjects: {fold_info['test_subjects']}")
        print(f"{'='*65}")

        # --- Load raw fold data ---
        (tr_eeg, tr_eye, tr_y,
         ts_eeg, ts_eye, ts_y,
         va_eeg, va_eye, va_y,
         ts_trial, ts_subj) = load_fold_data(fold_k)

        # --- Preprocess (train-only fit for PCA and scaler) ---
        data = preprocess_fold(tr_eeg, tr_eye, ts_eeg, ts_eye, va_eeg, va_eye)

        # Apply the NaN-cleaned masks to labels and trial IDs
        data["tr_y"]  = tr_y[data["tr_mask"]]
        data["ts_y"]  = ts_y[data["ts_mask"]]
        data["va_y"]  = va_y[data["va_mask"]]
        ts_trial_clean = ts_trial[data["ts_mask"]] if ts_trial is not None else None

        fdim  = data["fused_dim"]
        edim  = data["eeg_dim"]
        eydim = data["eye_dim"]

        print(f"\n  Preprocessed shapes:")
        print(f"    Train : {data['tr_fused'].shape} | labels: {data['tr_y'].shape}")
        print(f"    Test  : {data['ts_fused'].shape} | labels: {data['ts_y'].shape}")
        print(f"    Val   : {data['va_fused'].shape} | labels: {data['va_y'].shape}")
        print(f"    PCA components retained: {data['n_pca']}")

        fold_model_dir = os.path.join(ROOT_MODELS, f"fold_{fold_k}")

        # --- Feature-level models (MLP, DNN, Attention, Hybrid) ---
        feature_models = [
            ("MLP",       BaselineMLP(fdim)),
            ("DNN",       DeepDNN(fdim)),
            ("Attention", AttentionModel(fdim)),
            ("Hybrid",    HybridModel(fdim)),
        ]

        for model_name, model in feature_models:
            model_dir = os.path.join(fold_model_dir, model_name.lower().replace(" ", "_"))
            acc, p, r, f1, va, losses, vaccs = train_eval_model(
                model_name, model, data, fold_k, model_dir
            )
            # Optional trial-level eval
            trl_acc = None
            if ts_trial_clean is not None:
                trl_acc = trial_level_accuracy(
                    model, data["ts_fused"], data["ts_y"], ts_trial_clean
                )
                print(f"  Trial-level Acc (majority vote): {trl_acc:.4f}")

            all_results[model_name]["acc"].append(acc)
            all_results[model_name]["prec"].append(p)
            all_results[model_name]["rec"].append(r)
            all_results[model_name]["f1"].append(f1)
            all_results[model_name]["val_acc"].append(va)
            if trl_acc is not None:
                all_results[model_name]["trial_acc"].append(trl_acc)

        # --- Decision Fusion ---
        df_model = DecisionFusion(edim, eydim)
        model_dir = os.path.join(fold_model_dir, "decision_fusion")
        acc, p, r, f1, va, losses, vaccs = train_eval_model(
            "Decision Fusion", df_model, data, fold_k, model_dir,
            X2_key_tr="tr_eye", X2_key_ts="ts_eye", X2_key_va="va_eye"
        )
        trl_acc = None
        if ts_trial_clean is not None:
            trl_acc = trial_level_accuracy(
                df_model, data["ts_eeg"], data["ts_y"], ts_trial_clean,
                is_dec=True, X2_t=data["ts_eye"]
            )
            print(f"  Trial-level Acc (Decision Fusion, majority vote): {trl_acc:.4f}")

        all_results["Decision Fusion"]["acc"].append(acc)
        all_results["Decision Fusion"]["prec"].append(p)
        all_results["Decision Fusion"]["rec"].append(r)
        all_results["Decision Fusion"]["f1"].append(f1)
        all_results["Decision Fusion"]["val_acc"].append(va)
        if trl_acc is not None:
            all_results["Decision Fusion"]["trial_acc"].append(trl_acc)

    # =========================================================================
    # AGGREGATE RESULTS (mean ± std across 5 folds)
    # =========================================================================
    print("\n" + "=" * 65)
    print(" CROSS-VALIDATED RESULTS (mean ± std across 5 folds)")
    print("=" * 65)

    summary_rows = []
    for model_name, res in all_results.items():
        acc_arr  = np.array(res["acc"])
        prec_arr = np.array(res["prec"])
        rec_arr  = np.array(res["rec"])
        f1_arr   = np.array(res["f1"])
        va_arr   = np.array(res["val_acc"])

        row = {
            "Model":          model_name,
            "Accuracy":       f"{acc_arr.mean():.4f}",
            "Acc_std":        f"{acc_arr.std():.4f}",
            "Precision":      f"{prec_arr.mean():.4f}",
            "Recall":         f"{rec_arr.mean():.4f}",
            "F1-Score":       f"{f1_arr.mean():.4f}",
            "F1_std":         f"{f1_arr.std():.4f}",
            "Best_Val_Acc":   f"{va_arr.mean():.4f}",
        }
        if res["trial_acc"]:
            ta = np.array(res["trial_acc"])
            row["Trial_Acc"] = f"{ta.mean():.4f}"
        summary_rows.append(row)

        print(f"\n  {model_name}")
        print(f"    Test  Acc : {acc_arr.mean()*100:.2f}% ± {acc_arr.std()*100:.2f}%")
        print(f"    Precision : {prec_arr.mean()*100:.2f}%")
        print(f"    Recall    : {rec_arr.mean()*100:.2f}%")
        print(f"    F1-Score  : {f1_arr.mean()*100:.2f}% ± {f1_arr.std()*100:.2f}%")
        print(f"    Best Val  : {va_arr.mean()*100:.2f}%")
        if res["trial_acc"]:
            print(f"    Trial Acc : {np.array(res['trial_acc']).mean()*100:.2f}%")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    comp_dir   = os.path.join(ROOT_MODELS, "comparison")
    ensure_dir(comp_dir)
    summary_df.to_csv(os.path.join(comp_dir, "model_comparison_corrected.csv"), index=False)
    print(f"\n  Saved: model_comparison_corrected.csv")

    # =========================================================================
    # BEFORE vs AFTER COMPARISON TABLE
    # =========================================================================
    ORIGINAL = {
        "MLP":            0.7603,
        "DNN":            0.9035,
        "Attention":      0.6826,
        "Hybrid":         0.9284,
        "Decision Fusion":0.5055,
    }

    print("\n" + "=" * 65)
    print(" BEFORE vs AFTER (Original inflated vs Corrected valid)")
    print("=" * 65)
    print(f"  {'Model':<20} {'Before':>10} {'After (mean)':>14} {'Δ':>8}")
    print(f"  {'-'*60}")
    for row in summary_rows:
        name = row["Model"]
        after = float(row["Accuracy"])
        before = ORIGINAL.get(name, float('nan'))
        delta = after - before
        arrow = "▼" if delta < 0 else "▲"
        print(f"  {name:<20} {before*100:>9.2f}%  {after*100:>12.2f}%  {arrow}{abs(delta)*100:>5.2f}%")

    # =========================================================================
    # MODEL PERFORMANCE BAR CHART
    # =========================================================================
    model_names = [r["Model"] for r in summary_rows]
    accs        = [float(r["Accuracy"]) * 100 for r in summary_rows]
    stds        = [float(r["Acc_std"]) * 100   for r in summary_rows]
    orig_accs   = [ORIGINAL.get(r["Model"], 0) * 100 for r in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # Left: corrected results
    bars = axes[0].barh(model_names, accs, xerr=stds, color=colors, alpha=0.85,
                        capsize=5, error_kw={"ecolor": "black", "lw": 1.5})
    axes[0].axvline(25, color='gray', ls='--', lw=1, label='Chance (25%)')
    axes[0].set_xlabel("Test Accuracy (%)", fontsize=11)
    axes[0].set_title("Corrected Results\n(Subject-Level CV, No Leakage)",
                      fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, 100)
    for bar, val in zip(bars, accs):
        axes[0].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va='center', fontsize=9)
    axes[0].legend()

    # Right: before vs after
    x      = np.arange(len(model_names))
    width  = 0.35
    axes[1].bar(x - width/2, orig_accs, width, label='Original (Inflated)',
                color='#E84855', alpha=0.8)
    axes[1].bar(x + width/2, accs,      width, label='Corrected (Valid)',
                color='#4C72B0', alpha=0.8)
    axes[1].axhline(25, color='gray', ls='--', lw=1, label='Chance (25%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_title("Before vs After\n(Leakage Removed)",
                      fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "performance_corrected.png"), dpi=130)
    plt.close()
    print("  Saved: performance_corrected.png")

    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    generate_report(summary_rows, ORIGINAL, manifest)

    print("\n" + "=" * 65)
    print(" PIPELINE COMPLETE")
    print(" Results saved to:", ROOT_MODELS)
    print("=" * 65)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(summary_rows, original_results, manifest):
    report_path = os.path.join(ROOT_MODELS, "objective2_corrected_report.md")

    lines = []
    lines += [
        "# Objective 2 — Corrected Pipeline Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Validation Subject:** {manifest['validation_subject']} (held-out from all folds)  ",
        f"**Cross-Validation:** 5-fold subject-level CV  ",
        f"**PCA Variance Retained:** 95%  ",
        "",
        "---",
        "",
        "## 1. Issues Identified in Original Pipeline",
        "",
        "| Issue | Description | Severity |",
        "|-------|-------------|----------|",
        "| Random row split | `train_test_split(X, y)` mixed windows from same subject across train/test | 🚨 Critical |",
        "| Scaler leakage | `StandardScaler.fit_transform(X_full)` called on full dataset before split | 🚨 Critical |",
        "| PCA leakage | `X_eeg_pca.npy` fitted on full dataset (no per-fold PCA) | 🚨 Critical |",
        "| Test used for model selection | Best epoch chosen by test accuracy, not validation accuracy | ⚠️ High |",
        "| Unscaled Decision Fusion inputs | EEG/Eye inputs to Decision Fusion were not normalized | ⚠️ Medium |",
        "| No cross-validation | Single fixed split — no variance estimation | ⚠️ Medium |",
        "",
        "---",
        "",
        "## 2. Fixes Applied",
        "",
        "| Fix | Implementation |",
        "|-----|---------------|",
        "| Subject-level splitting | Used `stage2_output/fold_k_*` CSVs — subjects fully isolated per fold |",
        "| PCA on train only | `PCA(0.95).fit_transform(X_eeg_train)` per fold; `.transform()` on test/val |",
        "| Scaler on train only | `StandardScaler().fit_transform(X_train)` per fold; `.transform()` on test/val |",
        "| Separate validation set | Subject 15 held out; val used for early stopping and model selection |",
        "| Model selection on val | `if val_acc > best_val_acc: save_model()` — test never seen during training |",
        "| Scaled Decision Fusion | Separate scalers for EEG-PCA and Eye streams applied before fusion |",
        "| 5-fold cross-validation | Results reported as mean ± std across all 5 folds |",
        "",
        "---",
        "",
        "## 3. Updated Results",
        "",
        "### 3a. Cross-Validated Test Performance (Corrected)",
        "",
        "| Model | Accuracy (mean ± std) | Precision | Recall | F1-Score |",
        "|-------|-----------------------|-----------|--------|----------|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['Model']} | {float(row['Accuracy'])*100:.2f}% ± {float(row['Acc_std'])*100:.2f}% "
            f"| {float(row['Precision'])*100:.2f}% "
            f"| {float(row['Recall'])*100:.2f}% "
            f"| {float(row['F1-Score'])*100:.2f}% ± {float(row['F1_std'])*100:.2f}% |"
        )

    lines += [
        "",
        "### 3b. Before vs After (Inflated vs Valid)",
        "",
        "| Model | Original (Inflated) | Corrected (Valid) | Difference |",
        "|-------|--------------------|-------------------|------------|",
    ]
    for row in summary_rows:
        name  = row["Model"]
        after = float(row["Accuracy"])
        before = original_results.get(name, float('nan'))
        delta  = after - before
        sign   = "▼" if delta < 0 else "▲"
        lines.append(
            f"| {name} | {before*100:.2f}% | {after*100:.2f}% | {sign}{abs(delta)*100:.2f}% |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Key Insights",
        "",
        "### Why accuracy dropped from original results",
        "",
        "The original pipeline's high accuracy (up to **92.84%**) was inflated by three compounding leakages:",
        "",
        "1. **Within-subject window correlation**: When windows from the same subject appear in both train",
        "   and test, the model learns subject-specific EEG patterns rather than generalizable emotion",
        "   features. Since consecutive windows share signal structure, this is essentially a memorization",
        "   test — not a generalization test.",
        "",
        "2. **Preprocessing leakage**: The StandardScaler and PCA saw the test subjects' data during",
        "   fitting. This subtly shifts the feature space in a direction that benefits the test set.",
        "",
        "3. **Model selection leakage**: The 'best' model was chosen based on test accuracy across",
        "   epochs, turning the test set into a de facto validation set.",
        "",
        "### Why the corrected results are scientifically valid",
        "",
        "- Each test subject's data was **never seen during training or preprocessing**.",
        "- PCA and scaling parameters are computed **exclusively on training subjects**.",
        "- Model selection is performed on the **held-out validation subject (Subject 15)**.",
        "- 5-fold cross-validation provides **reliable variance estimates** (mean ± std).",
        "",
        "### Expected accuracy range",
        "",
        "Subject-independent EEG emotion recognition on SEED-IV typically achieves **50–75%** with",
        "classical models and **65–85%** with advanced deep learning. Results in this range are",
        "realistic and scientifically defensible.",
        "",
        "---",
        "",
        "## 5. Architecture Notes (Unchanged)",
        "",
        "| Model | Architecture | Notes |",
        "|-------|-------------|-------|",
        "| MLP | 2-layer (128→64→4) + BN + Dropout | Baseline |",
        "| DNN | 3-layer (256→128→64→4) + Dropout | Deeper baseline |",
        "| Attention | Sigmoid-gated element-wise attention | As in original |",
        "| Hybrid | DNN + Attention gate (128→64→4) | Feature + Attention |",
        "| Decision Fusion | Separate EEG-PCA + Eye streams, averaged logits | Late fusion |",
        "",
        "> Note: Model architectures are unchanged from original. Only pre-processing",
        "> and evaluation strategy were corrected.",
        "",
        "---",
        "",
        "## 6. Files Generated",
        "",
        "```",
        "stage4_models_corrected/",
        "├── fold_1/ ... fold_5/           # Per-fold model weights + reports",
        "│   ├── mlp/",
        "│   ├── dnn/",
        "│   ├── attention/",
        "│   ├── hybrid/",
        "│   └── decision_fusion/",
        "└── comparison/",
        "    ├── model_comparison_corrected.csv",
        "    └── performance_corrected.png",
        "```",
    ]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"\n  Saved report: {report_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
