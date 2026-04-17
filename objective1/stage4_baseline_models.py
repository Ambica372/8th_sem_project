# -*- coding: utf-8 -*-
"""
=============================================================================
Stage 4: Baseline Classification Analysis
Project : Physiological Emotion Recognition (SEED-IV)
=============================================================================

Models : SVM (RBF), Random Forest, LDA, k-NN
Modalities : EEG (310-D) and Eye (31-D) — trained SEPARATELY
Folds  : 5-fold subject-level CV from Stage 2

Pipeline per fold:
  1. Load fold_k_train / fold_k_test CSVs
  2. Fit 4 models on training windows
  3. Predict on test windows (window-level)
  4. Aggregate window predictions to trial-level:
       (a) Majority voting
       (b) Probability averaging
  5. Compute trial-level metrics: Accuracy, Macro P/R/F1
  6. Evaluate on fold_k_validation (held-out subject 15)

⚠ SVM subsample note:
  RBF-SVM scales as O(n²) — training on 27K samples × 310 features
  would take hours. We subsample SVM training to SVM_TRAIN_N=8000 windows
  (stratified by emotion label).  All OTHER models use the full training set.
  This is documented in the report.  Test/val prediction uses ALL samples.

=============================================================================
"""

import os, sys, io, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.svm              import SVC
from sklearn.ensemble          import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.metrics           import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       confusion_matrix)
from sklearn.utils             import resample

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STAGE2_DIR = r"c:\Users\Rose J Thachil\Documents\8th sem\stage2_output"
OUTPUT_DIR = r"c:\Users\Rose J Thachil\Documents\8th sem\stage4_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FOLDS         = 5
RANDOM_SEED     = 42
SVM_TRAIN_N     = 8000    # See note above
META_COLS       = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]
EMOTION_NAMES   = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
MODALITIES      = ["eeg", "eye"]
SEP  = "=" * 65
SEP2 = "-" * 50

# ---------------------------------------------------------------------------
# Model definitions (default params; seeds fixed where applicable)
# ---------------------------------------------------------------------------
def build_models():
    return {
        "SVM_RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_SEED,
            cache_size=2000, max_iter=2000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=RANDOM_SEED,
        ),
        "LDA": LinearDiscriminantAnalysis(
            solver="svd", n_components=3
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=7, metric="euclidean",
            weights="distance", n_jobs=-1,
        ),
    }


# =============================================================================
# MODULE 1: Data loading
# =============================================================================
def load_fold(fold_k, modality, split):
    """
    Load fold_k_{split}_{modality}.csv from STAGE2_DIR.
    Returns X (numpy array), y (labels), groups (subject, trial keys).
    """
    fname = "fold_" + str(fold_k) + "_" + split + "_" + modality + ".csv"
    path  = os.path.join(STAGE2_DIR, fname)
    df    = pd.read_csv(path)

    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].values.astype(np.float64)
    y = df["emotion_label"].values.astype(int)
    groups = df[["subject_id", "session_id", "trial_id", "window_id",
                 "emotion_label"]].copy()

    return X, y, groups


def load_validation(fold_k, modality):
    """Load fold_k_validation_{modality}.csv (held-out subject 15)."""
    fname = "fold_" + str(fold_k) + "_validation_" + modality + ".csv"
    return load_fold.__wrapped__ if False else _load_csv(fname)


def _load_csv(fname):
    path = os.path.join(STAGE2_DIR, fname)
    df   = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].values.astype(np.float64)
    y = df["emotion_label"].values.astype(int)
    groups = df[["subject_id", "session_id", "trial_id", "window_id",
                 "emotion_label"]].copy()
    return X, y, groups


# =============================================================================
# MODULE 2: Train model
# =============================================================================
def train_model(model_name, model, X_train, y_train):
    """
    Fit a model.
    For SVM: stratified subsample to SVM_TRAIN_N windows for speed.
    Returns fitted model.
    """
    t0 = time.time()
    if model_name == "SVM_RBF" and len(X_train) > SVM_TRAIN_N:
        # Stratified subsample
        idx = []
        classes, counts = np.unique(y_train, return_counts=True)
        per_class = SVM_TRAIN_N // len(classes)
        for cls in classes:
            cls_idx = np.where(y_train == cls)[0]
            chosen  = np.random.choice(
                cls_idx, min(per_class, len(cls_idx)), replace=False
            )
            idx.extend(chosen.tolist())
        np.random.shuffle(idx)
        X_fit = X_train[idx]
        y_fit = y_train[idx]
        print("    SVM: stratified subsample " + str(len(X_fit))
              + " / " + str(len(X_train)) + " windows for training")
    else:
        X_fit, y_fit = X_train, y_train

    # Handle NaN for Eye data: impute column median from training subset
    if np.isnan(X_fit).any():
        col_medians = np.nanmedian(X_fit, axis=0)
        for j in range(X_fit.shape[1]):
            mask = np.isnan(X_fit[:, j])
            if mask.any():
                X_fit[mask, j] = col_medians[j]

    model.fit(X_fit, y_fit)
    elapsed = round(time.time() - t0, 1)
    print("    " + model_name + " trained in " + str(elapsed) + "s")
    return model, col_medians if np.isnan(X_train).any() else None


# =============================================================================
# MODULE 3: Predict on windows
# =============================================================================
def predict_windows(model, X_test, col_medians=None):
    """
    Returns:
      y_pred      : (N,) hard labels
      y_proba     : (N, 4) class probabilities
    """
    X = X_test.copy()

    # NaN imputation using training medians (if any)
    if col_medians is not None:
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_medians[j]
    elif np.isnan(X).any():
        col_medians_ = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_medians_[j]

    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba


# =============================================================================
# MODULE 4: Trial-level aggregation
# =============================================================================
def aggregate_trials(groups_df, y_pred, y_proba, n_classes=4):
    """
    Aggregate window-level predictions to trial level.

    Parameters
    ----------
    groups_df : DataFrame with columns [subject_id, session_id, trial_id,
                                         window_id, emotion_label]
    y_pred    : (N,) predicted window labels
    y_proba   : (N, n_classes) predicted probabilities

    Returns
    -------
    trial_df : DataFrame with columns:
                 subject_id, session_id, trial_id,
                 true_label,
                 pred_majority, pred_proba_avg
    """
    df = groups_df.copy().reset_index(drop=True)
    df["pred_label"]  = y_pred

    for c in range(n_classes):
        df["prob_" + str(c)] = y_proba[:, c]

    trial_rows = []
    for (subj, sess, trial_id), grp in df.groupby(
            ["subject_id", "session_id", "trial_id"]):

        true_label = int(grp["emotion_label"].mode()[0])

        # A. Majority voting
        pred_maj = int(grp["pred_label"].mode()[0])

        # B. Probability averaging
        avg_proba = grp[["prob_" + str(c) for c in range(n_classes)]].mean().values
        pred_prob = int(np.argmax(avg_proba))

        trial_rows.append({
            "subject_id":    subj,
            "session_id":    sess,
            "trial_id":      trial_id,
            "true_label":    true_label,
            "pred_majority": pred_maj,
            "pred_proba_avg": pred_prob,
        })

    return pd.DataFrame(trial_rows)


# =============================================================================
# MODULE 5: Evaluate metrics
# =============================================================================
def evaluate_metrics(trial_df, agg_method="majority"):
    """
    Compute trial-level metrics.
    agg_method: 'majority' or 'proba_avg'
    Returns dict of metrics.
    """
    col = "pred_majority" if agg_method == "majority" else "pred_proba_avg"
    y_true = trial_df["true_label"].values
    y_pred = trial_df[col].values

    acc  = round(accuracy_score(y_true, y_pred) * 100, 2)
    prec = round(precision_score(y_true, y_pred, average="macro",
                                  zero_division=0) * 100, 2)
    rec  = round(recall_score(y_true, y_pred, average="macro",
                               zero_division=0) * 100, 2)
    f1   = round(f1_score(y_true, y_pred, average="macro",
                           zero_division=0) * 100, 2)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "confusion_matrix": cm.tolist(),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_stage4():
    print(SEP)
    print(" STAGE 4 -- BASELINE CLASSIFICATION")
    print(SEP)

    # Storage for all results
    all_results = []          # one row per (modality, model, fold, agg_method)
    val_results = []          # validation subject results

    for modality in MODALITIES:
        print("\n" + "#" * 65)
        print("# MODALITY: " + modality.upper())
        print("#" * 65)

        fold_metrics = {m: {"majority": [], "proba_avg": []}
                        for m in build_models()}

        # ------------------------------------------------------------------
        # CROSS-VALIDATION LOOP
        # ------------------------------------------------------------------
        for k in range(1, N_FOLDS + 1):
            print("\n" + SEP2)
            print("  FOLD " + str(k) + " [" + modality.upper() + "]")
            print(SEP2)

            # Load data
            X_train, y_train, g_train = load_fold(k, modality, "train")
            X_test,  y_test,  g_test  = load_fold(k, modality, "test")
            X_val,   y_val,   g_val   = _load_csv(
                "fold_" + str(k) + "_validation_" + modality + ".csv"
            )

            print("  Train: " + str(X_train.shape)
                  + " | Test: " + str(X_test.shape)
                  + " | Val: " + str(X_val.shape))

            models = build_models()

            fold_row = {
                "modality": modality, "fold": k,
            }

            for model_name, model in models.items():
                print("\n  -- " + model_name + " --")

                # Train
                trained_model, col_medians = train_model(
                    model_name, model, X_train, y_train
                )

                # Predict on test windows
                y_pred_w, y_proba_w = predict_windows(
                    trained_model, X_test, col_medians
                )
                window_acc_test = round(
                    accuracy_score(y_test, y_pred_w) * 100, 2
                )
                print("    Window accuracy (test): " + str(window_acc_test) + "%")

                # Aggregate to trials
                trial_df_test = aggregate_trials(g_test, y_pred_w, y_proba_w)

                # Evaluate both aggregation methods
                for agg in ["majority", "proba_avg"]:
                    m = evaluate_metrics(trial_df_test, agg)
                    fold_metrics[model_name][agg].append(m)

                    print("    Trial [" + agg + "] acc=" + str(m["accuracy"])
                          + "% f1=" + str(m["f1"]) + "%")

                    all_results.append({
                        "modality":        modality,
                        "model":           model_name,
                        "fold":            k,
                        "agg_method":      agg,
                        "accuracy":        m["accuracy"],
                        "precision":       m["precision"],
                        "recall":          m["recall"],
                        "f1":              m["f1"],
                    })

                # ------ Validation subject --------------------------------
                y_pred_val, y_proba_val = predict_windows(
                    trained_model, X_val, col_medians
                )
                trial_df_val = aggregate_trials(g_val, y_pred_val, y_proba_val)

                for agg in ["majority", "proba_avg"]:
                    mv = evaluate_metrics(trial_df_val, agg)
                    val_results.append({
                        "modality":   modality,
                        "model":      model_name,
                        "fold":       k,
                        "agg_method": agg,
                        "accuracy":   mv["accuracy"],
                        "precision":  mv["precision"],
                        "recall":     mv["recall"],
                        "f1":         mv["f1"],
                    })

        # ------------------------------------------------------------------
        # Save per-fold results CSV for this modality
        # ------------------------------------------------------------------
        mod_rows = [r for r in all_results if r["modality"] == modality]
        fold_df  = pd.DataFrame(mod_rows)
        fold_csv = os.path.join(OUTPUT_DIR,
                     "fold_results_" + modality + ".csv")
        fold_df.to_csv(fold_csv, index=False)
        print("\n  Saved: " + fold_csv)

    # ======================================================================
    # Summary: mean ± std across folds per model per modality
    # ======================================================================
    print("\n" + SEP)
    print(" RESULTS SUMMARY (mean +/- std over 5 folds, trial-level)")
    print(SEP)

    all_df   = pd.DataFrame(all_results)
    summary_rows = []

    for modality in MODALITIES:
        for model_name in build_models():
            for agg in ["majority", "proba_avg"]:
                sub = all_df[
                    (all_df["modality"]   == modality) &
                    (all_df["model"]      == model_name) &
                    (all_df["agg_method"] == agg)
                ]
                if sub.empty:
                    continue
                row = {
                    "modality":   modality,
                    "model":      model_name,
                    "agg_method": agg,
                }
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    vals = sub[metric].values
                    row[metric + "_mean"] = round(float(vals.mean()), 2)
                    row[metric + "_std"]  = round(float(vals.std()),  2)
                summary_rows.append(row)

                print("  [" + modality.upper() + " | " + model_name
                      + " | " + agg + "]"
                      + "  Acc=" + str(row["accuracy_mean"])
                      + "+/-" + str(row["accuracy_std"])
                      + "  F1="  + str(row["f1_mean"])
                      + "+/-" + str(row["f1_std"]))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "baseline_results_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\n  Saved: " + summary_csv)

    # ======================================================================
    # Validation summary
    # ======================================================================
    print("\n" + SEP)
    print(" VALIDATION SUBJECT 15 RESULTS (mean over 5 folds)")
    print(SEP)

    val_df = pd.DataFrame(val_results)
    val_summary_rows = []

    for modality in MODALITIES:
        for model_name in build_models():
            for agg in ["majority", "proba_avg"]:
                sub = val_df[
                    (val_df["modality"]   == modality) &
                    (val_df["model"]      == model_name) &
                    (val_df["agg_method"] == agg)
                ]
                if sub.empty:
                    continue
                row = {
                    "modality":   modality,
                    "model":      model_name,
                    "agg_method": agg,
                }
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    vals = sub[metric].values
                    row[metric + "_mean"] = round(float(vals.mean()), 2)
                    row[metric + "_std"]  = round(float(vals.std()),  2)
                val_summary_rows.append(row)

                print("  [" + modality.upper() + " | " + model_name
                      + " | " + agg + "]"
                      + "  Acc=" + str(row["accuracy_mean"])
                      + "+/-" + str(row["accuracy_std"])
                      + "  F1="  + str(row["f1_mean"])
                      + "+/-" + str(row["f1_std"]))

    val_summary_df = pd.DataFrame(val_summary_rows)
    val_csv = os.path.join(OUTPUT_DIR, "validation_results.csv")
    val_summary_df.to_csv(val_csv, index=False)
    print("\n  Saved: " + val_csv)

    # ======================================================================
    # Best model identification
    # ======================================================================
    print("\n" + SEP)
    print(" BEST MODEL PER MODALITY (by mean F1, majority vote)")
    print(SEP)

    best_info = {}
    for modality in MODALITIES:
        sub = summary_df[
            (summary_df["modality"]   == modality) &
            (summary_df["agg_method"] == "majority")
        ]
        if sub.empty:
            continue
        best_row = sub.loc[sub["f1_mean"].idxmax()]
        best_info[modality] = best_row
        print("  " + modality.upper()
              + " | Best model: " + best_row["model"]
              + "  F1=" + str(best_row["f1_mean"])
              + "+/-" + str(best_row["f1_std"])
              + "  Acc=" + str(best_row["accuracy_mean"]))

    # Save all_results and val_results raw
    all_df.to_csv(os.path.join(OUTPUT_DIR, "all_fold_results_raw.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "all_val_results_raw.csv"),  index=False)

    print("\n  Output directory: " + OUTPUT_DIR)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        sz = round(os.path.getsize(os.path.join(OUTPUT_DIR, fname)) / 1024, 1)
        print("    " + fname + "  (" + str(sz) + " KB)")

    return summary_df, val_summary_df, best_info


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    t_start = time.time()
    summary_df, val_summary_df, best_info = run_stage4()
    elapsed = round((time.time() - t_start) / 60, 1)
    print("\n  Total runtime: " + str(elapsed) + " min")
    print("\n>>> Stage 4 complete.")
