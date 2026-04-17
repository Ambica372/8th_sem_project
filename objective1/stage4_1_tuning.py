# -*- coding: utf-8 -*-
"""
=============================================================================
Stage 4.1: Hyperparameter Tuning + Confusion Matrix
Project : Physiological Emotion Recognition (SEED-IV)
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")   # must be before plt/sns import — prevents tkinter crash

import os, sys, io, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm                   import SVC
from sklearn.ensemble              import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.model_selection       import GridSearchCV, RandomizedSearchCV
from sklearn.metrics               import (accuracy_score, precision_score,
                                           recall_score, f1_score,
                                           confusion_matrix)

warnings.filterwarnings("ignore")

# Force UTF-8 + unbuffered output so terminal shows progress in real-time
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                  encoding="utf-8", errors="replace",
                                  line_buffering=True)

def log(msg=""):
    print(msg, flush=True)

np.random.seed(42)

# =============================================================================
# Paths
# =============================================================================
BASE_DIR   = r"c:\Users\Rose J Thachil\Documents\8th sem_new"
STAGE2_DIR = os.path.join(BASE_DIR, "stage2_output")
OUTPUT_DIR = os.path.join(BASE_DIR, "stage4_1_output")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "confusion_matrices")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# =============================================================================
# Constants
# =============================================================================
N_FOLDS       = 5
RANDOM_SEED   = 42
SVM_TRAIN_N   = 6000   # Keep SVM manageable (per-class 1500 x4)
META_COLS     = ["subject_id", "session_id", "trial_id",
                 "window_id", "emotion_label"]
EMOTION_NAMES = ["neutral", "sad", "fear", "happy"]
SEP  = "=" * 65
SEP2 = "-" * 50

# Hyperparameter search spaces
# SVM : GridSearch 9 combos x inner-3CV = 27 fits (on 6k subsample)
# RF  : RandomizedSearch 6 iters x 3CV  = 18 fits (faster than full grid)
# kNN : GridSearch 6 combos x 3CV       = 18 fits
# LDA : no tuning
PARAM_GRIDS = {
    "SVM_RBF": {
        "C":     [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.001],
    },
    "RandomForest": {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "LDA": {},
    "kNN": {
        "n_neighbors": [3, 5, 7],
        "weights":     ["uniform", "distance"],
    },
}

def build_base_models():
    return {
        "SVM_RBF": SVC(
            kernel="rbf", probability=True,
            random_state=RANDOM_SEED, cache_size=2000
        ),
        "RandomForest": RandomForestClassifier(
            n_jobs=-1, random_state=RANDOM_SEED
        ),
        "LDA": LinearDiscriminantAnalysis(solver="svd", n_components=3),
        "kNN": KNeighborsClassifier(n_jobs=-1),
    }

# =============================================================================
# Helpers
# =============================================================================
def load_csv(fname):
    path = os.path.join(STAGE2_DIR, fname)
    df   = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X   = df[feat_cols].values.astype(np.float64)
    y   = df["emotion_label"].values.astype(int)
    grp = df[["subject_id", "session_id", "trial_id",
               "window_id", "emotion_label"]].copy()
    return X, y, grp


def impute_nan(X_tr, X_te=None):
    meds = np.nanmedian(X_tr, axis=0)
    for j in range(X_tr.shape[1]):
        m = np.isnan(X_tr[:, j])
        if m.any(): X_tr[m, j] = meds[j]
    if X_te is not None:
        for j in range(X_te.shape[1]):
            m = np.isnan(X_te[:, j])
            if m.any(): X_te[m, j] = meds[j]


def stratified_sub(X, y, n_total):
    """Stratified subsample of n_total windows (equal per class)."""
    idx = []
    classes = np.unique(y)
    per_cls = n_total // len(classes)
    for cls in classes:
        ci = np.where(y == cls)[0]
        ch = np.random.choice(ci, min(per_cls, len(ci)), replace=False)
        idx.extend(ch.tolist())
    np.random.shuffle(idx)
    return X[idx], y[idx]


def aggregate_trials(grp_df, y_pred, y_proba):
    df = grp_df.copy().reset_index(drop=True)
    df["pred"] = y_pred
    for c in range(4):
        df[f"p{c}"] = y_proba[:, c]

    rows = []
    for (s, ss, t), g in df.groupby(["subject_id", "session_id", "trial_id"]):
        true_l = int(g["emotion_label"].mode()[0])
        maj    = int(g["pred"].mode()[0])
        avg_p  = g[[f"p{c}" for c in range(4)]].mean().values
        prob_l = int(np.argmax(avg_p))
        rows.append({"true": true_l, "majority": maj, "proba_avg": prob_l})
    return pd.DataFrame(rows)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="macro",
                                           zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_true, y_pred, average="macro",
                                        zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0) * 100, 2),
    }


def compute_per_class(y_true, y_pred):
    p  = precision_score(y_true, y_pred, average=None,
                         labels=[0,1,2,3], zero_division=0)
    r  = recall_score(y_true, y_pred, average=None,
                      labels=[0,1,2,3], zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None,
                  labels=[0,1,2,3], zero_division=0)
    return p, r, f1


def save_cm(y_true, y_pred, modality, model_name, fold):
    cm      = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3],
                               normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{modality.upper()} | {model_name} | Fold {fold}",
                 fontsize=13, fontweight="bold")

    for ax, mat, fmt, title in [
        (axes[0], cm,      "d",    "Raw Counts"),
        (axes[1], cm_norm, ".2f",  "Normalized (Recall per Emotion)"),
    ]:
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=EMOTION_NAMES,
                    yticklabels=EMOTION_NAMES,
                    ax=ax, linewidths=0.5, linecolor="grey")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    fname = f"cm_fold{fold}_{model_name}_{modality}.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    log(f"      CM saved: {fname}")


# =============================================================================
# Per-fold function
# =============================================================================
def run_fold(k, modality, all_results, all_per_class):
    log(f"\n{SEP2}")
    log(f"  Fold {k}  [{modality.upper()}]")
    log(SEP2)

    X_tr, y_tr, g_tr = load_csv(f"fold_{k}_train_{modality}.csv")
    X_te, y_te, g_te = load_csv(f"fold_{k}_test_{modality}.csv")
    impute_nan(X_tr, X_te)

    log(f"  Train: {X_tr.shape}  |  Test: {X_te.shape}")

    for model_name, base_model in build_base_models().items():
        t0 = time.time()
        log(f"\n    [{model_name}]")

        # --- subsample for SVM ---
        X_fit, y_fit = X_tr, y_tr
        if model_name == "SVM_RBF" and len(X_tr) > SVM_TRAIN_N:
            X_fit, y_fit = stratified_sub(X_tr, y_tr, SVM_TRAIN_N)
            log(f"      SVM subsample: {len(X_fit)}/{len(X_tr)} windows")

        # --- hyperparameter search ---
        grid = PARAM_GRIDS[model_name]
        if grid:
            if model_name == "RandomForest":
                # Randomized search — faster for RF's large grid (12 combos)
                searcher = RandomizedSearchCV(
                    base_model, grid, n_iter=6, cv=3,
                    scoring="f1_macro", n_jobs=-1,
                    random_state=RANDOM_SEED
                )
                log(f"      RandomizedSearchCV (n_iter=6, inner cv=3)...")
            else:
                searcher = GridSearchCV(
                    base_model, grid, cv=3,
                    scoring="f1_macro", n_jobs=-1
                )
                log(f"      GridSearchCV ({len(grid)} params x 3-fold)...")

            searcher.fit(X_fit, y_fit)
            best_model  = searcher.best_estimator_
            best_params = str(searcher.best_params_)
            log(f"      Best params: {best_params}")
        else:
            # LDA — no tuning
            base_model.fit(X_fit, y_fit)
            best_model  = base_model
            best_params = "default"
            log(f"      LDA: using default params")

        elapsed = round(time.time() - t0, 1)
        log(f"      Search took {elapsed}s")

        # --- predict on test windows ---
        log(f"      Predicting on {len(X_te)} test windows...")
        y_pred_w  = best_model.predict(X_te)
        y_proba_w = best_model.predict_proba(X_te)

        # --- trial aggregation ---
        trial_df = aggregate_trials(g_te, y_pred_w, y_proba_w)

        # --- metrics for both aggregation methods ---
        for agg in ["majority", "proba_avg"]:
            y_true_t = trial_df["true"].values
            y_pred_t = trial_df[agg].values

            m         = compute_metrics(y_true_t, y_pred_t)
            p, r, f1v = compute_per_class(y_true_t, y_pred_t)

            all_results.append({
                "modality": modality, "fold": k,
                "model": model_name, "agg": agg,
                "best_params": best_params, **m
            })

            for ci, cname in enumerate(EMOTION_NAMES):
                all_per_class.append({
                    "modality": modality, "fold": k,
                    "model": model_name, "agg": agg,
                    "class": cname,
                    "precision": round(p[ci]  * 100, 2),
                    "recall":    round(r[ci]  * 100, 2),
                    "f1":        round(f1v[ci] * 100, 2),
                })

            log(f"      [{agg:10}] Acc={m['accuracy']:.1f}%"
                f"  F1={m['f1']:.1f}%"
                f"  P={m['precision']:.1f}%"
                f"  R={m['recall']:.1f}%")

        # --- save confusion matrix (proba_avg preferred) ---
        save_cm(trial_df["true"].values, trial_df["proba_avg"].values,
                modality, model_name, k)


# =============================================================================
# Main
# =============================================================================
def run_stage4_1():
    log(SEP)
    log(" STAGE 4.1 -- HYPERPARAMETER TUNING + CONFUSION MATRICES")
    log(SEP)

    all_results   = []
    all_per_class = []

    for modality in ["eeg", "eye"]:
        log(f"\n{'#'*65}")
        log(f"  MODALITY: {modality.upper()}")
        log(f"{'#'*65}")
        for k in range(1, N_FOLDS + 1):
            run_fold(k, modality, all_results, all_per_class)

    # -----------------------------------------------------------------------
    # Build & save summaries
    # -----------------------------------------------------------------------
    log(f"\n{SEP}")
    log("  Saving results...")
    log(SEP)

    df_all   = pd.DataFrame(all_results)
    df_class = pd.DataFrame(all_per_class)

    # Tuned summary — proba_avg only
    pa = df_all[df_all["agg"] == "proba_avg"]
    summary_rows = []
    for (mod, model), grp in pa.groupby(["modality", "model"]):
        row = {"modality": mod, "model": model}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            row[f"{metric}_mean"] = round(grp[metric].mean(), 2)
            row[f"{metric}_std"]  = round(grp[metric].std(),  2)
        row["best_params_example"] = grp["best_params"].iloc[0]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "tuned_results_summary.csv"),
                      index=False)
    log("  Saved: tuned_results_summary.csv")

    # Per-class summary
    pc_pa = df_class[df_class["agg"] == "proba_avg"]
    pc_grp = (pc_pa
              .groupby(["modality", "model", "class"])[["precision","recall","f1"]]
              .agg(["mean","std"])
              .reset_index())
    pc_grp.columns = ["_".join(c).strip("_") for c in pc_grp.columns]
    pc_grp.to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)
    log("  Saved: per_class_metrics.csv")

    # -----------------------------------------------------------------------
    # Print final summary table
    # -----------------------------------------------------------------------
    log(f"\n{SEP}")
    log("  TUNED RESULTS SUMMARY  (Probability Avg, mean±std over 5 folds)")
    log(SEP)
    log(f"  {'Modality':<6} {'Model':<14} {'Accuracy':>10} {'F1 Macro':>10}")
    log(f"  {'-'*6} {'-'*14} {'-'*10} {'-'*10}")
    for _, row in summary_df.sort_values(["modality","f1_mean"],
                                          ascending=[True,False]).iterrows():
        log(f"  {row['modality'].upper():<6} {row['model']:<14}"
            f"  {row['accuracy_mean']:>5.1f}±{row['accuracy_std']:<4.1f}"
            f"  {row['f1_mean']:>5.1f}±{row['f1_std']:<4.1f}")

    log(f"\n  Output directory: {OUTPUT_DIR}")
    log(f"  Confusion matrices: {PLOTS_DIR}")

    return summary_df, pc_grp


if __name__ == "__main__":
    t0 = time.time()
    summary_df, pc_grp = run_stage4_1()
    elapsed = round((time.time() - t0) / 60, 1)
    log(f"\n  Total runtime: {elapsed} min")
    log(">>> Stage 4.1 COMPLETE.")
