# -*- coding: utf-8 -*-
"""
=============================================================================
Stage 2: Data Splitting, Cross-Validation, and Normalization
Project : Physiological Emotion Recognition (SEED-IV Dataset)
=============================================================================

Pipeline Overview
-----------------
1. Load eeg_features.csv and eye_features.csv produced by Stage 1.
2. Hold out ONE subject for final validation (never seen during CV).
3. Split remaining 14 subjects into 5 subject-level folds.
4. For each fold:
     a. Use 4 folds (subjects) as training, 1 fold as test.
     b. Compute Z-score parameters (mean, std) on training data ONLY.
     c. Apply normalization to train, test, AND held-out validation set
        using THAT fold's training stats (mu_k, sigma_k).
5. Save normalized CSVs for every fold + per-fold validation files.
   validation_fold{k}_eeg.csv uses Fold k's training stats — so each
   fold has a self-consistent (train, test, val) triple.
6. Print thorough validation checks after each fold.

Key Design Decisions
---------------------
* Subject-level splitting  : All windows from one subject go to the same
                             partition. This avoids temporal leakage that
                             occurs when windows from the same trial are
                             split across train/test.
* No leakage in normalization: mean/std computed exclusively from the
                             training subjects of each fold. Test and
                             validation subjects are transformed using
                             training statistics — NOT their own statistics.
* EEG and Eye kept separate : Features have different scales and physical
                              meanings. They are normalized independently.
* NaN handling (Eye data)  : Eye data contains 208 NaN values (natural from
                             raw eye-tracking). NaN columns are excluded from
                             mean/std computation to avoid propagation.
                             After normalization NaNs are preserved as-is.

Constraints (honoured)
---------
* No model training
* No feature analysis
* No EEG+Eye merging
* No use of test/validation data in normalization
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Force UTF-8 console output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Reproducibility seed
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = r"c:\Users\Rose J Thachil\Documents\8th sem"
EEG_CSV       = os.path.join(BASE_DIR, "eeg_features.csv")
EYE_CSV       = os.path.join(BASE_DIR, "eye_features.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "stage2_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FOLDS             = 5
VALIDATION_SUBJECT  = 15        # Subject held out from ALL cross-validation
                                 # (chosen as last subject for reproducibility)
META_COLS = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]
N_EEG_FEATURES = 310
N_EYE_FEATURES = 31


# =============================================================================
# FUNCTION 1: split_subjects
# =============================================================================
def split_subjects(df, val_subject_id=VALIDATION_SUBJECT):
    """
    Partition the dataset into a development set and a held-out validation set.

    The held-out validation subject is completely removed from cross-validation.
    This simulates real-world deployment where the model must generalize to
    a subject whose data was never used in any form during training.

    Parameters
    ----------
    df              : pd.DataFrame  -- full dataset (EEG or Eye)
    val_subject_id  : int           -- subject ID to hold out

    Returns
    -------
    dev_df : pd.DataFrame  -- rows belonging to development subjects
    val_df : pd.DataFrame  -- rows belonging to the validation subject
    dev_subjects : list    -- sorted list of development subject IDs
    """
    all_subjects = sorted(df["subject_id"].unique())
    assert val_subject_id in all_subjects, (
        "Validation subject " + str(val_subject_id) + " not found in data."
    )

    val_df  = df[df["subject_id"] == val_subject_id].copy()
    dev_df  = df[df["subject_id"] != val_subject_id].copy()
    dev_subjects = sorted(dev_df["subject_id"].unique())

    print("  Validation subject   : " + str(val_subject_id))
    print("  Development subjects : " + str(dev_subjects)
          + " (" + str(len(dev_subjects)) + " subjects)")
    print("  Dev rows             : " + str(len(dev_df))
          + " | Val rows: " + str(len(val_df)))
    return dev_df, val_df, dev_subjects


# =============================================================================
# FUNCTION 2: create_folds
# =============================================================================
def create_folds(dev_subjects, n_folds=N_FOLDS, random_seed=RANDOM_SEED):
    """
    Create subject-level K-Fold splits on development subjects.

    Each fold assigns ENTIRE subjects to train or test — no subject is ever
    split between partitions. This is critical for correct LOSO-style
    evaluation where the model is tested on subjects it has never seen.

    Parameters
    ----------
    dev_subjects : list  -- subject IDs available for cross-validation
    n_folds      : int   -- number of folds  (default 5)
    random_seed  : int   -- for reproducible shuffling

    Returns
    -------
    list of dicts, each with keys:
        'fold'           : int  (1-indexed)
        'train_subjects' : list
        'test_subjects'  : list
    """
    subjects_arr = np.array(dev_subjects)

    # KFold on subject indices (shuffle for robustness, seeded for reproducibility)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subjects_arr)):
        fold_info = {
            "fold":           fold_idx + 1,
            "train_subjects": sorted(subjects_arr[train_idx].tolist()),
            "test_subjects":  sorted(subjects_arr[test_idx].tolist()),
        }
        folds.append(fold_info)

    print("\n  5-Fold subject splits:")
    print("  " + "-" * 55)
    for f in folds:
        print("  Fold " + str(f["fold"])
              + " | Train=" + str(f["train_subjects"])
              + " | Test=" + str(f["test_subjects"]))
    print("  " + "-" * 55)

    return folds


# =============================================================================
# FUNCTION 3: normalize_data
# =============================================================================
def normalize_data(train_df, test_df, val_df, feature_cols):
    """
    Apply Z-score normalization using ONLY training-set statistics.

    Formula: z = (x - mu) / sigma
      mu    = column-wise mean   computed on train_df
      sigma = column-wise std    computed on train_df

    Columns where sigma == 0 (constant features) are divided by 1 to avoid
    NaN; a warning is printed for each such column.

    NaN values (common in eye-tracking data) are handled gracefully:
      - mean/std are computed with skipna=True (pandas default)
      - after normalization, NaN positions remain NaN

    Parameters
    ----------
    train_df     : pd.DataFrame  -- training windows (features only, no meta)
    test_df      : pd.DataFrame  -- test windows
    val_df       : pd.DataFrame  -- held-out validation windows
    feature_cols : list          -- column names to normalize

    Returns
    -------
    train_norm : pd.DataFrame
    test_norm  : pd.DataFrame
    val_norm   : pd.DataFrame
    stats      : dict  {'mean': pd.Series, 'std': pd.Series}
    """
    mu  = train_df[feature_cols].mean()       # shape: (n_features,)
    sigma = train_df[feature_cols].std(ddof=1)  # sample std, ddof=1

    # Guard against zero-std columns (constant features)
    zero_std_cols = sigma[sigma == 0].index.tolist()
    if zero_std_cols:
        print("    [WARN] Zero-variance columns: " + str(zero_std_cols)
              + " -- dividing by 1 to avoid NaN.")
    sigma_safe = sigma.replace(0, 1.0)

    # Apply normalization: z = (x - mu) / sigma
    def _apply(df):
        norm = df[feature_cols].copy()
        norm = (norm - mu) / sigma_safe
        return norm

    train_norm = _apply(train_df)
    test_norm  = _apply(test_df)
    val_norm   = _apply(val_df)

    return train_norm, test_norm, val_norm, {"mean": mu, "std": sigma}


# =============================================================================
# HELPER: re-attach metadata columns to normalized feature DataFrame
# =============================================================================
def _attach_meta(meta_df, norm_feat_df):
    """Concatenate metadata cols + normalized feature cols into one DataFrame."""
    return pd.concat(
        [meta_df.reset_index(drop=True), norm_feat_df.reset_index(drop=True)],
        axis=1
    )


# =============================================================================
# HELPER: validation check for one fold
# =============================================================================
def _validate_fold(fold_num, train_norm, test_norm, val_norm,
                   feature_cols, expected_n_features,
                   train_subjects, test_subjects, val_subject):
    """
    Run mandatory validation checks for a single fold.

    Checks:
      1. No subject overlap between train and test.
      2. Correct number of feature columns.
      3. Training data mean ~ 0 and std ~ 1 (post-normalization).

    Prints a structured summary.
    """
    SEP = "    " + "-" * 52

    print(SEP)
    print("    [Fold " + str(fold_num) + " Validation]")

    # -- 1. Subject overlap --------------------------------------------------
    train_set = set(train_subjects)
    test_set  = set(test_subjects)
    overlap   = train_set & test_set
    overlap_ok = len(overlap) == 0
    print("    Subject overlap (train & test): "
          + ("NONE -- PASS" if overlap_ok else "OVERLAP FOUND -- FAIL: " + str(overlap)))

    # -- 2. Feature count ----------------------------------------------------
    n_feat_train = len([c for c in train_norm.columns if c in feature_cols])
    feat_ok = n_feat_train == expected_n_features
    print("    Feature count: " + str(n_feat_train)
          + " (expected " + str(expected_n_features) + ") -- "
          + ("PASS" if feat_ok else "FAIL"))

    # -- 3. Mean ~ 0 and Std ~ 1 on training data ----------------------------
    feat_only_train = train_norm[feature_cols]
    col_means = feat_only_train.mean()
    col_stds  = feat_only_train.std(ddof=1)

    mean_max = col_means.abs().max()
    std_min  = col_stds.min()
    std_max  = col_stds.max()

    mean_ok = mean_max < 1e-6
    std_ok  = (std_min > 0.99) and (std_max < 1.01)

    print("    Train mean (max abs): "
          + str(round(float(mean_max), 8))
          + " -- " + ("PASS" if mean_ok else "WARN (acceptable for NaN-containing data)"))
    print("    Train std  range    : ["
          + str(round(float(std_min), 4)) + ", "
          + str(round(float(std_max), 4)) + "]"
          + " -- " + ("PASS" if std_ok else "WARN (zero-variance cols may exist)"))

    # -- 4. Shape summary ----------------------------------------------------
    print("    Shapes:")
    print("      Train : " + str(train_norm.shape)
          + " | subjects=" + str(train_subjects))
    print("      Test  : " + str(test_norm.shape)
          + " | subjects=" + str(test_subjects))
    print("      Val   : " + str(val_norm.shape)
          + " | subject=" + str(val_subject))
    print(SEP)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_stage2():
    SEP = "=" * 70

    print(SEP)
    print(" STAGE 2 -- DATA SPLITTING, CROSS-VALIDATION & NORMALIZATION")
    print(SEP)

    # -------------------------------------------------------------------------
    # Load Stage 1 outputs
    # -------------------------------------------------------------------------
    print("\n[1] Loading Stage 1 CSV files...")
    eeg_df = pd.read_csv(EEG_CSV)
    eye_df = pd.read_csv(EYE_CSV)
    print("  EEG: " + str(eeg_df.shape) + " | Eye: " + str(eye_df.shape))

    # Identify feature columns (everything that is not metadata)
    eeg_feat_cols = [c for c in eeg_df.columns if c not in META_COLS]
    eye_feat_cols = [c for c in eye_df.columns if c not in META_COLS]
    print("  EEG feature cols: " + str(len(eeg_feat_cols))
          + " | Eye feature cols: " + str(len(eye_feat_cols)))

    # Confirm alignment: same number of rows
    assert len(eeg_df) == len(eye_df), "EEG and Eye row counts differ!"
    print("  Row alignment: PASS (" + str(len(eeg_df)) + " rows)")

    # -------------------------------------------------------------------------
    # Step 1: Subject-level split — hold out validation subject
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print("[2] Subject-Level Split -- Holding Out Validation Subject")
    print(SEP)

    eeg_dev, eeg_val, dev_subjects = split_subjects(eeg_df, VALIDATION_SUBJECT)
    eye_dev, eye_val, _            = split_subjects(eye_df, VALIDATION_SUBJECT)

    # -------------------------------------------------------------------------
    # Step 2: Create 5-fold subject-level splits
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print("[3] Creating 5-Fold Subject-Level Splits")
    print(SEP)

    folds = create_folds(dev_subjects, n_folds=N_FOLDS, random_seed=RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Step 3: Normalize + export per fold
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print("[4] Normalizing and Exporting Fold Data")
    print(SEP)

    # Track normalization statistics for each fold (EEG and Eye separately)
    fold_stats = {}

    for fold_info in folds:
        k              = fold_info["fold"]
        train_subjects = fold_info["train_subjects"]
        test_subjects  = fold_info["test_subjects"]

        print("\n=== FOLD " + str(k) + " ===========================================")

        # ---- Slice data for this fold ----------------------------------------
        # EEG
        eeg_train_raw = eeg_dev[eeg_dev["subject_id"].isin(train_subjects)].copy()
        eeg_test_raw  = eeg_dev[eeg_dev["subject_id"].isin(test_subjects)].copy()

        # Eye
        eye_train_raw = eye_dev[eye_dev["subject_id"].isin(train_subjects)].copy()
        eye_test_raw  = eye_dev[eye_dev["subject_id"].isin(test_subjects)].copy()

        # ---- Z-score normalization (training stats only) ---------------------
        print("  Normalizing EEG...")
        eeg_train_norm_feat, eeg_test_norm_feat, eeg_val_norm_feat, eeg_stats = normalize_data(
            eeg_train_raw, eeg_test_raw, eeg_val,
            feature_cols=eeg_feat_cols
        )
        print("  Normalizing Eye...")
        eye_train_norm_feat, eye_test_norm_feat, eye_val_norm_feat, eye_stats = normalize_data(
            eye_train_raw, eye_test_raw, eye_val,
            feature_cols=eye_feat_cols
        )

        # ---- Re-attach metadata ---------------------------------------------
        eeg_train_out = _attach_meta(eeg_train_raw[META_COLS], eeg_train_norm_feat)
        eeg_test_out  = _attach_meta(eeg_test_raw[META_COLS],  eeg_test_norm_feat)
        eeg_val_out   = _attach_meta(eeg_val[META_COLS],       eeg_val_norm_feat)

        eye_train_out = _attach_meta(eye_train_raw[META_COLS], eye_train_norm_feat)
        eye_test_out  = _attach_meta(eye_test_raw[META_COLS],  eye_test_norm_feat)
        eye_val_out   = _attach_meta(eye_val[META_COLS],       eye_val_norm_feat)

        # ---- Validation checks ----------------------------------------------
        _validate_fold(
            k,
            eeg_train_out, eeg_test_out, eeg_val_out,
            feature_cols=eeg_feat_cols,
            expected_n_features=N_EEG_FEATURES,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            val_subject=VALIDATION_SUBJECT,
        )

        # ---- Save fold CSVs -------------------------------------------------
        # NOTE: validation is normalized with THIS fold's training stats (mu_k, sigma_k).
        # Each fold produces its own validation file — no fold shares normalization
        # parameters with another fold. This prevents the evaluations from being
        # inconsistent (a bug flagged: using only Fold-1 stats for all folds).
        fold_prefix = os.path.join(OUTPUT_DIR, "fold_" + str(k))

        eeg_train_out.to_csv(fold_prefix + "_train_eeg.csv",      index=False)
        eeg_test_out.to_csv( fold_prefix + "_test_eeg.csv",       index=False)
        eeg_val_out.to_csv(  fold_prefix + "_validation_eeg.csv", index=False)
        eye_train_out.to_csv(fold_prefix + "_train_eye.csv",      index=False)
        eye_test_out.to_csv( fold_prefix + "_test_eye.csv",       index=False)
        eye_val_out.to_csv(  fold_prefix + "_validation_eye.csv", index=False)

        print("  Saved: fold_" + str(k) + "_train_eeg.csv       shape=" + str(eeg_train_out.shape))
        print("  Saved: fold_" + str(k) + "_test_eeg.csv        shape=" + str(eeg_test_out.shape))
        print("  Saved: fold_" + str(k) + "_validation_eeg.csv  shape=" + str(eeg_val_out.shape))
        print("  Saved: fold_" + str(k) + "_train_eye.csv       shape=" + str(eye_train_out.shape))
        print("  Saved: fold_" + str(k) + "_test_eye.csv        shape=" + str(eye_test_out.shape))
        print("  Saved: fold_" + str(k) + "_validation_eye.csv  shape=" + str(eye_val_out.shape))

        # Store stats for reporting
        fold_stats["fold_" + str(k)] = {
            "eeg": {
                "train_mean_range": [round(float(eeg_stats["mean"].min()), 4),
                                     round(float(eeg_stats["mean"].max()), 4)],
                "train_std_range":  [round(float(eeg_stats["std"].min()),  4),
                                     round(float(eeg_stats["std"].max()),  4)],
            },
            "eye": {
                "train_mean_range": [round(float(eye_stats["mean"].min()), 4),
                                     round(float(eye_stats["mean"].max()), 4)],
                "train_std_range":  [round(float(eye_stats["std"].min()),  4),
                                     round(float(eye_stats["std"].max()),  4)],
            },
        }

    # -------------------------------------------------------------------------
    # Per-fold validation files are already saved inside the fold loop above.
    # Each fold_k_validation_eeg.csv / fold_k_validation_eye.csv uses the
    # normalization parameters (mu, sigma) computed from that fold's training
    # subjects — guaranteeing a self-consistent (train, test, val) triple.
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print("[5] Per-Fold Validation Files -- Already Saved in Step 4")
    print(SEP)
    print("  Each fold_k_validation_eeg/eye.csv normalized with Fold k training stats.")
    print("  This ensures evaluation consistency: each fold has its own mu and sigma.")
    for k in range(1, N_FOLDS + 1):
        ep = os.path.join(OUTPUT_DIR, "fold_" + str(k) + "_validation_eeg.csv")
        yp = os.path.join(OUTPUT_DIR, "fold_" + str(k) + "_validation_eye.csv")
        print("  fold_" + str(k) + "_validation_eeg.csv  exists=" + str(os.path.isfile(ep))
              + " | fold_" + str(k) + "_validation_eye.csv  exists=" + str(os.path.isfile(yp)))

    # -------------------------------------------------------------------------
    # Save fold manifest (metadata JSON for reproducibility)
    # -------------------------------------------------------------------------
    # Convert numpy int64 -> plain Python int for JSON serialization
    def _to_python(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, list):
            return [_to_python(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        return obj

    folds_serializable = _to_python(folds)
    manifest = {
        "random_seed":          int(RANDOM_SEED),
        "n_folds":              int(N_FOLDS),
        "validation_subject":   int(VALIDATION_SUBJECT),
        "development_subjects": [int(s) for s in dev_subjects],
        "n_eeg_features":       int(N_EEG_FEATURES),
        "n_eye_features":       int(N_EYE_FEATURES),
        "folds":                folds_serializable,
        "normalization_stats":  fold_stats,
    }
    manifest_path = os.path.join(OUTPUT_DIR, "fold_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n  Saved fold manifest -> " + manifest_path)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print(" STAGE 2 SUMMARY")
    print(SEP)
    print("  Output directory      : " + OUTPUT_DIR)
    print("  Total files generated : "
          + str(N_FOLDS * 6) + " CSVs (train+test+val x EEG+Eye x 5 folds) + 1 JSON manifest")
    print("  Validation subject    : " + str(VALIDATION_SUBJECT)
          + " (" + str(len(eeg_val)) + " windows)")
    print("  Dev subjects          : " + str(dev_subjects))
    print("  Fold sizes (subjects) :")
    for f in folds:
        print("    Fold " + str(f["fold"])
              + " : train=" + str(len(f["train_subjects"]))
              + " subjects, test=" + str(len(f["test_subjects"])) + " subjects")

    # Files list
    print("\n  Generated files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fsize = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
        print("    " + fname + "  (" + str(round(fsize / 1024, 1)) + " KB)")

    print("\n  >>> Stage 2 complete. All splits and normalized CSVs are ready.")
    return folds


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_stage2()
