# -*- coding: utf-8 -*-
"""
=============================================================================
Stage 1: Data Extraction and Structuring
Project : Physiological Emotion Recognition (SEED-IV Dataset)
=============================================================================

SEED-IV Dataset Structure (verified by inspection):
  - 3 sessions  (folders: 1, 2, 3)
  - 15 subjects per session
  - 24 trials per subject per session
  - EEG keys : de_movingAve1 ... de_movingAve24     shape (62, W, 5)
               de_LDS1 ... de_LDS24  (fallback)
  - Eye keys : eye_1 ... eye_24                      shape (31, W)
               ** TRANSPOSED relative to EEG **
  - Labels   : fixed per SEED-IV protocol (hardcoded below)

EEG feature dimensionality: 62 channels x 5 freq bands = 310-D
Eye feature dimensionality: 31-D per window
=============================================================================
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import scipy.io as sio

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# SEED-IV fixed emotion label sequences
# Labels: 0=neutral, 1=sad, 2=fear, 3=happy
# One label per trial; 24 trials per session per subject.
# ---------------------------------------------------------------------------
SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 0, 0, 2, 3, 0, 0, 2, 3, 0, 3, 2, 3, 0, 2, 0],
}

N_EEG_FEATURES = 310   # 62 channels x 5 frequency bands
N_EYE_FEATURES = 31    # 31 eye-movement features
N_TRIALS       = 24    # Fixed by SEED-IV protocol

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = r"c:\Users\Rose J Thachil\Documents\8th sem\dataset_mat_files"
EEG_DIR   = os.path.join(BASE_DIR, "eeg_feature_smooth", "eeg_feature_smooth")
EYE_DIR   = os.path.join(BASE_DIR, "eye_feature_smooth", "eye_feature_smooth")

OUTPUT_DIR    = r"c:\Users\Rose J Thachil\Documents\8th sem"
EEG_CSV_PATH  = os.path.join(OUTPUT_DIR, "eeg_features.csv")
EYE_CSV_PATH  = os.path.join(OUTPUT_DIR, "eye_features.csv")
EEG_XLSX_PATH = os.path.join(OUTPUT_DIR, "eeg_features.xlsx")
EYE_XLSX_PATH = os.path.join(OUTPUT_DIR, "eye_features.xlsx")


# =============================================================================
# FUNCTION 1: load_mat
# =============================================================================
def load_mat(filepath):
    """
    Load a single .mat file using scipy.io.loadmat.

    Parameters
    ----------
    filepath : str  - absolute path to the .mat file

    Returns
    -------
    dict  - {key: array} with MATLAB meta-keys stripped.
    """
    raw = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)

    # Strip MATLAB internal keys (__header__, __version__, __globals__)
    data = {k: v for k, v in raw.items() if not k.startswith("__")}

    print("  [load] " + os.path.basename(filepath)
          + " | keys=" + str(len(data)))
    return data


# =============================================================================
# FUNCTION 2: parse_structure
# =============================================================================
def parse_structure(mat_data, session_id):
    """
    Parse a loaded .mat dict and return per-trial 2-D arrays.

    Detection logic (in priority order):
      EEG files: looks for 'de_movingAve' prefix first, then 'de_LDS'
      Eye files: looks for 'eye_' prefix

    EEG array shape in file : (62, W, 5)  -> flattened to (W, 310)
    Eye array shape in file : (31, W)     -> transposed  to (W,  31)

    Parameters
    ----------
    mat_data   : dict  - output of load_mat()
    session_id : int   - 1, 2, or 3

    Returns
    -------
    dict  {trial_num (1-24): {'data': np.ndarray (W, F), 'label': int}}
    """
    labels   = SESSION_LABELS[session_id]
    all_keys = list(mat_data.keys())

    # ---- Detect key prefix ------------------------------------------------
    def _detect_prefix(candidates):
        """Return the common prefix shared by the 24 trial keys."""
        for prefix in ("de_movingAve", "de_LDS", "eye_"):
            matched = [k for k in candidates if k.startswith(prefix)]
            if matched:
                return prefix
        # Generic fallback: strip trailing digits
        numbered = [k for k in candidates if re.search(r"\d+$", k)]
        if numbered:
            return re.sub(r"\d+$", "", numbered[0])
        return None

    key_prefix = _detect_prefix(all_keys)
    print("         detected prefix: '" + str(key_prefix) + "'")

    trials = {}
    for trial_num in range(1, N_TRIALS + 1):
        key = key_prefix + str(trial_num)
        if key not in mat_data:
            print("         [WARN] key '" + key + "' not found, skipping trial " + str(trial_num))
            continue

        arr = mat_data[key]

        # Ensure plain numpy array
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        # EEG: shape (62, W, 5) -> (W, 310)
        if arr.ndim == 3 and arr.shape[0] == 62 and arr.shape[2] == 5:
            # Transpose: (62, W, 5) -> (W, 62, 5), then reshape -> (W, 310)
            arr = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)

        # Eye: shape (31, W) -> (W, 31)
        elif arr.ndim == 2 and arr.shape[0] == 31:
            arr = arr.T

        # Eye already (W, 31) - no change needed
        elif arr.ndim == 2 and arr.shape[1] == 31:
            pass

        # Unexpected shape: attempt salvage
        else:
            print("         [WARN] unexpected shape " + str(arr.shape)
                  + " for trial " + str(trial_num) + ". Attempting fallback reshape.")
            if arr.ndim > 2:
                arr = arr.reshape(-1, arr.shape[-1])
            # else leave as-is; validation will catch the mismatch

        trials[trial_num] = {
            "data":  arr,                   # (W, F)
            "label": labels[trial_num - 1], # single int label for this trial
        }

    return trials


# =============================================================================
# FUNCTION 3: flatten_windows
# =============================================================================
def flatten_windows(trials, subject_id, session_id, feature_prefix, n_features):
    """
    Convert the trial dict into a flat pandas DataFrame.

    Each row represents ONE temporal window.

    Columns:
      subject_id | session_id | trial_id | window_id | emotion_label
      + <feature_prefix>_0 ... <feature_prefix>_{n_features-1}

    Assumptions
    -----------
    - Windows within a trial are numbered 0, 1, 2, ...
    - All windows in a trial share the same emotion label.
    - If feature count mismatches, the array is padded/truncated (with a warning).

    Parameters
    ----------
    trials         : dict   - output of parse_structure()
    subject_id     : int
    session_id     : int
    feature_prefix : str    - 'eeg' or 'eye'
    n_features     : int    - expected feature count

    Returns
    -------
    pd.DataFrame
    """
    feat_cols = [feature_prefix + "_" + str(i) for i in range(n_features)]
    rows = []

    for trial_id, trial_info in sorted(trials.items()):
        data      = trial_info["data"]   # (W, F)
        label     = trial_info["label"]
        n_windows = data.shape[0]

        # Standardise feature count
        if data.shape[1] != n_features:
            print("    [WARN] S" + str(subject_id) + " Sess" + str(session_id)
                  + " Trial" + str(trial_id) + ": expected " + str(n_features)
                  + " features, got " + str(data.shape[1]) + ". Padding/truncating.")
            if data.shape[1] < n_features:
                pad  = np.zeros((n_windows, n_features - data.shape[1]))
                data = np.hstack([data, pad])
            else:
                data = data[:, :n_features]

        # One row per window
        meta = np.column_stack([
            np.full(n_windows, subject_id),
            np.full(n_windows, session_id),
            np.full(n_windows, trial_id),
            np.arange(n_windows),
            np.full(n_windows, label),
        ])
        meta_df = pd.DataFrame(meta, columns=[
            "subject_id", "session_id", "trial_id", "window_id", "emotion_label"
        ]).astype(int)

        feat_df = pd.DataFrame(data, columns=feat_cols)
        trial_df = pd.concat([meta_df, feat_df], axis=1)
        rows.append(trial_df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline():
    """
    Orchestrate full data extraction across all sessions and subjects.
    Saves eeg_features.csv and eye_features.csv (+ optional .xlsx).
    """
    SEP = "=" * 70
    sep = "-" * 70

    print(SEP)
    print(" STAGE 1 -- DATA EXTRACTION & STRUCTURING")
    print(SEP)

    eeg_dfs = []
    eye_dfs = []

    sessions = sorted(os.listdir(EEG_DIR))   # ['1', '2', '3']
    print("\nFound sessions: " + str(sessions))

    for session_folder in sessions:
        session_id = int(session_folder)

        eeg_session_dir = os.path.join(EEG_DIR, session_folder)
        eye_session_dir = os.path.join(EYE_DIR, session_folder)

        eeg_files = sorted(glob.glob(os.path.join(eeg_session_dir, "*.mat")))
        eye_files = sorted(glob.glob(os.path.join(eye_session_dir, "*.mat")))

        print("\n" + sep)
        print(" Session " + str(session_id)
              + " | EEG files: " + str(len(eeg_files))
              + " | Eye files: " + str(len(eye_files)))
        print(sep)

        # Build subject-number -> filepath lookup
        def _subject_num(filepath):
            """Extract leading numeric ID from filename like '3_20150919.mat'."""
            basename = os.path.basename(filepath)
            match    = re.match(r"^(\d+)_", basename)
            return int(match.group(1)) if match else None

        eeg_lookup = {_subject_num(f): f for f in eeg_files if _subject_num(f) is not None}
        eye_lookup = {_subject_num(f): f for f in eye_files if _subject_num(f) is not None}

        common_subjects = sorted(set(eeg_lookup.keys()) & set(eye_lookup.keys()))
        print(" Subjects with both EEG & Eye data: " + str(common_subjects))

        for subj_num in common_subjects:
            print("\n -- Subject " + str(subj_num) + " (Session " + str(session_id) + ") --")

            # Load and parse EEG
            eeg_mat    = load_mat(eeg_lookup[subj_num])
            eeg_trials = parse_structure(eeg_mat, session_id)
            eeg_df     = flatten_windows(
                eeg_trials, subj_num, session_id,
                feature_prefix="eeg", n_features=N_EEG_FEATURES
            )
            eeg_dfs.append(eeg_df)

            # Load and parse Eye
            eye_mat    = load_mat(eye_lookup[subj_num])
            eye_trials = parse_structure(eye_mat, session_id)
            eye_df     = flatten_windows(
                eye_trials, subj_num, session_id,
                feature_prefix="eye", n_features=N_EYE_FEATURES
            )
            eye_dfs.append(eye_df)

    # -------------------------------------------------------------------------
    # Concatenate all subjects/sessions
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print(" CONCATENATING ALL DATA")
    print(SEP)

    eeg_full = pd.concat(eeg_dfs, ignore_index=True)
    eye_full = pd.concat(eye_dfs, ignore_index=True)
    print("  EEG combined shape: " + str(eeg_full.shape))
    print("  Eye combined shape: " + str(eye_full.shape))

    # -------------------------------------------------------------------------
    # VALIDATION CHECKS
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print(" VALIDATION CHECKS")
    print(SEP)

    def validate(df, name, expected_n_features):
        meta_cols = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]
        feat_cols = [c for c in df.columns if c not in meta_cols]

        feat_ok   = len(feat_cols) == expected_n_features
        label_ok  = df["emotion_label"].isnull().sum() == 0
        tps       = df.groupby(["subject_id", "session_id"])["trial_id"].nunique()
        wpt       = df.groupby(["subject_id", "session_id", "trial_id"])["window_id"].count()

        print("\n[" + name + "]")
        print("  Total shape         : " + str(df.shape))
        print("  Feature columns     : " + str(len(feat_cols))
              + "  (expected " + str(expected_n_features) + ")")
        print("  Subjects            : " + str(sorted(df["subject_id"].unique())))
        print("  Sessions            : " + str(sorted(df["session_id"].unique())))
        print("  Emotion labels      : " + str(sorted(df["emotion_label"].unique()))
              + "  (0=neutral, 1=sad, 2=fear, 3=happy)")
        print("  Missing values      : " + str(df.isnull().sum().sum()))
        print("  Trials/subj/sess    : min=" + str(tps.min())
              + " max=" + str(tps.max())
              + " (expected " + str(N_TRIALS) + ")")
        print("  Windows/trial       : min=" + str(wpt.min())
              + " max=" + str(wpt.max())
              + " mean=" + str(round(wpt.mean(), 1)))
        print("  Feature count OK    : " + ("PASS" if feat_ok  else "FAIL -- MISMATCH!"))
        print("  No missing labels   : " + ("PASS" if label_ok else "FAIL -- missing labels!"))

        return feat_ok and label_ok

    eeg_ok = validate(eeg_full, "EEG DataFrame", N_EEG_FEATURES)
    eye_ok = validate(eye_full, "Eye DataFrame", N_EYE_FEATURES)

    # -------------------------------------------------------------------------
    # ALIGNMENT CHECK - same (subject, session, trial, window) indices?
    # -------------------------------------------------------------------------
    print("\n[Alignment Check]")
    eeg_idx = eeg_full[["subject_id", "session_id", "trial_id", "window_id"]].reset_index(drop=True)
    eye_idx = eye_full[["subject_id", "session_id", "trial_id", "window_id"]].reset_index(drop=True)

    if eeg_idx.equals(eye_idx):
        print("  EEG & Eye window indices are perfectly aligned -- PASS")
    else:
        print("  [WARN] EEG & Eye window indices differ! Check source data.")
        mismatch = eeg_idx.compare(eye_idx)
        print(mismatch.head().to_string())

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print(" EXPORTING FILES")
    print(SEP)

    eeg_full.to_csv(EEG_CSV_PATH, index=False)
    eye_full.to_csv(EYE_CSV_PATH, index=False)
    print("  EEG CSV saved  -> " + EEG_CSV_PATH)
    print("  Eye CSV saved  -> " + EYE_CSV_PATH)

    # Optional Excel export (requires openpyxl; may be slow for large data)
    try:
        eeg_full.to_excel(EEG_XLSX_PATH, index=False)
        eye_full.to_excel(EYE_XLSX_PATH, index=False)
        print("  EEG XLSX saved -> " + EEG_XLSX_PATH)
        print("  Eye XLSX saved -> " + EYE_XLSX_PATH)
    except Exception as exc:
        print("  [WARN] Excel export skipped: " + str(exc))

    # -------------------------------------------------------------------------
    # DATASET SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + SEP)
    print(" DATASET SUMMARY")
    print(SEP)
    print("  Sessions        : " + str(eeg_full["session_id"].nunique()))
    print("  Subjects        : " + str(eeg_full["subject_id"].nunique()) + " (per session)")
    print("  Total EEG rows  : " + str(f"{len(eeg_full):,}") + " windows")
    print("  Total Eye rows  : " + str(f"{len(eye_full):,}") + " windows")
    print("  EEG feature dim : " + str(N_EEG_FEATURES))
    print("  Eye feature dim : " + str(N_EYE_FEATURES))

    label_map = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
    print("\n  EEG label distribution:")
    for lbl, grp in eeg_full.groupby("emotion_label"):
        pct = 100 * len(grp) / len(eeg_full)
        print("    " + str(lbl) + " (" + label_map.get(lbl, "?") + ") : "
              + str(f"{len(grp):,}") + " windows  (" + str(round(pct, 1)) + "%)")

    print()
    if eeg_ok and eye_ok:
        print("  >>> All validation checks PASSED. Data is ready for the ML pipeline.")
    else:
        print("  >>> Some validation checks FAILED. Review warnings above.")

    return eeg_full, eye_full


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    eeg_df, eye_df = run_pipeline()
