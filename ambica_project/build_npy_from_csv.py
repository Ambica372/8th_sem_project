# -*- coding: utf-8 -*-
"""
=============================================================================
build_npy_from_csv.py
=============================================================================
Reads eeg_features.csv and eye_features.csv from not_great_obj1/ and
produces the four .npy files required by stratified_pipeline_new.py:

    processed_data/
        X_fused.npy      — EEG (PCA) + Eye concatenated  [N, eeg_pca_dim + eye_dim]
        X_eeg_pca.npy    — EEG features after PCA         [N, n_components]
        X_eye_clean.npy  — Eye features (NaN → 0)         [N, 31]
        y.npy            — Emotion labels (0-3)            [N]

Subject IDs are kept in their original order so downstream LOSO
positional assignment (np.repeat) remains valid.

Run from: ambica_project/
    python build_npy_from_csv.py
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(SCRIPT_DIR, "not_great_obj1")
EEG_CSV      = os.path.join(SRC_DIR, "eeg_features.csv")
EYE_CSV      = os.path.join(SRC_DIR, "eye_features.csv")
OUT_DIR      = os.path.join(SCRIPT_DIR, "processed_data")

os.makedirs(OUT_DIR, exist_ok=True)

# Metadata columns shared by both CSVs
META_COLS = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]

# PCA components to keep for EEG (310 raw → 50 compressed)
EEG_PCA_COMPONENTS = 50

# ---------------------------------------------------------------------------
# Step 1 — Load CSVs
# ---------------------------------------------------------------------------
print("=" * 65)
print(" BUILD NPY FROM CSV")
print("=" * 65)

print("\n[1] Loading CSVs ...")
eeg_df = pd.read_csv(EEG_CSV)
eye_df = pd.read_csv(EYE_CSV)
print(f"   EEG: {eeg_df.shape}  |  Eye: {eye_df.shape}")

assert len(eeg_df) == len(eye_df), \
    "Row count mismatch: EEG={} Eye={}".format(len(eeg_df), len(eye_df))

# Sort by subject → session → trial → window to ensure consistent ordering
sort_keys = ["subject_id", "session_id", "trial_id", "window_id"]
eeg_df = eeg_df.sort_values(sort_keys).reset_index(drop=True)
eye_df = eye_df.sort_values(sort_keys).reset_index(drop=True)

# Sanity-check labels are identical after sorting
assert (eeg_df["emotion_label"].values == eye_df["emotion_label"].values).all(), \
    "emotion_label mismatch between EEG and Eye after sort!"

print(f"   Subjects  : {sorted(eeg_df['subject_id'].unique())}")
print(f"   Classes   : {sorted(eeg_df['emotion_label'].unique())}")
print(f"   Total rows: {len(eeg_df)}")

# ---------------------------------------------------------------------------
# Step 2 — Extract feature arrays
# ---------------------------------------------------------------------------
print("\n[2] Extracting feature columns ...")

eeg_feat_cols = [c for c in eeg_df.columns if c not in META_COLS]
eye_feat_cols = [c for c in eye_df.columns if c not in META_COLS]
print(f"   EEG features: {len(eeg_feat_cols)}  |  Eye features: {len(eye_feat_cols)}")

eeg_raw = eeg_df[eeg_feat_cols].values.astype(np.float32)   # [N, 310]
eye_raw = eye_df[eye_feat_cols].values.astype(np.float32)   # [N, 31]
y_raw   = eeg_df["emotion_label"].values.astype(np.int64)    # [N]

# ---------------------------------------------------------------------------
# Step 3 — Remove rows with NaN/Inf in EEG
# ---------------------------------------------------------------------------
print("\n[3] Cleaning bad rows ...")

bad_eeg = (np.isnan(eeg_raw).any(axis=1) | np.isinf(eeg_raw).any(axis=1))
bad_eye = (np.isnan(eye_raw).any(axis=1) | np.isinf(eye_raw).any(axis=1))

# For Eye: NaN is acceptable (pupil blink etc.) — fill with 0 instead of dropping
print(f"   NaN/Inf rows in EEG: {bad_eeg.sum()}")
print(f"   NaN/Inf rows in Eye: {bad_eye.sum()}  -> will fill with 0")

# Fill Eye NaNs with 0 (column median would be better but adds leakage risk;
# 0 is safe since data is not yet scaled)
eye_raw = np.where(np.isnan(eye_raw) | np.isinf(eye_raw), 0.0, eye_raw)

# Drop rows where EEG itself is corrupt
keep = ~bad_eeg
eeg_raw = eeg_raw[keep]
eye_raw = eye_raw[keep]
y_raw   = y_raw[keep]

n_removed = int((~keep).sum())
print(f"   Removed {n_removed} corrupted EEG rows -> {len(y_raw)} clean samples")

# ---------------------------------------------------------------------------
# Step 4 — Scale EEG, then apply PCA
# ---------------------------------------------------------------------------
print(f"\n[4] Scaling EEG + PCA (n_components={EEG_PCA_COMPONENTS}) ...")

scaler_eeg = StandardScaler()
eeg_scaled = scaler_eeg.fit_transform(eeg_raw)   # [N, 310]

pca = PCA(n_components=EEG_PCA_COMPONENTS, random_state=42)
X_eeg_pca = pca.fit_transform(eeg_scaled).astype(np.float32)  # [N, 50]

explained = pca.explained_variance_ratio_.sum() * 100
print(f"   Variance explained by {EEG_PCA_COMPONENTS} components: {explained:.2f}%")

# ---------------------------------------------------------------------------
# Step 5 — Scale Eye features
# ---------------------------------------------------------------------------
print("\n[5] Scaling Eye features ...")

scaler_eye = StandardScaler()
X_eye_clean = scaler_eye.fit_transform(eye_raw).astype(np.float32)  # [N, 31]

# ---------------------------------------------------------------------------
# Step 6 — Build fused array
# ---------------------------------------------------------------------------
print("\n[6] Building fused array ...")

X_fused = np.concatenate([X_eeg_pca, X_eye_clean], axis=1).astype(np.float32)
print(f"   X_fused shape   : {X_fused.shape}")
print(f"   X_eeg_pca shape : {X_eeg_pca.shape}")
print(f"   X_eye_clean shape: {X_eye_clean.shape}")
print(f"   y shape          : {y_raw.shape}")

# ---------------------------------------------------------------------------
# Step 7 — Verify no remaining NaN/Inf
# ---------------------------------------------------------------------------
print("\n[7] Final NaN/Inf check ...")
for name, arr in [("X_fused", X_fused), ("X_eeg_pca", X_eeg_pca), ("X_eye_clean", X_eye_clean)]:
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    status  = "OK" if not has_nan and not has_inf else "FAIL"
    print(f"   {name:<18} NaN={has_nan}  Inf={has_inf}  [{status}]")

# ---------------------------------------------------------------------------
# Step 8 — Save
# ---------------------------------------------------------------------------
print(f"\n[8] Saving .npy files to: {OUT_DIR}")

np.save(os.path.join(OUT_DIR, "X_fused.npy"),    X_fused)
np.save(os.path.join(OUT_DIR, "X_eeg_pca.npy"),  X_eeg_pca)
np.save(os.path.join(OUT_DIR, "X_eye_clean.npy"), X_eye_clean)
np.save(os.path.join(OUT_DIR, "y.npy"),           y_raw)

for fname in ["X_fused.npy", "X_eeg_pca.npy", "X_eye_clean.npy", "y.npy"]:
    fpath = os.path.join(OUT_DIR, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"   Saved: {fname:<22} ({size_mb:.2f} MB)")

print("\n" + "=" * 65)
print(" DONE — processed_data/ is ready for stratified_pipeline_new.py")
print("=" * 65)
