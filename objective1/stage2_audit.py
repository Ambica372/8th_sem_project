# -*- coding: utf-8 -*-
"""
=============================================================================
Examiner Audit Script — Stage 2 Verification (Checks A through E)
=============================================================================
Runs every check from the examiner checklist:
  A. Subject leakage (set intersection)
  B. Normalization stats via quick_check() — train/test/val
  C. Feature count (shape verification)
  D. Fold integrity — each subject appears in test exactly once
  E. NaN count preserved before/after normalization (Eye data)
=============================================================================
"""

import os
import sys
import io
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUTPUT_DIR = r"c:\Users\Rose J Thachil\Documents\8th sem\stage2_output"
BASE_DIR   = r"c:\Users\Rose J Thachil\Documents\8th sem"
META_COLS  = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]
N_FOLDS    = 5

SEP  = "=" * 65
SEP2 = "-" * 55

PASS = "PASS"
FAIL = "FAIL"

results = []

def record(check, fold, modality, status, detail=""):
    mark = "OK" if status == PASS else "!!"
    results.append((check, fold, modality, status, detail))
    print("  [" + mark + "] " + check
          + ((" Fold " + str(fold)) if fold else "")
          + (" [" + modality + "]" if modality else "")
          + " -- " + status
          + (": " + detail if detail else ""))


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def load(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    df   = pd.read_csv(path)
    feat = df[[c for c in df.columns if c not in META_COLS]]
    return df, feat


def quick_check(df, label):
    """Return (global_mean, global_std) over all feature columns."""
    feat = df[[c for c in df.columns if c not in META_COLS]]
    gm = float(feat.mean().mean())
    gs = float(feat.std().mean())
    print("    " + label)
    print("      Global mean of column-means : " + str(round(gm, 6)))
    print("      Global mean of column-stds  : " + str(round(gs, 6)))
    return gm, gs


# ---------------------------------------------------------------------------
# CHECK A: Subject leakage — set intersection must be empty
# ---------------------------------------------------------------------------
print(SEP)
print("CHECK A -- Subject Leakage (set intersection)")
print(SEP)

for k in range(1, N_FOLDS + 1):
    for mod in ("eeg", "eye"):
        train_df, _ = load("fold_" + str(k) + "_train_" + mod + ".csv")
        test_df,  _ = load("fold_" + str(k) + "_test_"  + mod + ".csv")

        overlap = set(train_df["subject_id"].unique()) & set(test_df["subject_id"].unique())

        if overlap:
            record("A", k, mod, FAIL, "Overlap subjects: " + str(overlap))
        else:
            record("A", k, mod, PASS, "set() -- no overlap")


# ---------------------------------------------------------------------------
# CHECK B: Normalization stats — quick_check on train / test / val
# ---------------------------------------------------------------------------
print()
print(SEP)
print("CHECK B -- Normalization Stats (train~0/1, test+val NOT 0/1)")
print(SEP)

for k in range(1, N_FOLDS + 1):
    print("\n  -- Fold " + str(k) + " EEG --")
    train_df, _ = load("fold_" + str(k) + "_train_eeg.csv")
    test_df,  _ = load("fold_" + str(k) + "_test_eeg.csv")
    val_df,   _ = load("fold_" + str(k) + "_validation_eeg.csv")

    tm, ts = quick_check(train_df, "Train")
    xm, xs = quick_check(test_df,  "Test ")
    vm, vs = quick_check(val_df,   "Val  ")

    train_ok = abs(tm) < 0.001 and abs(ts - 1.0) < 0.01
    test_ok  = abs(xm) > 0.01 or  abs(xs - 1.0) > 0.01   # NOT centered
    val_ok   = abs(vm) > 0.01 or  abs(vs - 1.0) > 0.01

    record("B-train", k, "eeg", PASS if train_ok else FAIL,
           "mean=" + str(round(tm,4)) + " std=" + str(round(ts,4)))
    record("B-test",  k, "eeg", PASS if test_ok  else FAIL,
           "mean=" + str(round(xm,4)) + " std=" + str(round(xs,4)) + " (NOT 0/1 = good)")
    record("B-val",   k, "eeg", PASS if val_ok   else FAIL,
           "mean=" + str(round(vm,4)) + " std=" + str(round(vs,4)) + " (NOT 0/1 = good)")

print()
for k in range(1, N_FOLDS + 1):
    print("  -- Fold " + str(k) + " Eye --")
    train_df, _ = load("fold_" + str(k) + "_train_eye.csv")
    test_df,  _ = load("fold_" + str(k) + "_test_eye.csv")
    val_df,   _ = load("fold_" + str(k) + "_validation_eye.csv")

    tm, ts = quick_check(train_df, "Train")
    xm, xs = quick_check(test_df,  "Test ")
    vm, vs = quick_check(val_df,   "Val  ")

    train_ok = abs(tm) < 0.001 and abs(ts - 1.0) < 0.01
    test_ok  = abs(xm) > 0.005 or abs(xs - 1.0) > 0.01
    val_ok   = abs(vm) > 0.005 or abs(vs - 1.0) > 0.01

    record("B-train", k, "eye", PASS if train_ok else FAIL,
           "mean=" + str(round(tm,4)) + " std=" + str(round(ts,4)))
    record("B-test",  k, "eye", PASS if test_ok  else FAIL,
           "mean=" + str(round(xm,4)) + " std=" + str(round(xs,4)))
    record("B-val",   k, "eye", PASS if val_ok   else FAIL,
           "mean=" + str(round(vm,4)) + " std=" + str(round(vs,4)))


# ---------------------------------------------------------------------------
# CHECK C: Feature count (shape)
# ---------------------------------------------------------------------------
print()
print(SEP)
print("CHECK C -- Feature Count (EEG=310, Eye=31) + Total Columns")
print(SEP)

expected = {"eeg": (315, 310), "eye": (36, 31)}   # (total_cols, feat_cols)

for k in range(1, N_FOLDS + 1):
    for mod in ("eeg", "eye"):
        exp_total, exp_feat = expected[mod]
        for split in ("train", "test", "validation"):
            df, feat = load("fold_" + str(k) + "_" + split + "_" + mod + ".csv")
            total_c = df.shape[1]
            feat_c  = feat.shape[1]
            ok = (total_c == exp_total) and (feat_c == exp_feat)
            record("C", k, mod + "/" + split, PASS if ok else FAIL,
                   "shape=" + str(df.shape)
                   + " feat=" + str(feat_c) + " (exp " + str(exp_feat) + ")")


# ---------------------------------------------------------------------------
# CHECK D: Fold integrity — each dev subject in test exactly ONCE
# ---------------------------------------------------------------------------
print()
print(SEP)
print("CHECK D -- Fold Integrity (each subject in test exactly once)")
print(SEP)

# Collect test subjects per fold
test_subject_counts = {}  # subject -> count of folds where they are test
for k in range(1, N_FOLDS + 1):
    df, _ = load("fold_" + str(k) + "_train_eeg.csv")
    test_df, _ = load("fold_" + str(k) + "_test_eeg.csv")
    test_subjects  = sorted(test_df["subject_id"].unique())
    train_subjects = sorted(df["subject_id"].unique())
    print("  Fold " + str(k)
          + " | train=" + str(train_subjects)
          + " | test=" + str(test_subjects))
    for s in test_subjects:
        test_subject_counts[s] = test_subject_counts.get(s, 0) + 1

print()
all_ok = True
for s in sorted(test_subject_counts.keys()):
    cnt = test_subject_counts[s]
    ok  = (cnt == 1)
    all_ok = all_ok and ok
    record("D", None, "subj " + str(s), PASS if ok else FAIL,
           "appears in test " + str(cnt) + " fold(s) (expected 1)")

# Also verify subject 15 appears in NO test fold (held-out)
s15_found = any(
    15 in pd.read_csv(
        os.path.join(OUTPUT_DIR, "fold_" + str(k) + "_test_eeg.csv")
    )["subject_id"].values
    for k in range(1, N_FOLDS + 1)
)
record("D", None, "subj 15 isolation",
       FAIL if s15_found else PASS,
       "Subject 15 found in CV test folds: " + str(s15_found))


# ---------------------------------------------------------------------------
# CHECK E: NaN count preserved before/after normalization (Eye data)
# ---------------------------------------------------------------------------
print()
print(SEP)
print("CHECK E -- NaN Count Preserved (Eye Data)")
print(SEP)

eye_raw = pd.read_csv(os.path.join(BASE_DIR, "eye_features.csv"))
raw_nan_total = int(eye_raw.isnull().sum().sum())
print("  Raw eye_features.csv NaN count: " + str(raw_nan_total))

for k in range(1, N_FOLDS + 1):
    for split in ("train", "test", "validation"):
        fname = "fold_" + str(k) + "_" + split + "_eye.csv"
        df = pd.read_csv(os.path.join(OUTPUT_DIR, fname))
        feat_cols = [c for c in df.columns if c not in META_COLS]
        nan_count = int(df[feat_cols].isnull().sum().sum())
        print("  " + fname + "  NaN=" + str(nan_count))

# Validation: sum of all train NaNs + test NaNs must equal raw NaN * 5
# (because each raw window appears in 4 train sets + 1 test set over 5 folds,
#  and additionally in 5 validation sets — total NaN count is predictable)
train_nans = sum(
    int(pd.read_csv(os.path.join(OUTPUT_DIR, "fold_" + str(k) + "_train_eye.csv"))
       [[c for c in pd.read_csv(os.path.join(OUTPUT_DIR,
           "fold_" + str(k) + "_train_eye.csv"), nrows=0).columns
         if c not in META_COLS]].isnull().sum().sum())
    for k in range(1, N_FOLDS + 1)
)
test_nans = sum(
    int(pd.read_csv(os.path.join(OUTPUT_DIR, "fold_" + str(k) + "_test_eye.csv"))
       [[c for c in pd.read_csv(os.path.join(OUTPUT_DIR,
           "fold_" + str(k) + "_test_eye.csv"), nrows=0).columns
         if c not in META_COLS]].isnull().sum().sum())
    for k in range(1, N_FOLDS + 1)
)

# Every dev row appears in exactly 4 train folds + 1 test fold = 5 times
# Validation subject NaNs appear 5 times (once per fold val file)
dev_nan   = int(eye_raw[eye_raw["subject_id"] != 15]
               [[c for c in eye_raw.columns if c not in META_COLS]]
               .isnull().sum().sum())
val_nan   = int(eye_raw[eye_raw["subject_id"] == 15]
               [[c for c in eye_raw.columns if c not in META_COLS]]
               .isnull().sum().sum())

expected_total = dev_nan * 5 + val_nan * 5
actual_total   = train_nans + test_nans + val_nan * 5

print()
print("  Dev NaN (raw):       " + str(dev_nan) + "  x5 = " + str(dev_nan * 5))
print("  Val NaN (raw):       " + str(val_nan) + "  x5 = " + str(val_nan * 5))
print("  Expected total:      " + str(expected_total))
print("  Actual train+test NaN total: " + str(train_nans + test_nans))

nan_ok = (train_nans + test_nans) == dev_nan * 5
record("E", None, "eye NaN preservation",
       PASS if nan_ok else FAIL,
       "train+test NaN=" + str(train_nans + test_nans)
       + " expected=" + str(dev_nan * 5))


# ---------------------------------------------------------------------------
# FINAL AUDIT SUMMARY
# ---------------------------------------------------------------------------
print()
print(SEP)
print("EXAMINER AUDIT SUMMARY")
print(SEP)
print()

passed = [r for r in results if r[3] == PASS]
failed = [r for r in results if r[3] == FAIL]

print("  Total checks : " + str(len(results)))
print("  Passed       : " + str(len(passed)))
print("  Failed       : " + str(len(failed)))

if failed:
    print()
    print("  FAILURES:")
    for r in failed:
        print("    [!!] " + r[0] + " " + str(r[1]) + " " + r[2] + " -- " + r[4])
else:
    print()
    print("  ALL CHECKS PASSED.")
    print("  Stage 2 pipeline is correct, leak-free, and examiner-ready.")

print(SEP)
