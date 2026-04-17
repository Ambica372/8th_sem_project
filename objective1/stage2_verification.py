# -*- coding: utf-8 -*-
"""
=============================================================================
Stage 2 Verification Script
Project : Physiological Emotion Recognition (SEED-IV)
=============================================================================

Runs ALL 5 verification checks described in the checklist:
  1. Training data mean ~0, std ~1  (MOST IMPORTANT)
  2. Test data mean NOT 0, std NOT 1  (confirm no leakage)
  3. Validation subject mean NOT 0, std NOT 1  (CRITICAL)
  4. Per-fold consistency (all folds train mean ~0)
  5. Distribution histogram of first feature (saved as PNG)

Checks both EEG and Eye data separately.
=============================================================================
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no popup windows)
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR  = r"c:\Users\Rose J Thachil\Documents\8th sem\stage2_output"
PLOT_DIR    = os.path.join(OUTPUT_DIR, "verification_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

N_FOLDS = 5
META_COLS = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]

# Acceptable tolerances
MEAN_TOL = 1e-4   # train mean must be within ±this of 0
STD_TOL  = 1e-3   # train std  must be within ±this of 1

SEP   = "=" * 70
SEP2  = "-" * 55


# ---------------------------------------------------------------------------
# Helper: load a CSV and return only feature columns as numpy array
# ---------------------------------------------------------------------------
def load_features(filepath):
    df = pd.read_csv(filepath)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df[feat_cols].values.astype(np.float64), feat_cols


# ---------------------------------------------------------------------------
# Helper: compute and print mean/std summary for first 10 features
# ---------------------------------------------------------------------------
def print_stats(label, arr, n=10):
    col_mean = np.nanmean(arr, axis=0)
    col_std  = np.nanstd(arr,  axis=0, ddof=1)
    print("  " + label)
    print("    mean[:10] = " + str([round(float(v), 6) for v in col_mean[:n]]))
    print("    std [:10] = " + str([round(float(v), 6) for v in col_std[:n]]))
    return col_mean, col_std


# ---------------------------------------------------------------------------
# CHECK 1 + 4: Training data mean ~0, std ~1  (per fold)
# ---------------------------------------------------------------------------
def check_train(modality, fold_results):
    """
    modality : 'eeg' or 'eye'
    fold_results : dict to accumulate per-fold pass/fail
    """
    print(SEP)
    print("CHECK 1 & 4 -- Training Data: mean ~0, std ~1  [" + modality.upper() + "]")
    print(SEP)

    all_pass = True
    for k in range(1, N_FOLDS + 1):
        path = os.path.join(OUTPUT_DIR, "fold_" + str(k) + "_train_" + modality + ".csv")
        arr, _ = load_features(path)

        col_mean = np.nanmean(arr, axis=0)
        col_std  = np.nanstd(arr,  axis=0, ddof=1)

        max_abs_mean = float(np.max(np.abs(col_mean)))
        min_std      = float(np.nanmin(col_std[col_std > 0]))   # ignore zero-var cols
        max_std      = float(np.nanmax(col_std))

        mean_ok = max_abs_mean < MEAN_TOL
        std_ok  = (min_std > 1 - STD_TOL) and (max_std < 1 + STD_TOL)
        fold_pass = mean_ok and std_ok
        all_pass  = all_pass and fold_pass

        status = "PASS" if fold_pass else "FAIL"
        print("\n  Fold " + str(k) + " [" + status + "]")
        print("    Train shape       : " + str(arr.shape))
        print("    Max |mean|        : " + str(round(max_abs_mean, 8))
              + "  (threshold < " + str(MEAN_TOL) + ")"
              + ("  OK" if mean_ok else "  << WRONG"))
        print("    Std range         : [" + str(round(min_std, 6))
              + ", " + str(round(max_std, 6)) + "]"
              + ("  OK" if std_ok else "  << WRONG"))
        print("    mean[:5]  = " + str([round(float(v), 6) for v in col_mean[:5]]))
        print("    std [:5]  = " + str([round(float(v), 6) for v in col_std[:5]]))

        fold_results[modality]["fold_" + str(k)]["train"] = status

    overall = "ALL FOLDS PASS" if all_pass else "SOME FOLDS FAILED"
    print("\n  >> " + modality.upper() + " Train Check: " + overall)
    return all_pass


# ---------------------------------------------------------------------------
# CHECK 2: Test data mean NOT ~0 (confirm no leakage)
# ---------------------------------------------------------------------------
def check_test_leakage(modality, fold_results):
    print(SEP)
    print("CHECK 2 -- Test Data: mean NOT 0, std NOT 1  [" + modality.upper() + "]")
    print("           (If mean ~0 and std ~1 -> LEAKAGE!)")
    print(SEP)

    all_clear = True
    for k in range(1, N_FOLDS + 1):
        path = os.path.join(OUTPUT_DIR, "fold_" + str(k) + "_test_" + modality + ".csv")
        arr, _ = load_features(path)

        col_mean = np.nanmean(arr, axis=0)
        col_std  = np.nanstd(arr, axis=0, ddof=1)

        max_abs_mean = float(np.max(np.abs(col_mean)))
        min_std      = float(np.nanmin(col_std[col_std > 0]))
        max_std      = float(np.nanmax(col_std))

        # LEAKAGE if test mean is suspiciously close to 0 AND std close to 1
        leakage_flag = (max_abs_mean < 1e-6) and (min_std > 0.9999) and (max_std < 1.0001)
        status = "LEAKAGE DETECTED" if leakage_flag else "CLEAN (no leakage)"
        all_clear = all_clear and (not leakage_flag)

        print("\n  Fold " + str(k) + " [" + status + "]")
        print("    Test shape        : " + str(arr.shape))
        print("    Max |mean|        : " + str(round(max_abs_mean, 6))
              + ("  [GOOD: not ~0]" if max_abs_mean > 0.05 else "  [suspicious: too close to 0]"))
        print("    Std range         : [" + str(round(min_std, 4))
              + ", " + str(round(max_std, 4)) + "]")
        print("    mean[:5]  = " + str([round(float(v), 4) for v in col_mean[:5]]))
        print("    std [:5]  = " + str([round(float(v), 4) for v in col_std[:5]]))

        fold_results[modality]["fold_" + str(k)]["test_leakage"] = status

    overall = "NO LEAKAGE DETECTED" if all_clear else "LEAKAGE WARNING!"
    print("\n  >> " + modality.upper() + " Test Leakage Check: " + overall)
    return all_clear


# ---------------------------------------------------------------------------
# CHECK 3: Validation subject mean NOT ~0, std NOT ~1
# ---------------------------------------------------------------------------
def check_validation_leakage(modality, fold_results):
    print(SEP)
    print("CHECK 3 -- Validation Subject: mean NOT 0, std NOT 1  [" + modality.upper() + "]")
    print("           (Proves model sees truly unseen distribution)")
    print(SEP)

    path = os.path.join(OUTPUT_DIR, "validation_" + modality + ".csv")
    arr, _ = load_features(path)

    col_mean = np.nanmean(arr, axis=0)
    col_std  = np.nanstd(arr,  axis=0, ddof=1)

    max_abs_mean = float(np.max(np.abs(col_mean)))
    min_std      = float(np.nanmin(col_std[col_std > 0]))
    max_std      = float(np.nanmax(col_std))

    leakage_flag = (max_abs_mean < 1e-6) and (min_std > 0.9999) and (max_std < 1.0001)
    status = "LEAKAGE DETECTED" if leakage_flag else "CLEAN (unseen distribution confirmed)"

    print("\n  Validation shape  : " + str(arr.shape))
    print("  Max |mean|        : " + str(round(max_abs_mean, 6))
          + ("  [GOOD: not ~0]" if max_abs_mean > 0.05 else "  [suspicious]"))
    print("  Std range         : [" + str(round(min_std, 4)) + ", " + str(round(max_std, 4)) + "]")
    print("  mean[:10] = " + str([round(float(v), 4) for v in col_mean[:10]]))
    print("  std [:10] = " + str([round(float(v), 4) for v in col_std[:10]]))
    print("\n  >> " + modality.upper() + " Validation Check: " + status)

    fold_results[modality]["validation"] = status
    return not leakage_flag


# ---------------------------------------------------------------------------
# CHECK 5: Distribution histogram of first feature across train/test/val
# ---------------------------------------------------------------------------
def check_distribution(modality):
    print(SEP)
    print("CHECK 5 -- Feature Distribution Plot  [" + modality.upper() + "]")
    print(SEP)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(modality.upper() + " -- Feature[0] Distribution (Fold 1)",
                 fontsize=13, fontweight="bold")

    configs = [
        ("fold_1_train_" + modality + ".csv", "Train (Fold 1)", "steelblue",  axes[0]),
        ("fold_1_test_"  + modality + ".csv", "Test  (Fold 1)", "darkorange", axes[1]),
        ("validation_"   + modality + ".csv", "Validation (S15)","green",     axes[2]),
    ]

    for fname, title, color, ax in configs:
        path = os.path.join(OUTPUT_DIR, fname)
        arr, _ = load_features(path)
        feat_values = arr[:, 0]
        feat_clean  = feat_values[~np.isnan(feat_values)]

        m = round(float(np.mean(feat_clean)), 4)
        s = round(float(np.std(feat_clean, ddof=1)), 4)

        ax.hist(feat_clean, bins=60, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="x=0")
        ax.set_title(title + "\nmean=" + str(m) + "  std=" + str(s), fontsize=10)
        ax.set_xlabel("Normalized Feature Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        print("  " + title + ": mean=" + str(m) + "  std=" + str(s)
              + ("  [centered OK]" if abs(m) < 0.5 else "  [offset - expected for test/val]"))

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "dist_feature0_" + modality + ".png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved -> " + plot_path)


# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
def print_summary(fold_results, check_results):
    print(SEP)
    print("VERIFICATION SUMMARY")
    print(SEP)

    labels = {
        "train_ok":     "Check 1+4 (Train mean~0, std~1)",
        "test_ok":      "Check 2   (Test NOT centered - no leakage)",
        "val_ok":       "Check 3   (Val NOT centered  - no leakage)",
    }
    for key, label in labels.items():
        for mod in ("eeg", "eye"):
            result = check_results.get(mod + "_" + key, False)
            status = "PASS" if result else "FAIL"
            print("  [" + status + "] " + mod.upper() + " -- " + label)

    print()
    print("  Per-Fold Detail:")
    for mod in ("eeg", "eye"):
        print("  " + "-" * 50)
        print("  " + mod.upper())
        for k in range(1, N_FOLDS + 1):
            fkey = "fold_" + str(k)
            if fkey in fold_results[mod]:
                tr = fold_results[mod][fkey].get("train", "?")
                tl = fold_results[mod][fkey].get("test_leakage", "?")
                print("    Fold " + str(k)
                      + " | Train: " + tr
                      + " | Test leakage: " + tl)
        val = fold_results[mod].get("validation", "?")
        print("    Validation | " + val)

    print(SEP)
    all_ok = all(check_results.values())
    if all_ok:
        print(">>> ALL CHECKS PASSED. Stage 2 normalization is CORRECT and LEAKAGE-FREE.")
    else:
        print(">>> SOME CHECKS FAILED. Review the output above for details.")
    print(SEP)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(SEP)
    print(" STAGE 2 VERIFICATION -- 5-STEP CHECKLIST")
    print(SEP)

    fold_results = {
        "eeg": {"fold_" + str(k): {} for k in range(1, N_FOLDS + 1)},
        "eye": {"fold_" + str(k): {} for k in range(1, N_FOLDS + 1)},
    }
    check_results = {}

    for mod in ("eeg", "eye"):
        print("\n" + "#" * 70)
        print("# MODALITY: " + mod.upper())
        print("#" * 70)

        check_results[mod + "_train_ok"] = check_train(mod, fold_results)
        check_results[mod + "_test_ok"]  = check_test_leakage(mod, fold_results)
        check_results[mod + "_val_ok"]   = check_validation_leakage(mod, fold_results)
        check_distribution(mod)

    print_summary(fold_results, check_results)
    print("\nPlots saved to: " + PLOT_DIR)


if __name__ == "__main__":
    main()
