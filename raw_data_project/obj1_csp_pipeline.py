"""
=============================================================================
SCRIPT 1: CSP + ML PIPELINE ON RAW EEG DATA (SEED-IV) — MEMORY-SAFE
=============================================================================
MEMORY OPTIMIZATIONS APPLIED:
  1. Lazy session loading: one session at a time, freed after each fold batch
  2. float32 everywhere (halves RAM vs float64)
  3. In-place overwrites; del + gc.collect() after every major step
  4. Preprocessing done inside CV loop per fold — no global copy kept
  5. RAM monitoring via psutil at every checkpoint
  6. X_raw, X_pre, X_feat never coexist — overwritten sequentially
  7. GroupKFold indices computed once; slicing done lazily per fold

HOW TO USE LOCALLY:
  pip install mne scikit-learn scipy matplotlib numpy pandas reportlab psutil
  Set DATASET_ROOT and OUTPUT_DIR in SECTION 1 to your local paths.
  Run: python obj1_csp_pipeline.py
=============================================================================
"""

# =============================================================================
# SECTION 0 — IMPORTS
# =============================================================================
# Run once if not installed:
#   pip install mne scikit-learn scipy matplotlib numpy pandas reportlab psutil
import subprocess, sys

import os, gc, json, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import psutil                       # RAM monitoring
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import mne
from mne.decoding import CSP

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================
# --- LOCAL PATHS — edit these two lines to match your machine ---
DATASET_ROOT = r"C:\Users\Rose J Thachil\Desktop\8th_sem_project\raw_data_project\dataset\eeg_raw_data"
OUTPUT_DIR   = r"C:\Users\Rose J Thachil\Desktop\8th_sem_project\raw_data_project\outputs_obj1"
# ---------------------------------------------------------------

CONFIG = {
    "dataset_root"  : DATASET_ROOT,
    "output_dir"    : OUTPUT_DIR,
    "n_components"  : 4,          # CSP components (keep low to save memory)
    "sfreq"         : 200,
    "epoch_len_sec" : 4,
    "epoch_overlap" : 0.5,
    "n_channels"    : 62,
    "n_folds"       : 5,
    "n_classes"     : 4,
    "random_state"  : 42,
    "timestamp"     : datetime.now().strftime("%Y%m%d_%H%M%S"),
}

SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

OUT = Path(CONFIG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

with open(OUT / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

# =============================================================================
# SECTION 2 — LOGGING + RAM MONITORING
# =============================================================================
log_records = []

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    log_records.append({"timestamp": ts, "level": level, "message": msg})
    print(f"[{ts}] [{level}] {msg}")

def ram_mb():
    """Return current process RAM usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def checkpoint(label):
    """Print RAM usage at a named checkpoint."""
    log(f"[RAM] {label}: {ram_mb():.1f} MB used | "
        f"System available: {psutil.virtual_memory().available/1024**2:.0f} MB")

def save_log():
    pd.DataFrame(log_records).to_csv(OUT / "run_log.csv", index=False)

# =============================================================================
# SECTION 3 — LAZY DATA LOADING (one session at a time)
# =============================================================================

def load_eeg_mat(filepath):
    """
    Load raw EEG from a single SEED-IV .mat file.
    Returns list of trial arrays (n_channels, n_timepoints) as float32.
    float32 chosen deliberately — halves RAM vs float64 with no accuracy loss
    for CSP/bandpass pipelines.
    """
    mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    trial_keys = sorted(k for k in mat.keys() if not k.startswith("__"))
    trials = []
    for key in trial_keys:
        data = mat[key]
        if isinstance(data, np.ndarray) and data.ndim == 2:
            if data.shape[0] > data.shape[1]:
                data = data.T                       # → (n_channels, time)
            # ★ Cast to float32 immediately on load — no float64 retained
            trials.append(data.astype(np.float32, copy=False))
        del data
    del mat
    gc.collect()                                    # free .mat file memory now
    return trials


def sliding_epochs(trial, sfreq, epoch_len_sec, overlap):
    """
    Slice one trial into fixed-length epochs.
    Returns float32 array (n_epochs, n_channels, n_samples).
    """
    n_ch, n_time = trial.shape
    ep_len = int(sfreq * epoch_len_sec)
    step   = int(ep_len * (1 - overlap))
    epochs = [trial[:, s:s + ep_len] for s in range(0, n_time - ep_len + 1, step)]
    if not epochs:
        return None
    return np.stack(epochs, axis=0)   # already float32 from load_eeg_mat


def iter_sessions(dataset_root, config):
    """
    Generator — yields (X_session, y_session, groups_session) ONE SESSION AT A
    TIME. Each session is freed from memory before the next is loaded.
    This ensures only ~1 session worth of raw EEG lives in RAM at once.
    """
    root   = Path(dataset_root)
    sfreq  = config["sfreq"]
    ep_len = config["epoch_len_sec"]
    ovlap  = config["epoch_overlap"]

    for session_id in [1, 2, 3]:
        session_dir = root / str(session_id)
        if not session_dir.exists():
            log(f"Session dir missing: {session_dir}", "WARN")
            continue

        labels    = SESSION_LABELS[session_id]
        mat_files = sorted(session_dir.glob("*.mat"))
        log(f"Session {session_id}: {len(mat_files)} subjects found")
        checkpoint(f"Before Session {session_id}")

        X_list, y_list, g_list = [], [], []

        for mat_path in mat_files:
            subj_id = int(mat_path.stem.split("_")[0])
            log(f"  Loading subj {subj_id} | {mat_path.name}")
            try:
                trials = load_eeg_mat(mat_path)
            except Exception as e:
                log(f"  ERROR {mat_path.name}: {e}", "ERROR")
                continue

            n_trials = min(len(trials), len(labels))
            for t_idx in range(n_trials):
                trial = trials[t_idx]
                if trial.ndim != 2 or trial.shape[0] < 2:
                    continue
                epochs = sliding_epochs(trial, sfreq, ep_len, ovlap)
                if epochs is None or len(epochs) == 0:
                    continue
                X_list.append(epochs)
                label = labels[t_idx]
                y_list.extend([label] * len(epochs))
                g_list.extend([subj_id] * len(epochs))
                # ★ Free trial immediately after epoching — no dual copies
                del epochs, trial
            del trials
            gc.collect()

        if not X_list:
            log(f"Session {session_id}: no data, skipping", "WARN")
            continue

        X_sess = np.concatenate(X_list, axis=0)    # (N, C, T) float32
        y_sess = np.array(y_list, dtype=np.int32)
        g_sess = np.array(g_list, dtype=np.int32)

        # ★ Free per-subject lists before yielding the consolidated array
        del X_list, y_list, g_list
        gc.collect()
        checkpoint(f"After Session {session_id} assembled")

        yield X_sess, y_sess, g_sess

        # ★ Caller must del X_sess after use; we del refs here too
        del X_sess, y_sess, g_sess
        gc.collect()
        checkpoint(f"After Session {session_id} freed")


def load_all_subjects(dataset_root, config):
    """
    Collect data from all sessions by consuming the generator.
    Still builds one combined array — needed for GroupKFold splits.
    RAM peak = size of full dataset (unavoidable for CV index generation).
    For very large datasets consider per-session CV instead.
    """
    X_parts, y_parts, g_parts = [], [], []
    for X_s, y_s, g_s in iter_sessions(dataset_root, config):
        X_parts.append(X_s)
        y_parts.append(y_s)
        g_parts.append(g_s)
        # ★ X_s reference kept in list; session generator already freed its copy

    if not X_parts:
        raise RuntimeError("No data loaded. Check DATASET_ROOT & .mat keys.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    g = np.concatenate(g_parts, axis=0)

    # ★ Drop intermediate lists immediately
    del X_parts, y_parts, g_parts
    gc.collect()

    log(f"Total epochs: {X.shape[0]} | Shape: {X.shape} | dtype: {X.dtype}")
    checkpoint("After full dataset assembled")
    return X, y, g

# =============================================================================
# SECTION 4 — IN-PLACE PREPROCESSING (no duplicate arrays)
# =============================================================================

def preprocess_inplace(X):
    """
    DC removal + 1-40 Hz bandpass applied IN-PLACE on a float32 array.
    ★ No copy created — X is modified directly, halving transient RAM.
    """
    from scipy.signal import butter, sosfiltfilt

    log("  Preprocessing: DC removal + bandpass 1-40 Hz (in-place, float32)...")
    sos = butter(5, [1.0, 40.0], btype="bandpass",
                 fs=CONFIG["sfreq"], output="sos")

    # DC removal in-place (subtract per-epoch per-channel mean)
    X -= X.mean(axis=-1, keepdims=True)

    # Bandpass: process in small channel batches to limit working memory
    # sosfiltfilt needs float64 internally — we cast slice, filter, cast back
    for i in range(X.shape[0]):
        X[i] = sosfiltfilt(sos, X[i].astype(np.float64), axis=-1).astype(np.float32)

    return X   # same array, preprocessed

# =============================================================================
# SECTION 5 — CSP FEATURE EXTRACTION (per fold, no global fit)
# =============================================================================

def extract_csp_features(X_train, y_train, X_test, n_components):
    """
    ★ CSP fitted ONLY on training fold — prevents leakage.
    Returns 2D feature arrays; raw epoch arrays can be deleted after this.
    """
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    csp.fit(X_train, y_train)
    X_tr_feat = csp.transform(X_train)   # (N_train, n_components)
    X_te_feat = csp.transform(X_test)    # (N_test,  n_components)
    return X_tr_feat.astype(np.float32), X_te_feat.astype(np.float32), csp

# =============================================================================
# SECTION 6 — ML MODELS
# =============================================================================

def get_models(random_state):
    return {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale",
                   probability=True, random_state=random_state),
        "RandomForest": RandomForestClassifier(
                   n_estimators=100,          # reduced from 200 → saves RAM
                   max_depth=20,              # cap depth → avoids huge trees
                   random_state=random_state, n_jobs=-1),  # use all CPU cores
        "LogisticRegression": LogisticRegression(
                   C=1.0, max_iter=1000, solver="lbfgs",
                   multi_class="multinomial",
                   random_state=random_state),
    }

# =============================================================================
# SECTION 7 — GROUPKFOLD CV (memory-safe per-fold processing)
# =============================================================================

def run_cv(X, y, groups, config):
    """
    GroupKFold CV with strict per-fold memory management.
    Order of operations per fold:
      1. Slice train/test indices (views, not copies)
      2. Preprocess train/test IN-PLACE
      3. Extract CSP features → immediately del raw epoch slices
      4. Scale features → overwrite in-place
      5. Train & evaluate models
      6. del all fold-level arrays + gc.collect()
    """
    gkf    = GroupKFold(n_splits=config["n_folds"])
    n_comp = config["n_components"]
    rs     = config["random_state"]

    fold_records     = []
    model_histories  = {m: [] for m in get_models(rs)}

    log(f"\n{'='*60}")
    log(f"Starting {config['n_folds']}-Fold GroupKFold CV")
    log(f"{'='*60}")

    # Pre-compute all split indices (tiny lists of integers — negligible RAM)
    splits = list(gkf.split(X, y, groups))

    for fold_idx, (tr_idx, te_idx) in enumerate(splits, 1):
        checkpoint(f"Fold {fold_idx} start")
        log(f"\nFold {fold_idx} | Train={len(tr_idx)} | Test={len(te_idx)} | "
            f"Test subj: {np.unique(groups[te_idx])}")

        # ★ np indexing creates copies here — unavoidable for non-contiguous idx
        X_tr = X[tr_idx].copy()   # float32 (N_tr, C, T)
        X_te = X[te_idx].copy()   # float32 (N_te, C, T)
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        # 1. Preprocess in-place — no extra copy created
        X_tr = preprocess_inplace(X_tr)
        X_te = preprocess_inplace(X_te)
        checkpoint(f"Fold {fold_idx} after preprocess")

        # 2. CSP (per-fold fit on train only)
        try:
            X_tr_feat, X_te_feat, _ = extract_csp_features(
                X_tr, y_tr, X_te, n_comp)
        except Exception as e:
            log(f"  CSP failed fold {fold_idx}: {e}", "ERROR")
            del X_tr, X_te, y_tr, y_te
            gc.collect()
            continue

        # ★ Raw epoch arrays no longer needed — free immediately
        del X_tr, X_te
        gc.collect()
        checkpoint(f"Fold {fold_idx} after CSP (raw epochs freed)")

        # 3. Scale features in-place (overwrite X_tr_feat, X_te_feat)
        scaler = StandardScaler()
        X_tr_feat = scaler.fit_transform(X_tr_feat)
        X_te_feat = scaler.transform(X_te_feat)
        # Cast back to float32 (StandardScaler outputs float64)
        X_tr_feat = X_tr_feat.astype(np.float32, copy=False)
        X_te_feat = X_te_feat.astype(np.float32, copy=False)

        # 4. Train & evaluate
        models = get_models(rs)
        for model_name, clf in models.items():
            t0 = time.time()
            clf.fit(X_tr_feat, y_tr)
            y_pred   = clf.predict(X_te_feat)
            elapsed  = time.time() - t0
            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
            log(f"  {model_name:20s} | Acc={acc:.4f} | F1={f1:.4f} | "
                f"{elapsed:.1f}s")
            fold_records.append({
                "fold": fold_idx, "model": model_name,
                "accuracy": acc, "f1_weighted": f1,
                "train_size": len(tr_idx), "test_size": len(te_idx),
                "test_subjects": str(np.unique(groups[te_idx]).tolist()),
            })
            model_histories[model_name].append({"acc": acc, "f1": f1})
            # ★ Free trained model object after use
            del clf, y_pred
            gc.collect()

        # ★ Free all fold-level arrays before next fold
        del X_tr_feat, X_te_feat, y_tr, y_te, scaler, models
        gc.collect()
        checkpoint(f"Fold {fold_idx} end (all fold arrays freed)")

    return fold_records, model_histories

# =============================================================================
# SECTION 8 — METRICS SAVING
# =============================================================================

def save_metrics(fold_records):
    df = pd.DataFrame(fold_records)
    df.to_csv(OUT / "fold_metrics.csv", index=False)
    log(f"Saved fold_metrics.csv ({len(df)} rows)")

    summary_rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        summary_rows.append({
            "model"   : model,
            "mean_acc": sub["accuracy"].mean(),
            "std_acc" : sub["accuracy"].std(),
            "mean_f1" : sub["f1_weighted"].mean(),
            "std_f1"  : sub["f1_weighted"].std(),
            "n_folds" : len(sub),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    log("Saved summary_metrics.csv")
    return summary

# =============================================================================
# SECTION 9 — PLOTTING
# =============================================================================

def plot_fold_accuracy(fold_records):
    df = pd.DataFrame(fold_records)
    fig, ax = plt.subplots(figsize=(10, 5))
    for mdl in df["model"].unique():
        sub = df[df["model"] == mdl].sort_values("fold")
        ax.plot(sub["fold"], sub["accuracy"], marker="o", label=mdl)
    ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy")
    ax.set_title("Per-Fold Accuracy (CSP + ML)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT / "figures" / "fold_accuracy.png"
    plt.savefig(path, dpi=150); plt.close()
    log(f"Saved {path}")
    return str(path)

def plot_fold_f1(fold_records):
    df = pd.DataFrame(fold_records)
    fig, ax = plt.subplots(figsize=(10, 5))
    for mdl in df["model"].unique():
        sub = df[df["model"] == mdl].sort_values("fold")
        ax.plot(sub["fold"], sub["f1_weighted"], marker="s",
                linestyle="--", label=mdl)
    ax.set_xlabel("Fold"); ax.set_ylabel("F1 (Weighted)")
    ax.set_title("Per-Fold F1 Score (CSP + ML)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT / "figures" / "fold_f1.png"
    plt.savefig(path, dpi=150); plt.close()
    log(f"Saved {path}")
    return str(path)

def plot_summary_bar(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, err, title in zip(
        axes,
        ["mean_acc", "mean_f1"], ["std_acc", "std_f1"],
        ["Mean Accuracy ± Std", "Mean F1 (Weighted) ± Std"],
    ):
        bars = ax.bar(summary["model"], summary[metric],
                      yerr=summary[err], capsize=5,
                      color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_ylim(0, 1.05); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = OUT / "figures" / "summary_bar.png"
    plt.savefig(path, dpi=150); plt.close()
    log(f"Saved {path}")
    return str(path)

# =============================================================================
# SECTION 10 — PDF REPORT
# =============================================================================

def build_pdf_report(fold_records, summary, fig_paths):
    pdf_path = str(OUT / "csp_ml_report.pdf")
    doc    = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    def H(text, style="Heading1"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 0.1*inch))
    def P(text):
        story.append(Paragraph(text, styles["Normal"]))
        story.append(Spacer(1, 0.05*inch))

    H("CSP + ML Pipeline Report — SEED-IV Raw EEG")
    P(f"Generated: {CONFIG['timestamp']}")
    P(f"Dataset: {CONFIG['dataset_root']}")
    P(f"CV: {CONFIG['n_folds']}-Fold GroupKFold | CSP components: {CONFIG['n_components']}")
    story.append(Spacer(1, 0.2*inch))

    H("Summary Metrics", "Heading2")
    tbl_data = [["Model", "Mean Acc", "Std Acc", "Mean F1", "Std F1"]]
    for _, row in summary.iterrows():
        tbl_data.append([row["model"],
                         f"{row['mean_acc']:.4f}", f"{row['std_acc']:.4f}",
                         f"{row['mean_f1']:.4f}",  f"{row['std_f1']:.4f}"])
    tbl = Table(tbl_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF2FF")]),
    ]))
    story.append(tbl); story.append(Spacer(1, 0.3*inch))

    H("Fold-Wise Metrics", "Heading2")
    fold_tbl_data = [["Fold", "Model", "Accuracy", "F1 (Weighted)"]]
    for _, r in pd.DataFrame(fold_records).iterrows():
        fold_tbl_data.append([str(int(r["fold"])), r["model"],
                               f"{r['accuracy']:.4f}", f"{r['f1_weighted']:.4f}"])
    fold_tbl = Table(fold_tbl_data, hAlign="LEFT")
    fold_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DD8452")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FFF6EE")]),
    ]))
    story.append(fold_tbl); story.append(Spacer(1, 0.3*inch))

    H("Figures", "Heading2")
    for fig_path in fig_paths:
        if fig_path and Path(fig_path).exists():
            story.append(RLImage(fig_path, width=5*inch, height=2.8*inch))
            story.append(Spacer(1, 0.2*inch))

    H("Configuration", "Heading2")
    for k, v in CONFIG.items():
        P(f"<b>{k}</b>: {v}")

    doc.build(story)
    log(f"PDF report saved → {pdf_path}")
    return pdf_path

# =============================================================================
# SECTION 11 — MAIN EXECUTION
# =============================================================================

def main():
    log("=" * 60)
    log("CSP + ML Pipeline — SEED-IV Raw EEG (MEMORY-SAFE)")
    log("=" * 60)
    checkpoint("Pipeline start")

    # FALLBACK: warn if RAM is already low before starting
    avail_gb = psutil.virtual_memory().available / 1024**3
    if avail_gb < 4:
        log(f"WARNING: Only {avail_gb:.1f} GB RAM available. "
            "Consider reducing epoch_overlap or n_folds.", "WARN")

    # Step 1: Load data (lazy session-by-session, then combined for CV)
    log("\n[STEP 1] Loading raw EEG data (session-by-session)...")
    X, y, groups = load_all_subjects(CONFIG["dataset_root"], CONFIG)
    log(f"X: {X.shape} dtype={X.dtype} | y: {y.shape} | "
        f"Subjects: {np.unique(groups)}")
    checkpoint("After data load")

    # Step 2: Cross-validation
    log("\n[STEP 2] Running GroupKFold CV...")
    fold_records, model_histories = run_cv(X, y, groups, CONFIG)

    # ★ Free full dataset after CV — no longer needed
    del X, y, groups
    gc.collect()
    checkpoint("After CV (full dataset freed)")

    # Step 3: Metrics
    log("\n[STEP 3] Saving metrics...")
    summary = save_metrics(fold_records)
    print("\nSummary:\n", summary.to_string(index=False))

    # Step 4: Plots
    log("\n[STEP 4] Generating plots...")
    fp1 = plot_fold_accuracy(fold_records)
    fp2 = plot_fold_f1(fold_records)
    fp3 = plot_summary_bar(summary)

    # Step 5: PDF
    log("\n[STEP 5] Building PDF report...")
    build_pdf_report(fold_records, summary, [fp1, fp2, fp3])

    # Step 6: Save log
    save_log()
    checkpoint("Pipeline end")
    log(f"\n✅ Done. Results saved to: {OUT}")


if __name__ == "__main__":
    main()
