"""
=============================================================================
SCRIPT 2: RAW EEG + EYE DATA → DEEP LEARNING (MULTIMODAL) — SEED-IV
=============================================================================

HOW TO USE LOCALLY:
----------------------------
1. Install dependencies (once):
   pip install scipy numpy pandas matplotlib reportlab scikit-learn tensorflow

2. Verify folder structure:
   <project>/dataset/
       eeg_raw_data/
           1/  2/  3/      ← session folders with subject .mat files
       eye_raw_data/       ← eye .mat files (pupil, fixation, saccade, blink, PD)

3. Set EEG_ROOT and EYE_ROOT in SECTION 1 if your paths differ.

4. Run: python obj2_dl_pipeline.py

OUTPUTS (in obj2/):
   fold_metrics_dl.csv         per-fold accuracy + F1
   summary_metrics_dl.csv      mean ± std
   config_dl.json              saved configuration
   dl_report.pdf               PDF report with plots
   logs/
     dl_run_<ts>_epochs.csv    per-epoch loss/accuracy (CSV)
     dl_run_<ts>_epochs.json   per-epoch loss/accuracy (JSON)
     dl_run_<ts>.log           full console tee log
     dl_run_console.csv        console message log
     dl_run_<ts>_report.md     Markdown experiment summary
=============================================================================
"""

# =============================================================================
# SECTION 0 — IMPORTS
# =============================================================================
# Run once if not installed:
#   pip install scipy numpy pandas matplotlib reportlab scikit-learn tensorflow
import sys

import os, gc, json, time, warnings
from datetime import datetime

# Structured experiment logger (logger.py must be in the same directory)
from logger import ExperimentLogger
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# --- LOCAL PATHS — edit these two lines to match your machine ---
EEG_ROOT   = r"C:\Users\Rose J Thachil\Desktop\8th_sem_project\raw_data_project\dataset\eeg_raw_data"
EYE_ROOT   = r"C:\Users\Rose J Thachil\Desktop\8th_sem_project\raw_data_project\dataset\eye_raw_data"
OUTPUT_DIR = r"C:\Users\Rose J Thachil\Desktop\8th_sem_project\raw_data_project\obj2"
# ---------------------------------------------------------------

CONFIG = {
    "eeg_root"        : EEG_ROOT,
    "eye_root"        : EYE_ROOT,
    "output_dir"      : OUTPUT_DIR,
    "sfreq"           : 200,
    "epoch_len_sec"   : 4,
    "epoch_overlap"   : 0.5,
    "n_eeg_channels"  : 62,
    "eye_feature_dim" : 5,   # pupil_L, pupil_R, fix_x, fix_y, blink
    "n_classes"       : 4,
    "n_folds"         : 5,
    "epochs"          : 30,
    "batch_size"      : 32,
    "learning_rate"   : 1e-3,
    "random_state"    : 42,
    "timestamp"       : datetime.now().strftime("%Y%m%d_%H%M%S"),
}

SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

OUT = Path(CONFIG["output_dir"])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)

with open(OUT / "config_dl.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

tf.random.set_seed(CONFIG["random_state"])
np.random.seed(CONFIG["random_state"])

# =============================================================================
# SECTION 2 — LOGGING
# =============================================================================

# Instantiate the structured logger (shared across all functions via module-level)
# overwrite=False  → logs are appended so multiple runs are preserved
EXP_LOGGER: ExperimentLogger = None  # initialised in main() after OUT is ready

def log(msg: str, level: str = "INFO"):
    """Thin wrapper — delegates to ExperimentLogger if ready, else plain print."""
    if EXP_LOGGER is not None:
        EXP_LOGGER.log(msg, level)
    else:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [{level}] {msg}")

def save_log():
    """Persist console log CSV via the logger."""
    if EXP_LOGGER is not None:
        EXP_LOGGER.save_console_log()

# =============================================================================
# SECTION 3 — EEG DATA LOADING
# =============================================================================

def load_eeg_mat(filepath):
    mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    trial_keys = sorted([k for k in mat if not k.startswith("__")])
    trials = []
    for key in trial_keys:
        data = mat[key]
        if isinstance(data, np.ndarray) and data.ndim == 2:
            if data.shape[0] > data.shape[1]:
                data = data.T
            # ★ Cast to float32 immediately — halves RAM vs float64
            trials.append(data.astype(np.float32, copy=False))
            del data
    del mat
    gc.collect()
    return trials


def sliding_epochs(trial, sfreq, epoch_len_sec, overlap):
    n_ch, n_time = trial.shape
    ep = int(sfreq * epoch_len_sec)
    step = int(ep * (1 - overlap))
    out = []
    s = 0
    while s + ep <= n_time:
        out.append(trial[:, s:s+ep])
        s += step
    return np.array(out) if out else None

# =============================================================================
# SECTION 4 — EYE DATA LOADING
# =============================================================================

def load_eye_mat(filepath, signal_type):
    """
    Load an eye-tracking .mat file and return a flat feature vector per trial.
    signal_type: 'pupil' | 'fixation' | 'saccade' | 'blink' | 'PD'
    """
    try:
        mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat if not k.startswith("__")]
        vals = []
        for k in sorted(keys):
            d = mat[k]
            if isinstance(d, np.ndarray):
                vals.append(float(np.nanmean(d.ravel())))
            elif np.isscalar(d):
                vals.append(float(d))
        return np.array(vals, dtype=np.float32) if vals else np.zeros(1, dtype=np.float32)
    except Exception:
        return np.zeros(1, dtype=np.float32)


def get_eye_feature_for_session(eye_root, subj_id, session_date, n_trials):
    """
    Aggregate per-trial eye features by loading pupil, fixation, blink .mat files.
    Returns array (n_trials, n_eye_feats).
    """
    eye_dir = Path(eye_root)
    prefix  = f"{subj_id}_{session_date}"

    pupil_path = eye_dir / f"{prefix}_pupil.mat"
    fix_path   = eye_dir / f"{prefix}_fixation.mat"
    blink_path = eye_dir / f"{prefix}_blink.mat"

    def load_vec(p, sig):
        if p.exists():
            return load_eye_mat(str(p), sig)
        return np.zeros(1, dtype=np.float32)

    pupil_feats = load_vec(pupil_path, "pupil")
    fix_feats   = load_vec(fix_path,   "fixation")
    blink_feats = load_vec(blink_path, "blink")

    # Build a single feature vector and replicate for each trial
    combined = np.concatenate([pupil_feats[:3], fix_feats[:1], blink_feats[:1]])
    combined = combined[:CONFIG["eye_feature_dim"]]
    if len(combined) < CONFIG["eye_feature_dim"]:
        combined = np.pad(combined,
                          (0, CONFIG["eye_feature_dim"] - len(combined)))
    trial_eye = np.tile(combined, (n_trials, 1))   # (n_trials, eye_feat_dim)
    return trial_eye.astype(np.float32)

# =============================================================================
# SECTION 5 — LOAD ALL SUBJECTS (EEG + EYE)
# =============================================================================

def load_all_subjects(config):
    eeg_root = Path(config["eeg_root"])
    eye_root = Path(config["eye_root"])
    sfreq    = config["sfreq"]
    ep_len   = config["epoch_len_sec"]
    overlap  = config["epoch_overlap"]

    eeg_list, eye_list, y_list, g_list = [], [], [], []

    for sess_id in [1, 2, 3]:
        sess_dir = eeg_root / str(sess_id)
        if not sess_dir.exists():
            log(f"Session dir missing: {sess_dir}", "WARN")
            continue
        labels   = SESSION_LABELS[sess_id]
        mat_files = sorted(sess_dir.glob("*.mat"))
        log(f"Session {sess_id}: {len(mat_files)} subjects")

        for mat_path in mat_files:
            stem     = mat_path.stem                     # e.g. "1_20160518"
            subj_id  = int(stem.split("_")[0])
            sess_date = stem.split("_")[1]               # e.g. "20160518"

            log(f"  Subject {subj_id} | {mat_path.name}")
            try:
                trials = load_eeg_mat(str(mat_path))
            except Exception as e:
                log(f"  EEG load error: {e}", "ERROR"); continue

            n_trials = min(len(trials), len(labels))
            # Eye features per trial
            trial_eye = get_eye_feature_for_session(
                eye_root, subj_id, sess_date, n_trials)

            for t_idx in range(n_trials):
                trial = trials[t_idx]
                if trial.ndim != 2 or trial.shape[0] < 2:
                    continue
                epochs = sliding_epochs(trial, sfreq, ep_len, overlap)
                if epochs is None or len(epochs) == 0:
                    continue
                label     = labels[t_idx]
                eye_feat  = trial_eye[t_idx]             # (eye_feat_dim,)

                eeg_list.append(epochs)
                # Replicate eye feature for every epoch of this trial
                eye_list.extend([eye_feat] * len(epochs))
                y_list.extend([label] * len(epochs))
                g_list.extend([subj_id] * len(epochs))

    if not eeg_list:
        raise RuntimeError("No data loaded. Check paths & .mat structure.")

    X_eeg  = np.concatenate(eeg_list, axis=0)          # (N, n_ch, time) float32
    del eeg_list
    gc.collect()

    X_eye  = np.vstack(eye_list).astype(np.float32)    # (N, eye_feat_dim)
    del eye_list
    gc.collect()

    y      = np.array(y_list,  dtype=np.int32)
    groups = np.array(g_list,  dtype=np.int32)
    del y_list, g_list
    gc.collect()

    log(f"EEG: {X_eeg.shape} dtype={X_eeg.dtype} | Eye: {X_eye.shape} | Labels: {y.shape}")
    return X_eeg, X_eye, y, groups

# =============================================================================
# SECTION 6 — MULTIMODAL MODEL
# =============================================================================

def build_model(eeg_time_steps, eeg_channels, eye_feat_dim, n_classes, lr):
    """
    EEG branch : 1D CNN along time axis
    Eye branch : Dense layers on raw eye features
    Fusion     : Concatenation + Dense
    """
    # — EEG Branch (1D CNN) —
    eeg_input = layers.Input(shape=(eeg_time_steps, eeg_channels),
                              name="eeg_input")
    x = layers.Conv1D(64, kernel_size=7, activation="relu", padding="same")(eeg_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    eeg_out = layers.Dense(64, activation="relu", name="eeg_embedding")(x)

    # — Eye Branch —
    eye_input = layers.Input(shape=(eye_feat_dim,), name="eye_input")
    e = layers.Dense(64, activation="relu")(eye_input)
    e = layers.BatchNormalization()(e)
    e = layers.Dropout(0.3)(e)
    eye_out = layers.Dense(32, activation="relu", name="eye_embedding")(e)

    # — Fusion —
    fused = layers.Concatenate(name="fusion")([eeg_out, eye_out])
    f = layers.Dense(128, activation="relu")(fused)
    f = layers.Dropout(0.4)(f)
    f = layers.Dense(64, activation="relu")(f)
    output = layers.Dense(n_classes, activation="softmax", name="output")(f)

    model = models.Model(inputs=[eeg_input, eye_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# =============================================================================
# SECTION 7 — GROUPKFOLD EVALUATION
# =============================================================================

def run_cv(X_eeg, X_eye, y, groups, config):
    gkf    = GroupKFold(n_splits=config["n_folds"])
    n_ch   = X_eeg.shape[1]   # 62  — X_eeg is (N, ch, time)
    t_step = X_eeg.shape[2]   # 800 — after per-fold transpose → (N, time, ch)
    e_dim  = X_eye.shape[1]

    fold_records   = []
    history_list   = []
    all_val_acc    = []
    all_val_loss   = []

    log(f"\n{'='*60}")
    log(f"Starting {config['n_folds']}-Fold GroupKFold DL CV")
    log(f"{'='*60}")

    for fold_idx, (tr_idx, te_idx) in enumerate(
            gkf.split(X_eeg, y, groups), 1):

        log(f"\nFold {fold_idx} | Train={len(tr_idx)} Test={len(te_idx)} "
            f"| Test subj={np.unique(groups[te_idx])}")

        # Index separately so intermediate copies can be freed
        X_eeg_tr = np.ascontiguousarray(X_eeg[tr_idx].transpose(0, 2, 1))  # (N, time, ch)
        X_eeg_te = np.ascontiguousarray(X_eeg[te_idx].transpose(0, 2, 1))
        X_eye_tr, X_eye_te = X_eye[tr_idx], X_eye[te_idx]
        y_tr, y_te         = y[tr_idx],     y[te_idx]

        # Per-fold EEG normalization — IN-PLACE to avoid ~11 GiB temporary array
        eeg_mean = X_eeg_tr.mean(axis=(0, 1), keepdims=True)
        eeg_std  = X_eeg_tr.std(axis=(0, 1), keepdims=True) + 1e-8
        np.subtract(X_eeg_tr, eeg_mean, out=X_eeg_tr)
        np.divide(X_eeg_tr,   eeg_std,  out=X_eeg_tr)
        np.subtract(X_eeg_te, eeg_mean, out=X_eeg_te)
        np.divide(X_eeg_te,   eeg_std,  out=X_eeg_te)
        del eeg_mean, eeg_std
        gc.collect()

        # Per-fold eye normalization
        eye_scaler = StandardScaler()
        X_eye_tr = eye_scaler.fit_transform(X_eye_tr)
        X_eye_te = eye_scaler.transform(X_eye_te)

        # Build & train model
        model = build_model(t_step, n_ch, e_dim,
                            config["n_classes"], config["learning_rate"])
        if fold_idx == 1:
            model.summary()

        cb_list = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                    monitor="val_accuracy"),
            callbacks.ReduceLROnPlateau(patience=3, factor=0.5,
                                        monitor="val_accuracy"),
        ]

        hist = model.fit(
            [X_eeg_tr, X_eye_tr], y_tr,
            validation_data=([X_eeg_te, X_eye_te], y_te),
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=cb_list,
            verbose=1,
        )
        history_list.append(hist.history)
        all_val_acc.append(hist.history.get("val_accuracy", []))
        all_val_loss.append(hist.history.get("val_loss", []))

        # ── Structured epoch logging (no training logic changed) ──────────
        if EXP_LOGGER is not None:
            EXP_LOGGER.record_all_epochs(fold=fold_idx, history_dict=hist.history)
        # ─────────────────────────────────────────────────────────────────

        y_pred_prob = model.predict([X_eeg_te, X_eye_te], verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        log(f"  Fold {fold_idx} → Acc={acc:.4f} | F1={f1:.4f}")

        fold_records.append({
            "fold": fold_idx, "accuracy": acc,
            "f1_weighted": f1,
            "train_size": len(tr_idx), "test_size": len(te_idx),
            "test_subjects": str(np.unique(groups[te_idx]).tolist()),
        })

        # Free fold data before next iteration
        del X_eeg_tr, X_eeg_te, X_eye_tr, X_eye_te, y_tr, y_te
        gc.collect()
        tf.keras.backend.clear_session()

    return fold_records, all_val_acc, all_val_loss

# =============================================================================
# SECTION 8 — SAVE METRICS
# =============================================================================

def save_metrics(fold_records):
    df = pd.DataFrame(fold_records)
    df.to_csv(OUT / "fold_metrics_dl.csv", index=False)
    log(f"Saved fold_metrics_dl.csv ({len(df)} rows)")

    summary = pd.DataFrame([{
        "mean_acc" : df["accuracy"].mean(),
        "std_acc"  : df["accuracy"].std(),
        "mean_f1"  : df["f1_weighted"].mean(),
        "std_f1"   : df["f1_weighted"].std(),
        "n_folds"  : len(df),
    }])
    summary.to_csv(OUT / "summary_metrics_dl.csv", index=False)
    log("Saved summary_metrics_dl.csv")
    return df, summary

# =============================================================================
# SECTION 9 — PLOTTING
# =============================================================================

def plot_learning_curves(all_val_acc, all_val_loss):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(all_val_acc)))

    for i, (acc_hist, loss_hist) in enumerate(zip(all_val_acc, all_val_loss)):
        axes[0].plot(acc_hist,  color=colors_list[i], label=f"Fold {i+1}")
        axes[1].plot(loss_hist, color=colors_list[i], label=f"Fold {i+1}")

    axes[0].set_title("Validation Accuracy per Fold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss per Fold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = OUT / "figures" / "learning_curves.png"
    plt.savefig(path, dpi=150); plt.close()
    log(f"Saved {path}")
    return str(path)


def plot_fold_bar(fold_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    folds = fold_df["fold"].values

    for ax, col, title in zip(
        axes,
        ["accuracy", "f1_weighted"],
        ["Per-Fold Accuracy", "Per-Fold F1 (Weighted)"]
    ):
        bars = ax.bar(folds, fold_df[col].values, color="#4C72B0", alpha=0.8)
        ax.axhline(fold_df[col].mean(), color="red", linestyle="--",
                   label=f"Mean={fold_df[col].mean():.3f}")
        ax.set_xticks(folds)
        ax.set_xlabel("Fold"); ax.set_ylabel(col)
        ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, fold_df[col].values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = OUT / "figures" / "fold_bars.png"
    plt.savefig(path, dpi=150); plt.close()
    log(f"Saved {path}")
    return str(path)

# =============================================================================
# SECTION 10 — PDF REPORT
# =============================================================================

def build_pdf(fold_df, summary_df, fig_paths):
    pdf_path = str(OUT / "dl_report.pdf")
    doc    = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    def H(t, s="Heading1"):
        story.append(Paragraph(t, styles[s])); story.append(Spacer(1, 0.1*inch))

    def P(t):
        story.append(Paragraph(t, styles["Normal"])); story.append(Spacer(1, 0.05*inch))

    H("Deep Learning Multimodal Report — SEED-IV")
    P(f"Generated: {CONFIG['timestamp']}")
    P(f"EEG Root: {CONFIG['eeg_root']}")
    P(f"Eye Root: {CONFIG['eye_root']}")
    P(f"Cross-Validation: {CONFIG['n_folds']}-Fold GroupKFold (subject-based)")
    P("Model: EEG 1D-CNN + Eye Dense → Fusion → Softmax (4 classes)")
    story.append(Spacer(1, 0.2*inch))

    H("Summary Metrics", "Heading2")
    s = summary_df.iloc[0]
    tbl = Table([
        ["Metric", "Value"],
        ["Mean Accuracy", f"{s['mean_acc']:.4f}"],
        ["Std Accuracy",  f"{s['std_acc']:.4f}"],
        ["Mean F1",       f"{s['mean_f1']:.4f}"],
        ["Std F1",        f"{s['std_f1']:.4f}"],
        ["Folds",         str(int(s['n_folds']))],
    ], hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, colors.HexColor("#EEF2FF")]),
    ]))
    story.append(tbl); story.append(Spacer(1, 0.3*inch))

    H("Per-Fold Metrics", "Heading2")
    rows = [["Fold", "Accuracy", "F1 (Weighted)", "Test Subjects"]]
    for _, r in fold_df.iterrows():
        rows.append([str(int(r["fold"])), f"{r['accuracy']:.4f}",
                     f"{r['f1_weighted']:.4f}", r["test_subjects"]])
    tbl2 = Table(rows, hAlign="LEFT")
    tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#DD8452")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, colors.HexColor("#FFF6EE")]),
    ]))
    story.append(tbl2); story.append(Spacer(1, 0.3*inch))

    H("Model Architecture", "Heading2")
    P("EEG Branch: Input(time, 62 channels) → Conv1D(64,k7) → BN → Pool → "
      "Conv1D(128,k5) → BN → Pool → Conv1D(256,k3) → GlobalAvgPool → "
      "Dense(128) → Dropout(0.4) → Dense(64)")
    P("Eye Branch: Input(5 features) → Dense(64) → BN → Dropout(0.3) → Dense(32)")
    P("Fusion: Concatenate → Dense(128) → Dropout(0.4) → Dense(64) → Softmax(4)")
    story.append(Spacer(1, 0.2*inch))

    H("Figures", "Heading2")
    for fp in fig_paths:
        if fp and Path(fp).exists():
            story.append(RLImage(fp, width=5*inch, height=2.8*inch))
            story.append(Spacer(1, 0.2*inch))

    H("Configuration", "Heading2")
    for k, v in CONFIG.items():
        P(f"<b>{k}</b>: {v}")

    doc.build(story)
    log(f"PDF saved → {pdf_path}")
    return pdf_path

# =============================================================================
# SECTION 11 — MAIN
# =============================================================================

def main():
    global EXP_LOGGER  # allow the module-level variable to be set here

    # ── Initialise structured logger FIRST so tee captures everything ──
    EXP_LOGGER = ExperimentLogger(
        output_dir=OUT,
        experiment_name="dl_run",
        config=CONFIG,
        overwrite=False,   # set True to discard previous logs for this run
    )
    EXP_LOGGER.start_tee()   # all print() / log() now mirrors to .log file
    # ──────────────────────────────────────────────────────────────────

    log("=" * 60)
    log("DL Multimodal Pipeline — Raw EEG + Eye (SEED-IV)")
    log("=" * 60)

    log("\n[STEP 1] Loading raw EEG + Eye data...")
    X_eeg, X_eye, y, groups = load_all_subjects(CONFIG)
    log(f"EEG: {X_eeg.shape} | Eye: {X_eye.shape} | y: {y.shape} "
        f"| Subjects: {np.unique(groups)}")

    log("\n[STEP 2] Running GroupKFold cross-validation...")
    fold_records, all_val_acc, all_val_loss = run_cv(
        X_eeg, X_eye, y, groups, CONFIG)

    log("\n[STEP 3] Saving metrics...")
    fold_df, summary_df = save_metrics(fold_records)
    print("\nSummary:\n", summary_df.to_string(index=False))

    log("\n[STEP 4] Generating plots...")
    fp1 = plot_learning_curves(all_val_acc, all_val_loss)
    fp2 = plot_fold_bar(fold_df)

    log("\n[STEP 5] Building PDF report...")
    build_pdf(fold_df, summary_df, [fp1, fp2])

    # ── Structured logging finalization ───────────────────────────────
    log("\n[STEP 6] Saving structured logs + Markdown report...")
    EXP_LOGGER.save_epoch_logs()                          # CSV + JSON
    EXP_LOGGER.generate_md_report(fold_df, summary_df)   # .md report
    save_log()                                            # console CSV
    # ──────────────────────────────────────────────────────────────────

    log(f"\n✅ DL Pipeline complete. Results saved to: {OUT}")
    EXP_LOGGER.close()   # flush + close log file, restore stdout


if __name__ == "__main__":
    main()
