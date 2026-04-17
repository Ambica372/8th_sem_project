# -*- coding: utf-8 -*-
"""
=============================================================================
Stage 3: Feature Analysis
Project : Physiological Emotion Recognition (SEED-IV)
=============================================================================
Uses ONLY training data from the 5 cross-validation folds.

Sections:
  1. Statistical discrimination  -- ANOVA + Kruskal-Wallis per feature
  2. Correlation / redundancy    -- correlation matrix, |r|>0.9 pairs
  3. Dimensionality reduction    -- PCA (2D+3D), t-SNE (2D), UMAP (2D)
  4. EEG band/channel aggregation -- reshape 310-D -> (62 ch x 5 bands)
  5. Eye feature interpretation  -- distributions per emotion

EEG feature ordering (from Stage 1):
  Original MATLAB shape: (62, W, 5)
  After transpose+reshape: (W, 310)
  => feature_idx = channel_idx * 5 + band_idx
  => channel = feat_idx // 5,  band = feat_idx % 5
  Bands: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma
=============================================================================
"""

import os, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy       import stats
from scipy.stats import f_oneway, kruskal
from sklearn.decomposition    import PCA
from sklearn.manifold         import TSNE
from sklearn.preprocessing    import LabelEncoder
import umap

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STAGE2_DIR  = r"c:\Users\Rose J Thachil\Documents\8th sem\stage2_output"
OUTPUT_DIR  = r"c:\Users\Rose J Thachil\Documents\8th sem\stage3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FOLDS       = 5
META_COLS     = ["subject_id", "session_id", "trial_id", "window_id", "emotion_label"]
N_EEG         = 310
N_EYE         = 31
N_CHANNELS    = 62
N_BANDS       = 5
BAND_NAMES    = ["delta", "theta", "alpha", "beta", "gamma"]
EMOTION_NAMES = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}
EMOTION_COLORS= {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}
TSNE_SAMPLE   = 4000   # subsample for t-SNE/UMAP (speed)
HIGH_CORR_THR = 0.90

SEP  = "=" * 65
SEP2 = "-" * 50

# ---------------------------------------------------------------------------
# HELPER: load and stack training data across all folds
# ---------------------------------------------------------------------------
def load_train_all_folds(modality):
    """
    Load and concatenate fold_k_train_{modality}.csv for k=1..5.
    Returns combined DataFrame.
    NOTE: A window appears in exactly 4 train folds out of 5 (once in each
    fold it is NOT the test set). We deduplicate by (subject_id, trial_id,
    window_id) so each window appears ONCE in the combined set.
    """
    dfs = []
    for k in range(1, N_FOLDS + 1):
        path = os.path.join(STAGE2_DIR,
               "fold_" + str(k) + "_train_" + modality + ".csv")
        df = pd.read_csv(path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate: keep first occurrence of each (subject, trial, window)
    combined = combined.drop_duplicates(
        subset=["subject_id", "trial_id", "window_id"]
    ).reset_index(drop=True)
    return combined


def feat_cols(df, n_expected=None):
    cols = [c for c in df.columns if c not in META_COLS]
    if n_expected:
        assert len(cols) == n_expected, (
            "Expected " + str(n_expected) + " feat cols, got " + str(len(cols))
        )
    return cols


# ============================================================================
# SECTION 1: Statistical Discrimination
# ============================================================================
def statistical_tests(df, modality, n_feats):
    """
    Per-fold ANOVA and Kruskal-Wallis for every feature.
    Returns summary DataFrame: feature, anova_pval, kw_pval,
    anova_sig (bool), kw_sig (bool), sig_folds (count).
    """
    print("\n" + SEP)
    print("SECTION 1 -- Statistical Discrimination [" + modality.upper() + "]")
    print(SEP)

    fcols = feat_cols(df, n_feats)
    labels = df["emotion_label"].values
    groups = [0, 1, 2, 3]

    fold_anova_pvals = []
    fold_kw_pvals    = []

    for k in range(1, N_FOLDS + 1):
        path = os.path.join(STAGE2_DIR,
               "fold_" + str(k) + "_train_" + modality + ".csv")
        fdf  = pd.read_csv(path)
        flabels = fdf["emotion_label"].values
        X = fdf[fcols].values.astype(np.float64)

        anova_pvals = np.ones(n_feats)
        kw_pvals    = np.ones(n_feats)

        for i in range(n_feats):
            col = X[:, i]
            # Group samples by emotion label; drop NaN for eye data
            grps = [col[(flabels == g) & ~np.isnan(col)] for g in groups]
            # Skip if any group has <2 valid samples
            if any(len(g) < 2 for g in grps):
                continue
            try:
                _, anova_pvals[i] = f_oneway(*grps)
            except Exception:
                pass
            try:
                _, kw_pvals[i] = kruskal(*grps)
            except Exception:
                pass

        fold_anova_pvals.append(anova_pvals)
        fold_kw_pvals.append(kw_pvals)

        n_sig_anova = int((anova_pvals < 0.05).sum())
        n_sig_kw    = int((kw_pvals    < 0.05).sum())
        print("  Fold " + str(k)
              + " | ANOVA sig: " + str(n_sig_anova) + "/" + str(n_feats)
              + " | KW sig: "    + str(n_sig_kw)    + "/" + str(n_feats))

    fold_anova = np.stack(fold_anova_pvals)  # (N_FOLDS, n_feats)
    fold_kw    = np.stack(fold_kw_pvals)

    avg_anova = fold_anova.mean(axis=0)
    avg_kw    = fold_kw.mean(axis=0)
    sig_count_anova = (fold_anova < 0.05).sum(axis=0)  # how many folds significant
    sig_count_kw    = (fold_kw    < 0.05).sum(axis=0)

    result_df = pd.DataFrame({
        "feature":        fcols,
        "avg_anova_pval": avg_anova,
        "avg_kw_pval":    avg_kw,
        "anova_sig_all":  avg_anova < 0.05,
        "kw_sig_all":     avg_kw    < 0.05,
        "sig_folds_anova":sig_count_anova,
        "sig_folds_kw":   sig_count_kw,
    })
    result_df = result_df.sort_values("avg_anova_pval")

    # Summary
    n_sig_a = int((result_df["anova_sig_all"]).sum())
    n_sig_k = int((result_df["kw_sig_all"]).sum())
    n_both  = int((result_df["anova_sig_all"] & result_df["kw_sig_all"]).sum())
    print("\n  Summary (average across 5 folds):")
    print("    ANOVA significant (p<0.05)          : " + str(n_sig_a) + "/" + str(n_feats))
    print("    Kruskal-Wallis significant (p<0.05) : " + str(n_sig_k) + "/" + str(n_feats))
    print("    Both tests significant               : " + str(n_both)  + "/" + str(n_feats))

    top10 = result_df.head(10)
    print("\n  Top 10 most discriminative " + modality.upper() + " features (by avg ANOVA p-value):")
    for _, row in top10.iterrows():
        print("    " + row["feature"]
              + "  anova_p=" + str(round(row["avg_anova_pval"], 4))
              + "  kw_p="    + str(round(row["avg_kw_pval"],    4))
              + "  sig_folds=" + str(int(row["sig_folds_anova"])) + "/5")

    return result_df


# ============================================================================
# SECTION 2: Correlation and Redundancy
# ============================================================================
def correlation_analysis(df, modality, n_feats):
    """Compute correlation matrix from combined training data. Identify |r|>0.9 pairs."""
    print("\n" + SEP)
    print("SECTION 2 -- Correlation Analysis [" + modality.upper() + "]")
    print(SEP)

    fcols = feat_cols(df, n_feats)
    X = df[fcols]

    # Use pairwise-complete obs for eye NaN handling
    corr = X.corr(method="pearson")

    # Extract highly correlated pairs (upper triangle only)
    high_corr_pairs = []
    arr = corr.values
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            if abs(arr[i, j]) > HIGH_CORR_THR:
                high_corr_pairs.append({
                    "feat_a": fcols[i],
                    "feat_b": fcols[j],
                    "r":      round(arr[i, j], 4),
                })

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("r", ascending=False)
    print("  Feature pairs with |r| > " + str(HIGH_CORR_THR) + ": "
          + str(len(high_corr_pairs)))
    if len(high_corr_pairs) > 0:
        print("  Top 10 most correlated pairs:")
        for _, row in high_corr_df.head(10).iterrows():
            print("    " + row["feat_a"] + " <-> " + row["feat_b"]
                  + "  r=" + str(row["r"]))

    return corr, high_corr_df


# ============================================================================
# SECTION 3: Dimensionality Reduction
# ============================================================================
def dimensionality_reduction(df, modality, n_feats, sig_df=None):
    """PCA (2D+3D), t-SNE (2D), UMAP (2D) on training features."""
    print("\n" + SEP)
    print("SECTION 3 -- Dimensionality Reduction [" + modality.upper() + "]")
    print(SEP)

    fcols = feat_cols(df, n_feats)

    # --- Prepare data: drop NaN rows (for eye data compatibility) -----------
    X_full = df[fcols].values.astype(np.float64)
    y_full = df["emotion_label"].values.astype(int)
    s_full = df["subject_id"].values.astype(int)

    # Drop rows with any NaN (eye data only)
    valid = ~np.isnan(X_full).any(axis=1)
    X = X_full[valid]
    y = y_full[valid]
    s = s_full[valid]
    print("  Valid samples (no NaN): " + str(len(X)) + " / " + str(len(X_full)))

    # Subsample for t-SNE and UMAP
    if len(X) > TSNE_SAMPLE:
        idx = np.random.choice(len(X), TSNE_SAMPLE, replace=False)
        X_sub = X[idx];  y_sub = y[idx];  s_sub = s[idx]
    else:
        X_sub = X;  y_sub = y;  s_sub = s

    # ---- PCA 2D --------------------------------------------------------
    pca2 = PCA(n_components=2, random_state=42)
    emb2 = pca2.fit_transform(X)
    var2 = pca2.explained_variance_ratio_
    print("  PCA 2D | PC1=" + str(round(var2[0]*100,1))
          + "% | PC2=" + str(round(var2[1]*100,1)) + "% variance explained")

    _scatter_2d(emb2, y, s,
        title=modality.upper() + " PCA 2D (PC1=" + str(round(var2[0]*100,1))
              + "%, PC2=" + str(round(var2[1]*100,1)) + "%)",
        xlabel="PC1", ylabel="PC2",
        fname="pca2d_" + modality + ".png")

    # ---- PCA 3D --------------------------------------------------------
    pca3 = PCA(n_components=3, random_state=42)
    emb3 = pca3.fit_transform(X)
    var3 = pca3.explained_variance_ratio_
    print("  PCA 3D | PC1=" + str(round(var3[0]*100,1))
          + "% | PC2=" + str(round(var3[1]*100,1))
          + "% | PC3=" + str(round(var3[2]*100,1)) + "%")

    _scatter_3d(emb3, y,
        title=modality.upper() + " PCA 3D",
        fname="pca3d_" + modality + ".png")

    # ---- t-SNE ---------------------------------------------------------
    print("  Running t-SNE on " + str(len(X_sub)) + " samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40,
                n_iter=1000, learning_rate=200)
    emb_tsne = tsne.fit_transform(X_sub)
    print("  t-SNE done.")

    _scatter_2d(emb_tsne, y_sub, s_sub,
        title=modality.upper() + " t-SNE 2D",
        xlabel="tSNE-1", ylabel="tSNE-2",
        fname="tsne2d_" + modality + ".png")

    # ---- UMAP ----------------------------------------------------------
    print("  Running UMAP on " + str(len(X_sub)) + " samples...")
    reducer = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=30, min_dist=0.1)
    emb_umap = reducer.fit_transform(X_sub)
    print("  UMAP done.")

    _scatter_2d(emb_umap, y_sub, s_sub,
        title=modality.upper() + " UMAP 2D",
        xlabel="UMAP-1", ylabel="UMAP-2",
        fname="umap2d_" + modality + ".png")

    # ---- PCA variance explained cumulative curve -----------------------
    pca_full = PCA(random_state=42)
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    print("  Components to explain 95% variance: " + str(n95))
    _plot_cumvar(cumvar, n95, modality)

    return {"pca2_var": var2.tolist(), "pca3_var": var3.tolist(),
            "n_components_95pct": n95}


def _scatter_2d(emb, labels, subjects, title, xlabel, ylabel, fname):
    """Two-panel scatter: left=color by emotion, right=color by subject."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Left: emotion label
    for lbl, name in EMOTION_NAMES.items():
        mask = labels == lbl
        axes[0].scatter(emb[mask, 0], emb[mask, 1],
                        c=EMOTION_COLORS[lbl], label=name,
                        alpha=0.45, s=10, linewidths=0)
    axes[0].set_title("Colored by Emotion")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
    axes[0].legend(markerscale=3, fontsize=8)

    # Right: subject id
    subj_ids = sorted(np.unique(subjects))
    cmap = plt.cm.get_cmap("tab20", len(subj_ids))
    for i, sid in enumerate(subj_ids):
        mask = subjects == sid
        axes[1].scatter(emb[mask, 0], emb[mask, 1],
                        color=cmap(i), label="S" + str(sid),
                        alpha=0.4, s=10, linewidths=0)
    axes[1].set_title("Colored by Subject")
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel(ylabel)
    axes[1].legend(markerscale=3, fontsize=6, ncol=2)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, fname)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: " + fname)


def _scatter_3d(emb, labels, title, fname):
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    for lbl, name in EMOTION_NAMES.items():
        mask = labels == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                   c=EMOTION_COLORS[lbl], label=name,
                   alpha=0.4, s=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(markerscale=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: " + fname)


def _plot_cumvar(cumvar, n95, modality):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar)+1), cumvar * 100, color="#2176AE", lw=2)
    ax.axhline(95, color="#E84855", ls="--", lw=1.4, label="95% threshold")
    ax.axvline(n95, color="#E84855", ls=":", lw=1.4)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title(modality.upper() + " PCA Scree — " + str(n95)
                 + " components explain 95%", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pca_scree_" + modality + ".png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: pca_scree_" + modality + ".png")


# ============================================================================
# SECTION 4: EEG Band/Channel Aggregation
# ============================================================================
def eeg_band_channel_analysis(df):
    """
    Reshape EEG features from (N, 310) -> (N, 62, 5).
    feature_idx = channel * 5 + band  [from Stage 1 transpose logic]
    Plot:
      - Mean feature value per emotion per frequency band (band aggregation)
      - Mean feature value per emotion per channel (channel aggregation)
    """
    print("\n" + SEP)
    print("SECTION 4 -- EEG Band & Channel Aggregation")
    print(SEP)

    fcols = ["eeg_" + str(i) for i in range(N_EEG)]
    X = df[fcols].values.astype(np.float64)   # (N, 310)
    y = df["emotion_label"].values

    # Reshape: (N, 62, 5)
    X3d = X.reshape(-1, N_CHANNELS, N_BANDS)

    # ---- Band-wise mean per emotion ----------------------------------------
    band_data = {}   # band_name -> {emotion: mean_over_channels}
    for b, bname in enumerate(BAND_NAMES):
        band_feats = X3d[:, :, b]   # (N, 62) — all channels for this band
        band_data[bname] = {}
        for lbl, ename in EMOTION_NAMES.items():
            mask = y == lbl
            band_data[bname][ename] = float(np.nanmean(band_feats[mask]))

    band_df = pd.DataFrame(band_data).T    # (5 bands, 4 emotions)
    print("  Band-wise mean feature values per emotion (averaged over channels):")
    print(band_df.round(4).to_string())

    # ---- Channel-wise mean per emotion (collapsed over bands) ---------------
    ch_means = {}   # emotion -> (62,) array
    for lbl, ename in EMOTION_NAMES.items():
        mask = y == lbl
        ch_means[ename] = np.nanmean(X3d[mask].mean(axis=2), axis=0)  # (62,)

    # ---- Plot: band-wise grouped bar chart ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("EEG Feature Aggregation by Band and Channel",
                 fontsize=12, fontweight="bold")

    x  = np.arange(N_BANDS)
    w  = 0.18
    emotions = list(EMOTION_NAMES.values())
    for i, (lbl, ename) in enumerate(EMOTION_NAMES.items()):
        vals = [band_data[b][ename] for b in BAND_NAMES]
        axes[0].bar(x + i*w, vals, w, label=ename,
                    color=EMOTION_COLORS[lbl], alpha=0.85)
    axes[0].set_xticks(x + 1.5*w)
    axes[0].set_xticklabels(BAND_NAMES, fontsize=10)
    axes[0].set_title("Band-wise Mean (avg over 62 channels)")
    axes[0].set_ylabel("Mean Normalized Value")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    # Plot: channel-wise bar (too many channels — use line plot)
    for lbl, ename in EMOTION_NAMES.items():
        axes[1].plot(range(N_CHANNELS), ch_means[ename],
                     label=ename, color=EMOTION_COLORS[lbl],
                     lw=1.2, alpha=0.85)
    axes[1].set_title("Channel-wise Mean (avg over 5 bands)")
    axes[1].set_xlabel("Channel Index (0-61)")
    axes[1].set_ylabel("Mean Normalized Value")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "eeg_band_channel_agg.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: eeg_band_channel_agg.png")

    # ---- Heatmap: (channel x band) mean values per emotion ------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, (lbl, ename) in zip(axes, EMOTION_NAMES.items()):
        mask    = y == lbl
        heatmap = np.nanmean(X3d[mask], axis=0)   # (62, 5)
        im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r",
                       vmin=-1.5, vmax=1.5)
        ax.set_title(ename, fontsize=11, fontweight="bold",
                     color=EMOTION_COLORS[lbl])
        ax.set_xlabel("Frequency Band")
        ax.set_xticks(range(N_BANDS))
        ax.set_xticklabels(BAND_NAMES, fontsize=8)
    axes[0].set_ylabel("Channel Index")
    fig.suptitle("EEG Mean Feature Heatmap (Channel x Band) per Emotion",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[-1], label="Normalized Value")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "eeg_heatmap_ch_band.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: eeg_heatmap_ch_band.png")

    return band_df, ch_means


# ============================================================================
# SECTION 5: Eye Feature Interpretation
# ============================================================================
def eye_feature_interpretation(df, sig_df):
    """
    Plot distribution of top-10 most discriminative eye features per emotion.
    """
    print("\n" + SEP)
    print("SECTION 5 -- Eye Feature Interpretation")
    print(SEP)

    fcols = feat_cols(df, N_EYE)
    top_features = sig_df["feature"].head(10).tolist()
    print("  Top 10 discriminative Eye features: " + str(top_features))

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    fig.suptitle("Eye Feature Distributions per Emotion (Top 10 by ANOVA)",
                 fontsize=11, fontweight="bold")

    for ax, feat in zip(axes, top_features):
        for lbl, ename in EMOTION_NAMES.items():
            mask = df["emotion_label"] == lbl
            vals = df.loc[mask, feat].dropna()
            ax.hist(vals, bins=35, alpha=0.50, label=ename,
                    color=EMOTION_COLORS[lbl], density=True)
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel("Normalized Value", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "eye_top10_distributions.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: eye_top10_distributions.png")

    # Box-plots for top 6 eye features
    top6 = top_features[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    fig.suptitle("Eye Feature Box Plots per Emotion (Top 6)",
                 fontsize=11, fontweight="bold")
    for ax, feat in zip(axes, top6):
        data = [df.loc[df["emotion_label"] == lbl, feat].dropna().values
                for lbl in range(4)]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color": "black", "lw": 2},
                        showfliers=False)
        for patch, lbl in zip(bp["boxes"], range(4)):
            patch.set_facecolor(EMOTION_COLORS[lbl])
            patch.set_alpha(0.75)
        ax.set_xticklabels([EMOTION_NAMES[l] for l in range(4)], fontsize=8)
        ax.set_title(feat, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "eye_boxplots.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: eye_boxplots.png")


# ============================================================================
# SECTION 2B: Correlation Heatmap
# ============================================================================
def plot_corr_heatmap(corr, modality, n_feats):
    """Save correlation heatmap. For large matrices (EEG) sample every 5th."""
    step = 5 if n_feats > 100 else 1
    sub_corr = corr.iloc[::step, ::step]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_corr, cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True,
                xticklabels=False, yticklabels=False, ax=ax,
                cbar_kws={"shrink": 0.7})
    title_suffix = (" (every 5th feature shown)" if step == 5 else "")
    ax.set_title(modality.upper() + " Correlation Matrix" + title_suffix,
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = "corr_heatmap_" + modality + ".png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: " + fname)


# ============================================================================
# SECTION 6: Top-feature significance bar charts
# ============================================================================
def plot_significance(sig_df, modality, top_n=30):
    """Bar chart of sig_folds_anova for top features."""
    top = sig_df.head(top_n).copy()
    top = top.sort_values("sig_folds_anova", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(modality.upper() + " -- Top " + str(top_n)
                 + " Feature Significance", fontsize=12, fontweight="bold")

    # Significant folds count
    axes[0].barh(top["feature"], top["sig_folds_anova"],
                 color="#2176AE", alpha=0.8)
    axes[0].axvline(3, color="red", ls="--", lw=1.2, label="3/5 folds")
    axes[0].set_xlabel("Folds Significant (ANOVA, out of 5)")
    axes[0].set_title("Fold Significance Count")
    axes[0].legend(); axes[0].tick_params(axis="y", labelsize=7)

    # Average ANOVA p-value
    axes[1].barh(top["feature"], -np.log10(top["avg_anova_pval"] + 1e-300),
                 color="#E84855", alpha=0.8)
    axes[1].axvline(-np.log10(0.05), color="navy", ls="--", lw=1.2,
                    label="p=0.05")
    axes[1].set_xlabel("-log10(avg ANOVA p-value)")
    axes[1].set_title("Statistical Significance (-log10 scale)")
    axes[1].legend(); axes[1].tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    fname = "significance_" + modality + ".png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=130, bbox_inches="tight")
    plt.close()
    print("  Saved: " + fname)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print(SEP)
    print(" STAGE 3 -- FEATURE ANALYSIS")
    print(SEP)

    # ------------------------------------------------------------------
    # Load combined training data (deduplicated)
    # ------------------------------------------------------------------
    print("\n[Loading training data from all folds...]")
    eeg_df = load_train_all_folds("eeg")
    eye_df = load_train_all_folds("eye")
    print("  EEG combined (dedup): " + str(eeg_df.shape))
    print("  Eye combined (dedup): " + str(eye_df.shape))
    print("  EEG NaN: " + str(eeg_df.isnull().sum().sum()))
    print("  Eye NaN: " + str(eye_df.isnull().sum().sum()))

    # ------------------------------------------------------------------
    # SECTION 1: Statistical tests
    # ------------------------------------------------------------------
    eeg_sig = statistical_tests(eeg_df, "eeg", N_EEG)
    eye_sig = statistical_tests(eye_df, "eye", N_EYE)

    # Save
    eeg_sig.to_csv(os.path.join(OUTPUT_DIR, "eeg_feature_significance.csv"), index=False)
    eye_sig.to_csv(os.path.join(OUTPUT_DIR, "eye_feature_significance.csv"), index=False)
    print("\n  Saved: eeg_feature_significance.csv")
    print("  Saved: eye_feature_significance.csv")

    # Significance bar charts
    plot_significance(eeg_sig, "eeg", top_n=30)
    plot_significance(eye_sig, "eye", top_n=31)

    # ------------------------------------------------------------------
    # SECTION 2: Correlation analysis
    # ------------------------------------------------------------------
    eeg_corr, eeg_high_corr = correlation_analysis(eeg_df, "eeg", N_EEG)
    eye_corr, eye_high_corr = correlation_analysis(eye_df, "eye", N_EYE)

    # Save correlation matrices
    eeg_corr.to_csv(os.path.join(OUTPUT_DIR, "eeg_corr_matrix.csv"))
    eye_corr.to_csv(os.path.join(OUTPUT_DIR, "eye_corr_matrix.csv"))
    eeg_high_corr.to_csv(os.path.join(OUTPUT_DIR, "eeg_high_corr_pairs.csv"), index=False)
    eye_high_corr.to_csv(os.path.join(OUTPUT_DIR, "eye_high_corr_pairs.csv"), index=False)
    print("\n  Saved: eeg_corr_matrix.csv, eye_corr_matrix.csv")

    # Heatmaps
    plot_corr_heatmap(eeg_corr, "eeg", N_EEG)
    plot_corr_heatmap(eye_corr, "eye", N_EYE)

    # ------------------------------------------------------------------
    # SECTION 3: Dimensionality reduction
    # ------------------------------------------------------------------
    eeg_dr_stats = dimensionality_reduction(eeg_df, "eeg", N_EEG, eeg_sig)
    eye_dr_stats = dimensionality_reduction(eye_df, "eye", N_EYE, eye_sig)

    # ------------------------------------------------------------------
    # SECTION 4: EEG band/channel aggregation
    # ------------------------------------------------------------------
    band_df, ch_means = eeg_band_channel_analysis(eeg_df)
    band_df.to_csv(os.path.join(OUTPUT_DIR, "eeg_band_mean_per_emotion.csv"))
    print("  Saved: eeg_band_mean_per_emotion.csv")

    # ------------------------------------------------------------------
    # SECTION 5: Eye feature interpretation
    # ------------------------------------------------------------------
    eye_feature_interpretation(eye_df, eye_sig)

    # ------------------------------------------------------------------
    # Collect key numbers for the report
    # ------------------------------------------------------------------
    n_eeg_sig_both = int((eeg_sig["anova_sig_all"] & eeg_sig["kw_sig_all"]).sum())
    n_eye_sig_both = int((eye_sig["anova_sig_all"] & eye_sig["kw_sig_all"]).sum())
    top5_eeg = eeg_sig["feature"].head(5).tolist()
    top5_eye = eye_sig["feature"].head(5).tolist()

    print("\n" + SEP)
    print(" STAGE 3 COMPLETE")
    print(SEP)
    print("  EEG features significant (both tests): " + str(n_eeg_sig_both) + "/310")
    print("  Eye features significant (both tests): " + str(n_eye_sig_both) + "/31")
    print("  Top 5 EEG: " + str(top5_eeg))
    print("  Top 5 Eye: " + str(top5_eye))
    print("  EEG PCA 95% variance: " + str(eeg_dr_stats["n_components_95pct"]) + " components")
    print("  Eye PCA 95% variance: " + str(eye_dr_stats["n_components_95pct"]) + " components")
    print("  Output dir: " + OUTPUT_DIR)
    print("  Plots dir:  " + PLOT_DIR)
    print()
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not os.path.isdir(os.path.join(OUTPUT_DIR, fname)):
            sz = round(os.path.getsize(os.path.join(OUTPUT_DIR, fname)) / 1024, 1)
            print("    " + fname + "  (" + str(sz) + " KB)")

    return {
        "eeg_sig": eeg_sig, "eye_sig": eye_sig,
        "eeg_corr": eeg_corr, "eye_corr": eye_corr,
        "eeg_high_corr": eeg_high_corr, "eye_high_corr": eye_high_corr,
        "band_df": band_df, "eeg_dr": eeg_dr_stats, "eye_dr": eye_dr_stats,
        "n_eeg_sig_both": n_eeg_sig_both, "n_eye_sig_both": n_eye_sig_both,
        "top5_eeg": top5_eeg, "top5_eye": top5_eye,
    }


if __name__ == "__main__":
    results = main()
