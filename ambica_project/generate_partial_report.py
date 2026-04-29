import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "results", "stratified_loso"))
RESULTS_CSV = os.path.join(OUTPUT_DIR, "subject_results.csv")

def plot_summary(summary_df, out_dir):
    models = summary_df["model"].tolist()
    means  = summary_df["mean_accuracy"].values * 100
    stds   = summary_df["std_accuracy"].values  * 100
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(models, means, xerr=stds, color=colors[:len(models)],
            alpha=0.85, capsize=5)
    ax.set_xlabel("Mean LOSO Test Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Partial LOSO Results — Mean Accuracy ± Std across Subjects")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_performance_chart_partial.png"), dpi=130)
    plt.close()

def plot_variance(subject_df, out_dir):
    models = subject_df["model"].unique().tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [subject_df[subject_df["model"] == m]["accuracy"].values * 100
            for m in models]
    ax.boxplot(data, labels=models, patch_artist=True)
    ax.set_ylabel("LOSO Test Accuracy per Subject (%)")
    ax.set_title("Partial Subject-Level Accuracy Variance (LOSO)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_fold_variance_partial.png"), dpi=130)
    plt.close()

def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"File not found: {RESULTS_CSV}")
        return

    subject_df = pd.read_csv(RESULTS_CSV)
    
    print("\n\n" + "=" * 65)
    print("  PARTIAL RESULTS: Subject | Model | Accuracy")
    print("=" * 65)
    pivot = subject_df.pivot(index="subject", columns="model", values="accuracy")
    print(pivot.to_string())

    model_names = ["MLP", "DNN", "Attention", "Hybrid", "Decision Fusion"]
    summary_rows = []
    
    for mn in model_names:
        m_df = subject_df[subject_df["model"] == mn]
        if m_df.empty:
            continue
        summary_rows.append({
            "model":          mn,
            "mean_accuracy":  round(m_df["accuracy"].mean(),  4),
            "std_accuracy":   round(m_df["accuracy"].std(),   4),
            "mean_precision": round(m_df["precision"].mean(), 4),
            "mean_recall":    round(m_df["recall"].mean(),    4),
            "mean_f1":        round(m_df["f1"].mean(),        4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_results_partial.csv"), index=False)

    print("\n\n" + "=" * 65)
    print("  Model | Mean Accuracy | Std")
    print("=" * 65)
    for row in summary_rows:
        print("  {:20s}  Mean: {:.4f}   Std: {:.4f}".format(
            row["model"], row["mean_accuracy"], row["std_accuracy"]))

    print("\nGenerating partial plots...")
    plot_summary(summary_df, OUTPUT_DIR)
    plot_variance(subject_df, OUTPUT_DIR)
    
    print("\n" + "=" * 65)
    print(f"  PARTIAL REPORT COMPLETE — Results in: {OUTPUT_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
