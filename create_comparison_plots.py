import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "final_dataset_ml_ready_numeric_plus_extended_overfitting_report.csv"
OUT_PNG = "model_comparison_metrics.png"


def main():
    df = pd.read_csv(CSV_PATH)

    models = df["model"].tolist()
    roc_vals = df["val_roc_mean"].tolist()
    pr_vals = df["val_pr_mean"].tolist()

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], roc_vals, width, label="ROC-AUC", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], pr_vals, width, label="PR-AUC", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score")
    ax.set_title("Model Comparison – Validation Metrics (5-fold CV)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main() 