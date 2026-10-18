import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from benchmarks.utils import (
    get_classifier,
    CACHE_DIR,
    RESULTS_DIR,
)

def compute_ece(y_true, y_probs, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and bin statistics.
    y_true: binary labels (0 or 1)
    y_probs: predicted probability for positive class (spoof)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_probs, bins) - 1
    # Fix edge case where prob == 1.0
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = (bin_indices == i)
        count = np.sum(mask)
        bin_counts.append(int(count))
        if count > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_probs[mask])
            bin_accs.append(float(acc))
            bin_confs.append(float(conf))
            ece += (count / total_samples) * np.abs(acc - conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bins[i] + bins[i+1]) / 2.0)

    return float(ece), bin_accs, bin_confs, bin_counts, bins

def run_t9(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T9: Calibration Analysis & Reliability Diagrams")
    print("=" * 60)

    # 1. Dev Embeddings
    dev_data = torch.load("data/embeddings/dev_embeddings.pt", weights_only=False)
    dev_emb = dev_data["embeddings"].float()
    dev_labels = dev_data["labels"].numpy()

    clf = get_classifier()
    with torch.no_grad():
        dev_probs = torch.softmax(clf(dev_emb), dim=-1)[:, 1].numpy()

    dev_ece, dev_accs, dev_confs, dev_counts, bins = compute_ece(dev_labels, dev_probs, n_bins=10)

    # 2. Eval Embeddings
    eval_cache_name = "eval_embeddings_full.pt" if full else ("eval_embeddings_quick.pt" if quick else "eval_embeddings_stratified_3000.pt")
    eval_cache = CACHE_DIR / eval_cache_name

    eval_data = torch.load(eval_cache, weights_only=False)
    eval_emb = eval_data["embeddings"]
    eval_labels = eval_data["labels"].numpy()

    with torch.no_grad():
        eval_probs = torch.softmax(clf(eval_emb), dim=-1)[:, 1].numpy()

    eval_ece, eval_accs, eval_confs, eval_counts, _ = compute_ece(eval_labels, eval_probs, n_bins=10)

    print(f"Dev Expected Calibration Error (ECE, 10 bins):  {dev_ece*100:.2f}%")
    print(f"Eval Expected Calibration Error (ECE, 10 bins): {eval_ece*100:.2f}%")

    res_df = pd.DataFrame([
        {"Split": "Dev", "ECE": dev_ece, "Samples": len(dev_labels)},
        {"Split": "Eval", "ECE": eval_ece, "Samples": len(eval_labels)},
    ])
    res_df.to_csv(RESULTS_DIR / "t9_calibration.csv", index=False)
    print(f"Calibration metrics saved to {RESULTS_DIR / 't9_calibration.csv'}")

    # Plot Reliability Diagrams
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Dev diagram
    ax1 = axes[0]
    ax1.bar(bin_centers, dev_accs, width=0.08, alpha=0.7, color="#2b6cb0", edgecolor="black", label="Outputs")
    ax1.plot([0, 1], [0, 1], "--", color="red", label="Perfect Calibration")
    ax1.set_xlabel("Confidence (Predicted Probability)", fontsize=11)
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_title(f"Dev Reliability Diagram (ECE = {dev_ece*100:.2f}%)", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # Eval diagram
    ax2 = axes[1]
    ax2.bar(bin_centers, eval_accs, width=0.08, alpha=0.7, color="#c53030", edgecolor="black", label="Outputs")
    ax2.plot([0, 1], [0, 1], "--", color="red", label="Perfect Calibration")
    ax2.set_xlabel("Confidence (Predicted Probability)", fontsize=11)
    ax2.set_ylabel("Empirical Accuracy", fontsize=11)
    ax2.set_title(f"Eval Reliability Diagram (ECE = {eval_ece*100:.2f}%)", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "t9_calibration.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Reliability diagram plot saved to {plot_path}")

    # Plot Confidence Histograms for Correct vs Incorrect predictions
    dev_preds = (dev_probs >= 0.5).astype(int)
    dev_correct = (dev_preds == dev_labels)

    eval_preds = (eval_probs >= 0.5).astype(int)
    eval_correct = (eval_preds == eval_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(dev_probs[dev_correct], bins=25, alpha=0.6, color="green", label="Dev Correct")
    ax1.hist(dev_probs[~dev_correct], bins=25, alpha=0.7, color="red", label="Dev Incorrect")
    ax1.set_title("Dev: Correct vs Incorrect Predictions by Confidence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Spoof Probability")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.hist(eval_probs[eval_correct], bins=25, alpha=0.6, color="green", label="Eval Correct")
    ax2.hist(eval_probs[~eval_correct], bins=25, alpha=0.7, color="red", label="Eval Incorrect")
    ax2.set_title("Eval: Correct vs Incorrect Predictions by Confidence", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Spoof Probability")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    conf_plot_path = RESULTS_DIR / "t9_confidence_hist.png"
    plt.savefig(conf_plot_path, dpi=200)
    plt.close()
    print(f"Confidence histogram plot saved to {conf_plot_path}")

    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t9(quick=args.quick, full=args.full)
