import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

from benchmarks.utils import (
    get_classifier,
    evaluate_metrics,
    FROZEN_DEV_THRESHOLD,
    CACHE_DIR,
    RESULTS_DIR,
)

def plot_det_curve(fpr, fnr, ax, label="Classifier"):
    """Plot Detection Error Tradeoff (DET) curve using normal deviate scale."""
    # Clip probabilities to avoid inf in probit transform
    fpr_clipped = np.clip(fpr, 1e-4, 1.0 - 1e-4)
    fnr_clipped = np.clip(fnr, 1e-4, 1.0 - 1e-4)
    
    x = norm.ppf(fpr_clipped)
    y = norm.ppf(fnr_clipped)
    
    ax.plot(x, y, label=label, lw=2)

def run_t3(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T3: Full Metrics & ROC/DET Curves")
    print("=" * 60)

    if full:
        cache_name = "eval_embeddings_full.pt"
    elif quick:
        cache_name = "eval_embeddings_quick.pt"
    else:
        cache_name = "eval_embeddings_stratified_3000.pt"

    eval_cache = CACHE_DIR / cache_name
    dev_cache = Path("data/embeddings/dev_embeddings.pt")

    if not eval_cache.exists():
        print(f"Eval cache {eval_cache} not found. Running T1 first...")
        from benchmarks.t1_eval_generalization import run_t1
        run_t1(quick=quick, full=full)

    # 1. Evaluate on Dev
    dev_data = torch.load(dev_cache, weights_only=False)
    dev_emb = dev_data["embeddings"].float()
    dev_labels = dev_data["labels"].numpy()

    clf = get_classifier()
    with torch.no_grad():
        dev_probs = torch.softmax(clf(dev_emb), dim=-1)[:, 1].numpy()

    dev_metrics_frozen = evaluate_metrics(dev_labels, dev_probs, threshold=FROZEN_DEV_THRESHOLD)
    dev_metrics_05 = evaluate_metrics(dev_labels, dev_probs, threshold=0.5)

    # 2. Evaluate on Eval
    eval_data = torch.load(eval_cache, weights_only=False)
    eval_emb = eval_data["embeddings"]
    eval_labels = eval_data["labels"].numpy()
    with torch.no_grad():
        eval_probs = torch.softmax(clf(eval_emb), dim=-1)[:, 1].numpy()

    eval_metrics_frozen = evaluate_metrics(eval_labels, eval_probs, threshold=FROZEN_DEV_THRESHOLD)
    eval_metrics_05 = evaluate_metrics(eval_labels, eval_probs, threshold=0.5)

    # Summary table
    table_rows = [
        {"Split": "Dev", "Threshold": 0.5, **dev_metrics_05},
        {"Split": "Dev", "Threshold": FROZEN_DEV_THRESHOLD, **dev_metrics_frozen},
        {"Split": "Eval", "Threshold": 0.5, **eval_metrics_05},
        {"Split": "Eval", "Threshold": FROZEN_DEV_THRESHOLD, **eval_metrics_frozen},
    ]

    summary_df = pd.DataFrame(table_rows)
    summary_df.to_csv(RESULTS_DIR / "t3_full_metrics.csv", index=False)
    print(f"Full metrics summary saved to {RESULTS_DIR / 't3_full_metrics.csv'}")

    # Plot ROC Curves
    dev_fpr, dev_tpr, _ = roc_curve(dev_labels, dev_probs, pos_label=1)
    eval_fpr, eval_tpr, _ = roc_curve(eval_labels, eval_probs, pos_label=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(dev_fpr, dev_tpr, label=f"Dev (AUC = {auc(dev_fpr, dev_tpr):.4f}, EER = {dev_metrics_frozen['eer']*100:.2f}%)", color="#2b6cb0", lw=2)
    ax.plot(eval_fpr, eval_tpr, label=f"Eval (AUC = {auc(eval_fpr, eval_tpr):.4f}, EER = {eval_metrics_frozen['eer']*100:.2f}%)", color="#c53030", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate (Bona fide as Spoof)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Spoof Detected)", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = RESULTS_DIR / "t3_roc_curve.png"
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    print(f"ROC plot saved to {roc_plot_path}")

    # Plot DET Curves
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_det_curve(dev_fpr, 1.0 - dev_tpr, ax, label=f"Dev (EER = {dev_metrics_frozen['eer']*100:.2f}%)")
    plot_det_curve(eval_fpr, 1.0 - eval_tpr, ax, label=f"Eval (EER = {eval_metrics_frozen['eer']*100:.2f}%)")

    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    tick_labels = ["0.1%", "1%", "5%", "10%", "20%", "50%"]
    norm_ticks = norm.ppf(ticks)
    ax.set_xticks(norm_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(norm_ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("False Alarm Rate (Human rejected)", fontsize=11)
    ax.set_ylabel("Miss Rate (Fake accepted)", fontsize=11)
    ax.set_title("Detection Error Tradeoff (DET) Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    det_plot_path = RESULTS_DIR / "t3_det_curve.png"
    plt.savefig(det_plot_path, dpi=200)
    plt.close()
    print(f"DET plot saved to {det_plot_path}")

    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t3(quick=args.quick, full=args.full)
