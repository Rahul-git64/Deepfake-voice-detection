import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from benchmarks.utils import (
    get_classifier,
    compute_eer,
    FROZEN_DEV_THRESHOLD,
    CACHE_DIR,
    RESULTS_DIR,
)

def run_t10(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T10: Threshold Sensitivity & Shift Analysis")
    print("=" * 60)

    # Load Dev
    dev_data = torch.load("data/embeddings/dev_embeddings.pt", weights_only=False)
    dev_emb = dev_data["embeddings"].float()
    dev_labels = dev_data["labels"].numpy()

    clf = get_classifier()
    with torch.no_grad():
        dev_probs = torch.softmax(clf(dev_emb), dim=-1)[:, 1].numpy()

    # Load Eval
    eval_cache_name = "eval_embeddings_full.pt" if full else ("eval_embeddings_quick.pt" if quick else "eval_embeddings_stratified_3000.pt")
    eval_cache = CACHE_DIR / eval_cache_name

    eval_data = torch.load(eval_cache, weights_only=False)
    eval_emb = eval_data["embeddings"]
    eval_labels = eval_data["labels"].numpy()

    with torch.no_grad():
        eval_probs = torch.softmax(clf(eval_emb), dim=-1)[:, 1].numpy()

    # Calculate optimal thresholds
    dev_eer, dev_opt_th, _, _, _ = compute_eer(dev_labels, dev_probs)
    eval_eer, eval_opt_th, _, _, _ = compute_eer(eval_labels, eval_probs)

    print(f"Dev Optimal EER Threshold:  {dev_opt_th:.4f} (EER: {dev_eer*100:.2f}%)")
    print(f"Eval Optimal EER Threshold: {eval_opt_th:.4f} (EER: {eval_eer*100:.2f}%)")
    print(f"Frozen Dev Threshold Used:  {FROZEN_DEV_THRESHOLD:.4f}")

    # Threshold sweep from 0.05 to 0.95 in 0.02 increments
    thresholds = np.linspace(0.05, 0.95, 46)

    sweep_records = []
    for th in thresholds:
        # Dev metrics
        dev_preds = (dev_probs >= th).astype(int)
        dev_fp = np.sum((dev_labels == 0) & (dev_preds == 1))
        dev_tn = np.sum((dev_labels == 0) & (dev_preds == 0))
        dev_fn = np.sum((dev_labels == 1) & (dev_preds == 0))
        dev_tp = np.sum((dev_labels == 1) & (dev_preds == 1))

        dev_far = dev_fn / (dev_tp + dev_fn) if (dev_tp + dev_fn) > 0 else 0.0  # Miss fake
        dev_frr = dev_fp / (dev_tn + dev_fp) if (dev_tn + dev_fp) > 0 else 0.0  # Reject human
        dev_acc = (dev_tp + dev_tn) / len(dev_labels)
        dev_bacc = 0.5 * ((dev_tp / (dev_tp + dev_fn)) + (dev_tn / (dev_tn + dev_fp)))

        # Eval metrics
        eval_preds = (eval_probs >= th).astype(int)
        eval_fp = np.sum((eval_labels == 0) & (eval_preds == 1))
        eval_tn = np.sum((eval_labels == 0) & (eval_preds == 0))
        eval_fn = np.sum((eval_labels == 1) & (eval_preds == 0))
        eval_tp = np.sum((eval_labels == 1) & (eval_preds == 1))

        eval_far = eval_fn / (eval_tp + eval_fn) if (eval_tp + eval_fn) > 0 else 0.0
        eval_frr = eval_fp / (eval_tn + eval_fp) if (eval_tn + eval_fp) > 0 else 0.0
        eval_acc = (eval_tp + eval_tn) / len(eval_labels)
        eval_bacc = 0.5 * ((eval_tp / (eval_tp + eval_fn)) + (eval_tn / (eval_tn + eval_fp)))

        sweep_records.append({
            "threshold": float(th),
            "dev_far": float(dev_far),
            "dev_frr": float(dev_frr),
            "dev_acc": float(dev_acc),
            "dev_bacc": float(dev_bacc),
            "eval_far": float(eval_far),
            "eval_frr": float(eval_frr),
            "eval_acc": float(eval_acc),
            "eval_bacc": float(eval_bacc),
        })

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv(RESULTS_DIR / "t10_threshold_sweep.csv", index=False)
    print(f"Threshold sweep metrics saved to {RESULTS_DIR / 't10_threshold_sweep.csv'}")

    # Plot FAR and FRR curves across thresholds
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Dev Plot
    ax1 = axes[0]
    ax1.plot(sweep_df["threshold"], sweep_df["dev_far"] * 100, label="FAR (Fake Accepted / Missed)", color="#e53e3e", lw=2)
    ax1.plot(sweep_df["threshold"], sweep_df["dev_frr"] * 100, label="FRR (Human Rejected / False Alarm)", color="#3182ce", lw=2)
    ax1.axvline(FROZEN_DEV_THRESHOLD, color="black", linestyle="--", label=f"Frozen Dev Thr ({FROZEN_DEV_THRESHOLD:.4f})")
    ax1.axvline(dev_opt_th, color="green", linestyle=":", label=f"Dev Opt EER Thr ({dev_opt_th:.4f})")
    ax1.set_xlabel("Decision Threshold", fontsize=11)
    ax1.set_ylabel("Error Rate (%)", fontsize=11)
    ax1.set_title("Dev: FAR vs FRR across Decision Thresholds", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Eval Plot
    ax2 = axes[1]
    ax2.plot(sweep_df["threshold"], sweep_df["eval_far"] * 100, label="FAR (Fake Accepted / Missed)", color="#e53e3e", lw=2)
    ax2.plot(sweep_df["threshold"], sweep_df["eval_frr"] * 100, label="FRR (Human Rejected / False Alarm)", color="#3182ce", lw=2)
    ax2.axvline(FROZEN_DEV_THRESHOLD, color="black", linestyle="--", label=f"Frozen Dev Thr ({FROZEN_DEV_THRESHOLD:.4f})")
    ax2.axvline(eval_opt_th, color="purple", linestyle=":", label=f"Eval Opt EER Thr ({eval_opt_th:.4f})")
    ax2.set_xlabel("Decision Threshold", fontsize=11)
    ax2.set_ylabel("Error Rate (%)", fontsize=11)
    ax2.set_title("Eval: FAR vs FRR across Decision Thresholds", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "t10_threshold_sensitivity.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Threshold sensitivity plot saved to {plot_path}")

    return {
        "dev_optimal_threshold": float(dev_opt_th),
        "dev_optimal_eer": float(dev_eer),
        "eval_optimal_threshold": float(eval_opt_th),
        "eval_optimal_eer": float(eval_eer),
        "frozen_threshold": float(FROZEN_DEV_THRESHOLD),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t10(quick=args.quick, full=args.full)
