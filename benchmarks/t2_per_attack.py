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

ATTACK_NAMES = {
    "A01": "Neural TTS (WF/Tacotron)",
    "A02": "Vocoder (WORLD)",
    "A03": "Vocoder (FD)",
    "A04": "Waveform Concatenation",
    "A05": "VC (Voice Conversion)",
    "A06": "VC (Transform-based)",
    "A07": "Vocoder (WORLD+Neural)",
    "A08": "Neural TTS (Mel-scale)",
    "A09": "Neural VC (Deep CNN)",
    "A10": "Neural TTS (Seq2seq)",
    "A11": "Neural VC (VAEs)",
    "A12": "Neural TTS (Tacotron2)",
    "A13": "VC (WaveNet-based)",
    "A14": "Neural VC (CycleGAN)",
    "A15": "Neural VC (StarGAN)",
    "A16": "Neural TTS (Transformer)",
    "A17": "VC (Transfer learning)",
    "A18": "VC (Direct waveform)",
    "A19": "VC (Spectral transfer)",
}

def run_t2(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T2: Per-Attack Breakdown Test")
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

    eval_data = torch.load(eval_cache, weights_only=False)
    eval_emb = eval_data["embeddings"]
    eval_labels = eval_data["labels"].numpy()
    eval_attacks = np.array(eval_data["attacks"])
    eval_bonafide_mask = (eval_labels == 0)

    clf = get_classifier()
    with torch.no_grad():
        eval_probs = torch.softmax(clf(eval_emb), dim=-1)[:, 1].numpy()

    # Load dev data to also evaluate seen attacks A01-A06
    dev_data = torch.load(dev_cache, weights_only=False)
    dev_emb = dev_data["embeddings"].float()
    dev_labels = dev_data["labels"].numpy()
    with torch.no_grad():
        dev_probs = torch.softmax(clf(dev_emb), dim=-1)[:, 1].numpy()

    # Load dev protocols to get dev attack mapping
    from benchmarks.utils import get_dev_protocol_df
    dev_df = get_dev_protocol_df()
    fn_to_dev_atk = dict(zip(dev_df["filename"], dev_df["attack"]))
    dev_attacks = np.array([fn_to_dev_atk.get(fn, "-") for fn in dev_data["filenames"]])
    dev_bonafide_mask = (dev_labels == 0)

    results = []

    # Attacks in training: A01 to A06
    for atk in [f"A{i:02d}" for i in range(1, 7)]:
        mask = (dev_attacks == atk)
        n_samples = np.sum(mask)
        if n_samples == 0:
            continue

        spoof_scores = dev_probs[mask]
        bona_scores = dev_probs[dev_bonafide_mask]

        # Spoof detection rate at frozen threshold (0.6402)
        detection_rate = np.mean(spoof_scores >= FROZEN_DEV_THRESHOLD)

        # Attack EER vs bona fide
        y_pair = np.concatenate([np.ones(len(spoof_scores)), np.zeros(len(bona_scores))])
        scores_pair = np.concatenate([spoof_scores, bona_scores])
        eer, th, _, _, _ = compute_eer(y_pair, scores_pair)

        results.append({
            "attack_id": atk,
            "type": "Seen in Training",
            "split": "dev",
            "description": ATTACK_NAMES.get(atk, "Unknown"),
            "num_clips": int(n_samples),
            "detection_rate_at_frozen": float(detection_rate),
            "eer_vs_bonafide": float(eer),
            "eer_threshold": float(th),
            "mean_spoof_prob": float(np.mean(spoof_scores)),
        })

    # Unseen attacks: A07 to A19
    for atk in [f"A{i:02d}" for i in range(7, 20)]:
        mask = (eval_attacks == atk)
        n_samples = np.sum(mask)
        if n_samples == 0:
            continue

        spoof_scores = eval_probs[mask]
        bona_scores = eval_probs[eval_bonafide_mask]

        detection_rate = np.mean(spoof_scores >= FROZEN_DEV_THRESHOLD)

        y_pair = np.concatenate([np.ones(len(spoof_scores)), np.zeros(len(bona_scores))])
        scores_pair = np.concatenate([spoof_scores, bona_scores])
        eer, th, _, _, _ = compute_eer(y_pair, scores_pair)

        results.append({
            "attack_id": atk,
            "type": "Unseen in Eval",
            "split": "eval",
            "description": ATTACK_NAMES.get(atk, "Unknown"),
            "num_clips": int(n_samples),
            "detection_rate_at_frozen": float(detection_rate),
            "eer_vs_bonafide": float(eer),
            "eer_threshold": float(th),
            "mean_spoof_prob": float(np.mean(spoof_scores)),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "t2_per_attack.csv", index=False)
    print(f"Per-attack metrics saved to {RESULTS_DIR / 't2_per_attack.csv'}")

    # Display table
    print("\n--- Per-Attack Performance Summary ---")
    print(res_df[["attack_id", "type", "num_clips", "detection_rate_at_frozen", "eer_vs_bonafide", "mean_spoof_prob"]].to_string(index=False))

    # Plot Bar Chart sorted by difficulty (lowest detection rate first)
    sorted_df = res_df.sort_values(by="detection_rate_at_frozen", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2b6cb0" if t == "Seen in Training" else "#c53030" for t in sorted_df["type"]]
    bars = ax.bar(sorted_df["attack_id"], sorted_df["detection_rate_at_frozen"] * 100, color=colors, edgecolor="black", alpha=0.85)

    ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Detection Rate (%) @ Thr=0.6402", fontsize=12)
    ax.set_xlabel("Attack ID", fontsize=12)
    ax.set_title("Detection Rate by Attack Algorithm (Sorted by Difficulty)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)

    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2b6cb0", edgecolor="black", label="Seen in Train (A01-A06)"),
        Patch(facecolor="#c53030", edgecolor="black", label="Unseen in Eval (A07-A19)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True)

    # Annotate values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90 if h < 90 else 0)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "t2_per_attack.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Bar chart plot saved to {plot_path}")

    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t2(quick=args.quick, full=args.full)
