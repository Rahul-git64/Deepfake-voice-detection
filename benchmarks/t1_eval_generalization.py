import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm

from benchmarks.utils import (
    sample_eval_clips,
    preprocess_audio_bytes,
    extract_embedding_from_waveform,
    predict_from_embedding,
    evaluate_metrics,
    get_wav2vec2,
    get_classifier,
    FROZEN_DEV_THRESHOLD,
    EVAL_DIR,
    CACHE_DIR,
    RESULTS_DIR,
)

def run_t1(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T1: Evaluation Generalization Test")
    print("=" * 60)

    if full:
        sample_size = 71237
        min_atk = 4914
        cache_name = "eval_embeddings_full.pt"
    elif quick:
        sample_size = 300
        min_atk = 15
        cache_name = "eval_embeddings_quick.pt"
    else:
        sample_size = 3000
        min_atk = 150
        cache_name = "eval_embeddings_stratified_3000.pt"

    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        print(f"Loading cached eval embeddings from {cache_path}...")
        data = torch.load(cache_path, weights_only=False)
        embeddings = data["embeddings"]
        labels = data["labels"]
        filenames = data["filenames"]
        attacks = data.get("attacks", [])
        speakers = data.get("speakers", [])
    else:
        print(f"Preparing eval dataset (target={sample_size}, min_per_attack={min_atk})...")
        if full:
            from benchmarks.utils import get_eval_protocol_df
            df = get_eval_protocol_df()
        else:
            df = sample_eval_clips(target_total=sample_size, min_per_attack=min_atk)

        print(f"Selected {len(df)} eval clips. Extracting features using wav2vec2...")
        wav2vec2 = get_wav2vec2()

        embeddings_list = []
        labels_list = []
        filenames_list = []
        attacks_list = []
        speakers_list = []

        t0 = time.time()
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting eval embeddings"):
            fn = row["filename"]
            flac_path = EVAL_DIR / f"{fn}.flac"
            if not flac_path.exists():
                print(f"Warning: {flac_path} not found, skipping.")
                continue

            with open(flac_path, "rb") as f:
                raw_bytes = f.read()

            try:
                waveform = preprocess_audio_bytes(raw_bytes)
                emb = extract_embedding_from_waveform(waveform, wav2vec2_model=wav2vec2)
                embeddings_list.append(emb.cpu())
                labels_list.append(1 if row["label"] == "spoof" else 0)
                filenames_list.append(fn)
                attacks_list.append(row["attack"])
                speakers_list.append(row["speaker"])
            except Exception as e:
                print(f"Error processing {fn}: {e}")

        elapsed = time.time() - t0
        print(f"Extracted {len(embeddings_list)} embeddings in {elapsed:.1f} s ({elapsed/max(1, len(embeddings_list)):.3f} s/clip)")

        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.tensor(labels_list, dtype=torch.long)
        filenames = filenames_list
        attacks = attacks_list
        speakers = speakers_list

        torch.save(
            {
                "embeddings": embeddings,
                "labels": labels,
                "filenames": filenames,
                "attacks": attacks,
                "speakers": speakers,
            },
            cache_path,
        )
        print(f"Cached eval embeddings saved to {cache_path}")

    # Model evaluation on eval split
    classifier = get_classifier()
    with torch.no_grad():
        logits = classifier(embeddings)
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()

    y_true = labels.numpy()
    metrics_frozen = evaluate_metrics(y_true, probs, threshold=FROZEN_DEV_THRESHOLD)
    metrics_05 = evaluate_metrics(y_true, probs, threshold=0.5)

    print("\n--- T1 Eval Results ---")
    print(f"Total Eval Clips: {len(y_true)} (Spoof: {sum(y_true == 1)}, Bonafide: {sum(y_true == 0)})")
    print(f"Optimal Eval EER: {metrics_frozen['eer']*100:.2f}% (at threshold {metrics_frozen['eer_threshold']:.4f})")
    print(f"Metrics at FROZEN Dev Threshold ({FROZEN_DEV_THRESHOLD:.4f}):")
    print(f"  Accuracy:          {metrics_frozen['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {metrics_frozen['balanced_accuracy']*100:.2f}%")
    print(f"  ROC-AUC:           {metrics_frozen['roc_auc']:.4f}")
    print(f"  PR-AUC:            {metrics_frozen['pr_auc']:.4f}")
    print(f"  FAR (Spoof Miss):  {metrics_frozen['far']*100:.2f}%")
    print(f"  FRR (Human False): {metrics_frozen['frr']*100:.2f}%")
    print(f"  Confusion Matrix:  {metrics_frozen['confusion_matrix']}")

    print(f"\nMetrics at 0.5 Threshold:")
    print(f"  Accuracy:          {metrics_05['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {metrics_05['balanced_accuracy']*100:.2f}%")
    print(f"  FAR:               {metrics_05['far']*100:.2f}%")
    print(f"  FRR:               {metrics_05['frr']*100:.2f}%")

    # Save to CSV and JSON
    res_df = pd.DataFrame([
        {
            "split": "eval",
            "mode": "quick" if quick else ("full" if full else "stratified_3000"),
            "num_samples": len(y_true),
            "eer": metrics_frozen["eer"],
            "eer_optimal_threshold": metrics_frozen["eer_threshold"],
            "frozen_dev_threshold": FROZEN_DEV_THRESHOLD,
            "accuracy_at_frozen": metrics_frozen["accuracy"],
            "balanced_accuracy_at_frozen": metrics_frozen["balanced_accuracy"],
            "roc_auc": metrics_frozen["roc_auc"],
            "pr_auc": metrics_frozen["pr_auc"],
            "far_at_frozen": metrics_frozen["far"],
            "frr_at_frozen": metrics_frozen["frr"],
            "accuracy_at_05": metrics_05["accuracy"],
            "balanced_accuracy_at_05": metrics_05["balanced_accuracy"],
            "far_at_05": metrics_05["far"],
            "frr_at_05": metrics_05["frr"],
        }
    ])
    res_df.to_csv(RESULTS_DIR / "t1_eval_generalization.csv", index=False)
    print(f"Results saved to {RESULTS_DIR / 't1_eval_generalization.csv'}")

    return metrics_frozen, metrics_05

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t1(quick=args.quick, full=args.full)
