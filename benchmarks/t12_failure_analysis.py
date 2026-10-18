import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display

from benchmarks.utils import (
    get_classifier,
    FROZEN_DEV_THRESHOLD,
    EVAL_DIR,
    CACHE_DIR,
    RESULTS_DIR,
)

def run_t12(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T12: Failure Case Analysis (Top False Accepts & Rejects)")
    print("=" * 60)

    eval_cache_name = "eval_embeddings_full.pt" if full else ("eval_embeddings_quick.pt" if quick else "eval_embeddings_stratified_3000.pt")
    eval_cache = CACHE_DIR / eval_cache_name

    if not eval_cache.exists():
        print(f"Eval cache {eval_cache} not found. Running T1 first...")
        from benchmarks.t1_eval_generalization import run_t1
        run_t1(quick=quick, full=full)

    data = torch.load(eval_cache, weights_only=False)
    emb = data["embeddings"]
    labels = data["labels"].numpy()
    filenames = np.array(data["filenames"])
    attacks = np.array(data["attacks"])
    speakers = np.array(data.get("speakers", ["-"] * len(labels)))

    clf = get_classifier()
    with torch.no_grad():
        probs = torch.softmax(clf(emb), dim=-1)[:, 1].numpy()

    # Definitions:
    # False Accept (FA): True label = 1 (Spoof), but predicted as Bonafide (prob < FROZEN_DEV_THRESHOLD).
    # Most confident FA: lowest predicted spoof probability (closest to 0.0).
    fa_mask = (labels == 1) & (probs < FROZEN_DEV_THRESHOLD)
    fa_indices = np.where(fa_mask)[0]
    # Sort by spoof_prob ascending (most confident spoof misclassified as human)
    sorted_fa_idx = fa_indices[np.argsort(probs[fa_indices])]

    # False Reject (FR): True label = 0 (Bonafide), but predicted as Spoof (prob >= FROZEN_DEV_THRESHOLD).
    # Most confident FR: highest predicted spoof probability (closest to 1.0).
    fr_mask = (labels == 0) & (probs >= FROZEN_DEV_THRESHOLD)
    fr_indices = np.where(fr_mask)[0]
    sorted_fr_idx = fr_indices[np.argsort(probs[fr_indices])[::-1]]

    top_n = 5 if quick else 30
    top_fa_idx = sorted_fa_idx[:top_n]
    top_fr_idx = sorted_fr_idx[:top_n]

    print(f"Total False Accepts in Eval: {len(sorted_fa_idx)}/{np.sum(labels == 1)}")
    print(f"Total False Rejects in Eval: {len(sorted_fr_idx)}/{np.sum(labels == 0)}")

    fa_rows = []
    for rank, idx in enumerate(top_fa_idx, 1):
        fa_rows.append({
            "rank": rank,
            "type": "False Accept (Missed Fake)",
            "filename": filenames[idx],
            "attack": attacks[idx],
            "speaker": speakers[idx],
            "spoof_prob": float(probs[idx]),
            "bonafide_prob": float(1.0 - probs[idx]),
        })

    fr_rows = []
    for rank, idx in enumerate(top_fr_idx, 1):
        fr_rows.append({
            "rank": rank,
            "type": "False Reject (False Alarm on Human)",
            "filename": filenames[idx],
            "attack": attacks[idx],
            "speaker": speakers[idx],
            "spoof_prob": float(probs[idx]),
            "bonafide_prob": float(1.0 - probs[idx]),
        })

    df_top_failures = pd.concat([pd.DataFrame(fa_rows), pd.DataFrame(fr_rows)], ignore_index=True)
    df_top_failures.to_csv(RESULTS_DIR / "t12_failure_analysis.csv", index=False)
    print(f"Top failure cases saved to {RESULTS_DIR / 't12_failure_analysis.csv'}")

    # Generate waveform + spectrogram inspection plots for top 10 (5 FA + 5 FR)
    inspect_dir = RESULTS_DIR / "failure_plots"
    inspect_dir.mkdir(parents=True, exist_ok=True)

    plot_samples = [("FA", r) for r in fa_rows[:min(5, len(fa_rows))]] + [("FR", r) for r in fr_rows[:min(5, len(fr_rows))]]

    print(f"Generating diagnostic plots for {len(plot_samples)} top failures...")
    for ftype, item in plot_samples:
        fn = item["filename"]
        flac_path = EVAL_DIR / f"{fn}.flac"
        if not flac_path.exists():
            continue

        audio, sr = sf.read(flac_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Waveform
        t = np.linspace(0, len(audio) / sr, len(audio))
        ax_wave.plot(t, audio, color="#2b6cb0", lw=0.8)
        ax_wave.set_ylabel("Amplitude")
        title_str = (
            f"[{ftype} #{item['rank']}] {fn} | Attack: {item['attack']} | "
            f"Spoof Prob: {item['spoof_prob']:.4f} (Threshold: {FROZEN_DEV_THRESHOLD})"
        )
        ax_wave.set_title(title_str, fontsize=11, fontweight="bold")
        ax_wave.grid(alpha=0.3)

        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=1024, hop_length=256)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax_spec, cmap="magma")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_xlabel("Time (s)")
        fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")

        plt.tight_layout()
        plot_path = inspect_dir / f"{ftype}_{item['rank']:02d}_{fn}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"Diagnostic waveform and spectrogram plots saved to {inspect_dir}")

    # Attack frequency among all false accepts
    fa_attack_counts = pd.Series(attacks[fa_indices]).value_counts().to_dict()
    print("\nAttack breakdown among False Accepts (Uncaught Spoofs):", fa_attack_counts)

    return df_top_failures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t12(quick=args.quick, full=args.full)
