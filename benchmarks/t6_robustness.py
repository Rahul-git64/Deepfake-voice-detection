import os
import sys
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
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
    DEV_DIR,
    EVAL_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
)

# Robustness transform implementations

def apply_mp3_compression(wav_np: np.ndarray, sr: int, bitrate_kbps: int) -> np.ndarray:
    """Encodes to mp3 at specified bitrate and decodes back via ffmpeg subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_path = in_f.name
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
        mp3_path = mp3_f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
        out_path = out_f.name

    try:
        sf.write(in_path, wav_np, sr)
        # Convert wav to mp3
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-b:a", f"{bitrate_kbps}k", mp3_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        # Decode mp3 back to wav
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(sr), out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        out_data, _ = sf.read(out_path)
        return out_data
    finally:
        for p in [in_path, mp3_path, out_path]:
            if os.path.exists(p):
                os.remove(p)

def apply_telephone_simulation(wav_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulates PSTN telephone line:
    1. Bandpass filter 300 - 3400 Hz (Chebyshev / Butterworth 5th order).
    2. Downsample to 8 kHz.
    3. 8-bit mu-law companding (encode and decode).
    4. Resample back to 16 kHz.
    """
    # 1. Bandpass filter (300 to 3400 Hz)
    sos = signal.butter(5, [300, 3400], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, wav_np)

    # 2. Downsample to 8 kHz
    num_samples_8k = int(len(filtered) * 8000 / sr)
    resampled_8k = signal.resample(filtered, num_samples_8k)

    # 3. 8-bit mu-law companding
    mu = 255.0
    x = np.clip(resampled_8k, -1.0, 1.0)
    mu_law_enc = np.sign(x) * (np.log(1.0 + mu * np.abs(x)) / np.log(1.0 + mu))
    # Quantize to 8-bit signed (-128 to 127)
    quantized = np.round(mu_law_enc * 128.0) / 128.0
    # Mu-law expand back to linear
    mu_law_dec = np.sign(quantized) * (1.0 / mu) * ((1.0 + mu) ** np.abs(quantized) - 1.0)

    # 4. Upsample back to 16 kHz
    num_samples_16k = int(len(mu_law_dec) * sr / 8000)
    resampled_16k = signal.resample(mu_law_dec, num_samples_16k)
    return resampled_16k.astype(np.float32)

def apply_additive_white_noise(wav_np: np.ndarray, snr_db: float) -> np.ndarray:
    """Adds white Gaussian noise at the specified SNR (dB)."""
    signal_power = np.mean(wav_np ** 2)
    if signal_power == 0:
        return wav_np
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(wav_np))
    return (wav_np + noise).astype(np.float32)

def apply_gain_and_clipping(wav_np: np.ndarray, gain_db: float) -> np.ndarray:
    """Applies gain in dB and clips to [-1.0, 1.0]."""
    gain_linear = 10.0 ** (gain_db / 20.0)
    return np.clip(wav_np * gain_linear, -1.0, 1.0).astype(np.float32)

def apply_truncation(wav_np: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Truncates audio to first duration_sec seconds."""
    max_len = int(duration_sec * sr)
    if len(wav_np) <= max_len:
        return wav_np
    return wav_np[:max_len]

def run_t6(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T6: Robustness & Audio Perturbation Test")
    print("=" * 60)

    # Use a balanced sample of clips from dev or eval
    target_clips = 50 if quick else 1000
    cache_path = CACHE_DIR / f"robustness_raw_clips_{target_clips}.pt"

    from benchmarks.utils import get_dev_protocol_df
    dev_df = get_dev_protocol_df()

    if cache_path.exists():
        print(f"Loading cached robustness audio clips from {cache_path}...")
        sample_clips = torch.load(cache_path, weights_only=False)
    else:
        print(f"Sampling {target_clips} balanced clips from Dev split...")
        half = target_clips // 2
        spoof_df = dev_df[dev_df["label"] == "spoof"].sample(n=half, random_state=RANDOM_SEED)
        bona_df = dev_df[dev_df["label"] == "bonafide"].sample(n=half, random_state=RANDOM_SEED)
        sampled_df = pd.concat([spoof_df, bona_df]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        sample_clips = []
        for _, row in sampled_df.iterrows():
            fn = row["filename"]
            flac_path = DEV_DIR / f"{fn}.flac"
            if not flac_path.exists():
                continue
            data, sr = sf.read(flac_path)
            if data.ndim > 1:
                data = data.mean(axis=-1)
            sample_clips.append({
                "filename": fn,
                "label": 1 if row["label"] == "spoof" else 0,
                "attack": row["attack"],
                "audio": data.astype(np.float32),
                "sr": sr,
            })
        torch.save(sample_clips, cache_path)
        print(f"Loaded and saved {len(sample_clips)} raw audio clips to {cache_path}")

    wav2vec2 = get_wav2vec2()
    classifier = get_classifier()

    # Define conditions to test
    conditions = [
        ("Clean Baseline", lambda w, sr: w),
        ("MP3 128 kbps", lambda w, sr: apply_mp3_compression(w, sr, 128)),
        ("MP3 64 kbps", lambda w, sr: apply_mp3_compression(w, sr, 64)),
        ("MP3 32 kbps", lambda w, sr: apply_mp3_compression(w, sr, 32)),
        ("Telephone Simulation (PSTN)", lambda w, sr: apply_telephone_simulation(w, sr)),
        ("White Noise (SNR 20 dB)", lambda w, sr: apply_additive_white_noise(w, 20.0)),
        ("White Noise (SNR 10 dB)", lambda w, sr: apply_additive_white_noise(w, 10.0)),
        ("White Noise (SNR 5 dB)", lambda w, sr: apply_additive_white_noise(w, 5.0)),
        ("Gain -10 dB", lambda w, sr: apply_gain_and_clipping(w, -10.0)),
        ("Gain +10 dB (with clipping)", lambda w, sr: apply_gain_and_clipping(w, 10.0)),
        ("Truncation 4s", lambda w, sr: apply_truncation(w, sr, 4.0)),
        ("Truncation 2s", lambda w, sr: apply_truncation(w, sr, 2.0)),
        ("Truncation 1s", lambda w, sr: apply_truncation(w, sr, 1.0)),
    ]

    results = []
    y_true = np.array([c["label"] for c in sample_clips])

    for cond_name, transform_fn in conditions:
        print(f"\nProcessing condition: {cond_name}...")
        cond_cache = CACHE_DIR / f"robustness_{cond_name.replace(' ', '_').replace('+', 'plus').replace('-', 'minus')}_{target_clips}.pt"

        if cond_cache.exists():
            print(f"Loading cached predictions for {cond_name}...")
            probs = torch.load(cond_cache, weights_only=False)
        else:
            probs_list = []
            for item in tqdm(sample_clips, desc=cond_name):
                audio = item["audio"]
                sr = item["sr"]
                try:
                    transformed = transform_fn(audio, sr)
                    waveform = torch.from_numpy(transformed).float()
                    emb = extract_embedding_from_waveform(waveform, wav2vec2_model=wav2vec2)
                    _, spoof_prob = predict_from_embedding(emb, classifier=classifier)
                    probs_list.append(spoof_prob)
                except Exception as e:
                    print(f"Error on {item['filename']}: {e}")
                    probs_list.append(0.5)

            probs = np.array(probs_list)
            torch.save(probs, cond_cache)

        metrics = evaluate_metrics(y_true, probs, threshold=FROZEN_DEV_THRESHOLD)
        spoof_mask = (y_true == 1)
        det_rate = np.mean(probs[spoof_mask] >= FROZEN_DEV_THRESHOLD) if np.sum(spoof_mask) > 0 else 0.0

        row = {
            "Condition": cond_name,
            "EER": float(metrics["eer"]),
            "AUC": float(metrics["roc_auc"]),
            "Detection_Rate_at_Frozen": float(det_rate),
            "Accuracy_at_Frozen": float(metrics["accuracy"]),
            "Balanced_Accuracy": float(metrics["balanced_accuracy"]),
            "FAR_at_Frozen": float(metrics["far"]),
            "FRR_at_Frozen": float(metrics["frr"]),
        }
        results.append(row)
        print(f"  EER: {metrics['eer']*100:.2f}%, AUC: {metrics['roc_auc']:.4f}, Spoof Detection Rate: {det_rate*100:.2f}%, BAcc: {metrics['balanced_accuracy']*100:.2f}%")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "t6_robustness.csv", index=False)
    print(f"\nRobustness results saved to {RESULTS_DIR / 't6_robustness.csv'}")

    # Plot Degradation Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot of EER by condition
    ax1 = axes[0]
    ax1.barh(res_df["Condition"], res_df["EER"] * 100, color="#2b6cb0", edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Equal Error Rate (%) [Lower is better]", fontsize=11)
    ax1.set_title("EER Degradation Under Perturbations", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.invert_yaxis()

    # Bar plot of Detection Rate
    ax2 = axes[1]
    ax2.barh(res_df["Condition"], res_df["Detection_Rate_at_Frozen"] * 100, color="#c53030", edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Spoof Detection Rate (%) @ 0.6402 [Higher is better]", fontsize=11)
    ax2.set_title("Spoof Detection Rate Under Perturbations", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 105)
    ax2.grid(alpha=0.3)
    ax2.invert_yaxis()

    plt.tight_layout()
    plot_path = RESULTS_DIR / "t6_robustness.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Robustness plots saved to {plot_path}")

    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t6(quick=args.quick, full=args.full)
