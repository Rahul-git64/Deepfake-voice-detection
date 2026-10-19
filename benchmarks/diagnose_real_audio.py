import os
import sys
import io
import time
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torchaudio
import requests
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from benchmarks.utils import (
    get_wav2vec2,
    get_classifier,
    extract_embedding_from_waveform,
    predict_from_embedding,
    FROZEN_DEV_THRESHOLD,
    TARGET_SR,
    RESULTS_DIR,
    DEV_DIR,
    RANDOM_SEED,
)

RECORDINGS_DIR = Path("benchmarks/data/my_recordings")
API_PREDICT_URL = "http://127.0.0.1:8080/predict"

def decode_audio_file(file_path: Path) -> Tuple[np.ndarray, int, int]:
    """
    Decodes audio using soundfile, or falls back to ffmpeg if soundfile fails (e.g. webm/opus, m4a).
    Returns: (audio_mono_float, original_sr, original_channels)
    """
    try:
        data, sr = sf.read(str(file_path))
        orig_channels = 1 if data.ndim == 1 else data.shape[-1]
        if data.ndim > 1:
            data = data.mean(axis=-1)
        return data.astype(np.float32), sr, orig_channels
    except Exception:
        # Fall back to ffmpeg subprocess decode to 16kHz mono WAV in memory
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
        try:
            cmd = ["ffmpeg", "-y", "-i", str(file_path), "-ar", "16000", "-ac", "1", tmp_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            data, sr = sf.read(tmp_path)
            return data.astype(np.float32), sr, 1
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

def compute_acoustic_descriptors(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Computes physical and spectral descriptors for a recording."""
    duration = len(y) / sr
    if duration <= 0:
        return {"duration": 0, "rms": 0, "peak": 0, "noise_floor": 0, "bandwidth_rolloff": 0, "silence_share": 1.0}

    # RMS & Peak
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))

    # Frame-wise energy for silence share & noise floor
    frame_len = int(0.025 * sr)  # 25ms
    hop_len = int(0.010 * sr)    # 10ms
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))

    # Noise floor: estimate from bottom 10th percentile energy
    noise_floor = float(np.percentile(frame_rms, 10)) if len(frame_rms) > 0 else 0.0

    # Silence share: proportion of frames below -40 dB relative to peak
    silence_thresh = max(1e-5, peak * 0.01)  # -40 dB
    silence_share = float(np.mean(frame_rms < silence_thresh)) if len(frame_rms) > 0 else 0.0

    # 99% Spectral rolloff (effective upper bandwidth limit)
    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99, hop_length=hop_len)
        bandwidth_rolloff = float(np.median(rolloff))
    except Exception:
        bandwidth_rolloff = float(sr / 2.0)

    return {
        "duration": float(duration),
        "rms": float(rms),
        "peak": float(peak),
        "noise_floor": float(noise_floor),
        "bandwidth_rolloff": float(bandwidth_rolloff),
        "silence_share": float(silence_share),
    }

def run_diagnostic_variations(y: np.ndarray, sr: int, wav2vec2, classifier) -> Dict[str, float]:
    """
    Computes spoof probabilities across experimental pipeline variations:
    (b) in-process raw (standard 16kHz mono)
    (c) after peak/loudness normalization (-1 dB peak)
    (d) after top-and-tail silence trimming (top_db=30)
    (e) after bandlimiting to 4 kHz (simulating telephony cutoff)
    """
    res = {}

    # Resample to 16kHz if not already
    if sr != TARGET_SR:
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    else:
        y_16k = y.copy()

    # (b) In-process standard
    wav_tensor = torch.from_numpy(y_16k).float()
    emb = extract_embedding_from_waveform(wav_tensor, wav2vec2_model=wav2vec2)
    _, p_b = predict_from_embedding(emb, classifier=classifier)
    res["prob_in_process"] = float(p_b)

    # (c) Peak normalization to 0.89 (-1 dB)
    peak = np.max(np.abs(y_16k))
    if peak > 1e-6:
        y_norm = (y_16k / peak) * 0.89
    else:
        y_norm = y_16k
    emb_c = extract_embedding_from_waveform(torch.from_numpy(y_norm).float(), wav2vec2_model=wav2vec2)
    _, p_c = predict_from_embedding(emb_c, classifier=classifier)
    res["prob_peak_normalized"] = float(p_c)

    # (d) Silence trimmed (librosa trim)
    y_trimmed, _ = librosa.effects.trim(y_16k, top_db=30)
    if len(y_trimmed) < int(0.5 * TARGET_SR):
        y_trimmed = y_16k  # prevent collapsing to 0
    emb_d = extract_embedding_from_waveform(torch.from_numpy(y_trimmed).float(), wav2vec2_model=wav2vec2)
    _, p_d = predict_from_embedding(emb_d, classifier=classifier)
    res["prob_silence_trimmed"] = float(p_d)

    # (e) Bandlimited to 4 kHz (lowpass butterworth filter at 4000 Hz)
    from scipy.signal import butter, sosfilt
    sos = butter(N=6, Wn=4000, fs=TARGET_SR, btype="lowpass", output="sos")
    y_4k = sosfilt(sos, y_16k).astype(np.float32)
    emb_e = extract_embedding_from_waveform(torch.from_numpy(y_4k).float(), wav2vec2_model=wav2vec2)
    _, p_e = predict_from_embedding(emb_e, classifier=classifier)
    res["prob_bandlimited_4khz"] = float(p_e)

    return res

def query_live_api(file_path: Path) -> Tuple[float, str]:
    """Queries the running FastAPI /predict endpoint."""
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f.read(), "audio/wav")}
        resp = requests.post(API_PREDICT_URL, files=files, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return float(data["probabilities"]["spoof"]), data["prediction"]
    except Exception:
        pass
    return float("nan"), "api_unreachable"

def run_diagnose():
    print("=" * 70)
    print("PHASE 0: REAL AUDIO PIPELINE & ACOUSTIC DIAGNOSTICS")
    print("=" * 70)

    # Find audio files in my_recordings
    exts = ["*.wav", "*.flac", "*.mp3", "*.m4a", "*.webm", "*.ogg"]
    audio_files = []
    for ext in exts:
        audio_files.extend(list(RECORDINGS_DIR.glob(ext)))

    wav2vec2 = get_wav2vec2()
    classifier = get_classifier()

    # If user hasn't added recordings yet, create a set of diagnostic sample clips
    if len(audio_files) == 0:
        print(f"Notice: No user audio files currently found in {RECORDINGS_DIR}.")
        print("Generating a diagnostic evaluation set (30 varied bona fide human clips with phone, laptop, and noise perturbations)...")
        from benchmarks.utils import get_dev_protocol_df
        dev_df = get_dev_protocol_df()
        bona_sample = dev_df[dev_df["label"] == "bonafide"].sample(n=30, random_state=RANDOM_SEED)

        for idx, (_, row) in enumerate(bona_sample.iterrows(), 1):
            fn = row["filename"]
            flac_path = DEV_DIR / f"{fn}.flac"
            if not flac_path.exists():
                continue
            data, sr = sf.read(str(flac_path))

            # Simulate various capture profiles:
            if idx <= 8:
                # Smartphone profile: slight ambient noise
                noise = np.random.normal(0, 0.005, len(data))
                sim_data = (data + noise).astype(np.float32)
                out_name = f"sim_phone_mic_{idx:02d}.wav"
            elif idx <= 16:
                # Laptop profile: laptop mic fan rumble (low-frequency noise + unnormalized gain)
                rumble = np.random.normal(0, 0.02, len(data))
                sim_data = (data * 0.4 + rumble).astype(np.float32)
                out_name = f"sim_laptop_mic_{idx:02d}.wav"
            elif idx <= 24:
                # Real phone call simulation: 8 kHz downsample + bandpass
                from benchmarks.t6_robustness import apply_telephone_simulation
                sim_data = apply_telephone_simulation(data, sr=sr)
                out_name = f"sim_phonecall_pstn_{idx:02d}.wav"
            else:
                # Browser recorded: unpadded, short, dynamic gain
                sim_data = (data * 1.5).clip(-1.0, 1.0).astype(np.float32)
                out_name = f"sim_browser_capture_{idx:02d}.wav"

            out_path = RECORDINGS_DIR / out_name
            sf.write(str(out_path), sim_data, sr)
            audio_files.append(out_path)

        print(f"Created {len(audio_files)} diagnostic audio captures in {RECORDINGS_DIR}")

    print(f"\nProcessing {len(audio_files)} audio clips for acoustic and pipeline diagnosis...")
    records = []

    for fpath in audio_files:
        try:
            raw_audio, orig_sr, orig_channels = decode_audio_file(fpath)
            desc = compute_acoustic_descriptors(raw_audio, orig_sr)
            var_probs = run_diagnostic_variations(raw_audio, orig_sr, wav2vec2, classifier)
            api_prob, api_pred = query_live_api(fpath)

            records.append({
                "filename": fpath.name,
                "orig_sr": orig_sr,
                "orig_channels": orig_channels,
                "duration_s": desc["duration"],
                "rms": desc["rms"],
                "peak": desc["peak"],
                "noise_floor": desc["noise_floor"],
                "bandwidth_rolloff_hz": desc["bandwidth_rolloff"],
                "silence_share": desc["silence_share"],
                "prob_live_api": api_prob,
                "api_prediction": api_pred,
                "prob_in_process": var_probs["prob_in_process"],
                "prob_peak_normalized": var_probs["prob_peak_normalized"],
                "prob_silence_trimmed": var_probs["prob_silence_trimmed"],
                "prob_bandlimited_4khz": var_probs["prob_bandlimited_4khz"],
                "flagged_as_ai": var_probs["prob_in_process"] >= FROZEN_DEV_THRESHOLD,
            })
        except Exception as e:
            print(f"Error analyzing {fpath.name}: {e}")

    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "diagnose_real_audio.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nDiagnostic results saved to {out_csv}")

    # Correlations between acoustic descriptors and spoof probability
    print("\n--- Correlation: Acoustic Descriptors vs. In-Process Spoof Probability ---")
    corr_rows = []
    for col in ["duration_s", "rms", "peak", "noise_floor", "bandwidth_rolloff_hz", "silence_share"]:
        vals = df[col].values
        probs = df["prob_in_process"].values
        p_corr, p_val = pearsonr(vals, probs)
        s_corr, s_val = spearmanr(vals, probs)
        corr_rows.append({
            "Descriptor": col,
            "Pearson_r": round(float(p_corr), 4),
            "Pearson_pval": round(float(p_val), 4),
            "Spearman_rho": round(float(s_corr), 4),
            "Spearman_pval": round(float(s_val), 4),
        })
    corr_df = pd.DataFrame(corr_rows)
    print(corr_df.to_string(index=False))

    # Mean spoof probability before and after each intervention
    print("\n--- Impact of Diagnostic Audio Preprocessing Interventions ---")
    interventions = {
        "Raw In-Process (Current Pipeline)": df["prob_in_process"].mean(),
        "After Peak Normalization (-1 dB)": df["prob_peak_normalized"].mean(),
        "After Silence Trimming (30 dB)": df["prob_silence_trimmed"].mean(),
        "After 4 kHz Bandlimiting": df["prob_bandlimited_4khz"].mean(),
    }
    for k, v in interventions.items():
        print(f"  {k:<35}: Mean Spoof Prob = {v:.4f}")

    # Plot Scatter plots: Spoof probability vs each descriptor
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    descriptors = [
        ("duration_s", "Duration (seconds)"),
        ("rms", "RMS Energy"),
        ("peak", "Peak Amplitude"),
        ("noise_floor", "Noise Floor Energy"),
        ("bandwidth_rolloff_hz", "99% Bandwidth Rolloff (Hz)"),
        ("silence_share", "Silence Share Proportion"),
    ]

    for ax, (col, label) in zip(axes, descriptors):
        ax.scatter(df[col], df["prob_in_process"], color="#2b6cb0", alpha=0.7, edgecolors="none")
        ax.axhline(FROZEN_DEV_THRESHOLD, color="#e53e3e", linestyle="--", label=f"Threshold ({FROZEN_DEV_THRESHOLD})")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Predicted Spoof Prob", fontsize=10)
        ax.set_title(f"Spoof Prob vs {col}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out_plot = RESULTS_DIR / "diagnose_real_audio.png"
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"Diagnostic scatter plot saved to {out_plot}")

    return df, corr_df

if __name__ == "__main__":
    run_diagnose()
