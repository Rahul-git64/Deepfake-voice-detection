import io
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import requests

from benchmarks.utils import (
    get_wav2vec2,
    get_classifier,
    extract_embedding_from_waveform,
    predict_from_embedding,
    FROZEN_DEV_THRESHOLD,
    TARGET_SR,
    RESULTS_DIR,
)

API_URL = "http://127.0.0.1:8080/predict"

def generate_signals(duration_sec: float = 3.0, sr: int = TARGET_SR):
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    signals = {}

    # 1. Complete silence
    signals["Pure Silence"] = np.zeros(n_samples, dtype=np.float32)

    # 2. Pure sine wave 440 Hz (Concert A)
    signals["Pure Sine 440 Hz"] = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # 3. Pure sine wave 1000 Hz
    signals["Pure Sine 1000 Hz"] = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    # 4. White noise
    signals["White Noise"] = np.random.normal(0, 0.2, n_samples).astype(np.float32)

    # 5. Pink noise (1/f noise)
    uneven = np.random.randn(n_samples)
    fft_vals = np.fft.rfft(uneven)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0  # avoid division by zero
    fft_pink = fft_vals / np.sqrt(freqs)
    pink_signal = np.fft.irfft(fft_pink, n=n_samples)
    pink_signal = pink_signal / np.max(np.abs(pink_signal)) * 0.4
    signals["Pink Noise"] = pink_signal.astype(np.float32)

    # 6. Musical chord sequence (C major chord: C4 261.63, E4 329.63, G4 392.00)
    chord = (
        0.2 * np.sin(2 * np.pi * 261.63 * t) +
        0.2 * np.sin(2 * np.pi * 329.63 * t) +
        0.2 * np.sin(2 * np.pi * 392.00 * t)
    )
    signals["Musical Chord (C Major)"] = chord.astype(np.float32)

    return signals

def run_t8(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T8: Non-Speech Sanity Tests (Model & API)")
    print("=" * 60)

    signals = generate_signals(duration_sec=3.0, sr=TARGET_SR)
    wav2vec2 = get_wav2vec2()
    classifier = get_classifier()

    # Check API health first
    api_available = False
    try:
        h_resp = requests.get("http://127.0.0.1:8080/health", timeout=3)
        if h_resp.status_code == 200:
            api_available = True
            print("FastAPI backend at http://127.0.0.1:8080 is HEALTHY.")
    except Exception:
        print("FastAPI backend not reachable at http://127.0.0.1:8080.")

    results = []

    for name, sig in signals.items():
        # Encode to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, sig, TARGET_SR, format="WAV")
        raw_bytes = buf.getvalue()

        # In-process Model evaluation
        waveform = torch.from_numpy(sig).float()
        emb = extract_embedding_from_waveform(waveform, wav2vec2_model=wav2vec2)
        probs, spoof_prob = predict_from_embedding(emb, classifier=classifier)

        model_pred_label = "spoof" if spoof_prob >= FROZEN_DEV_THRESHOLD else "bonafide"

        # API evaluation if active
        api_pred_label = "N/A"
        api_spoof_prob = float("nan")
        api_conf = float("nan")
        if api_available:
            try:
                files = {"file": (f"{name.replace(' ', '_')}.wav", raw_bytes, "audio/wav")}
                resp = requests.post(API_URL, files=files, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    api_pred_label = data["prediction"]
                    api_spoof_prob = data["probabilities"]["spoof"]
                    api_conf = data["confidence"]
            except Exception as e:
                api_pred_label = f"Error: {e}"

        row = {
            "Signal": name,
            "Model_Spoof_Prob": round(float(spoof_prob), 4),
            "Model_Bonafide_Prob": round(float(probs[0]), 4),
            "Model_Prediction_at_Frozen": model_pred_label,
            "API_Prediction": api_pred_label,
            "API_Spoof_Prob": api_spoof_prob,
        }
        results.append(row)
        print(f"Signal: {name:<25} | Model Spoof Prob: {spoof_prob:.4f} -> {model_pred_label:<8} | API: {api_pred_label}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "t8_non_speech_sanity.csv", index=False)
    print(f"\nNon-speech sanity results saved to {RESULTS_DIR / 't8_non_speech_sanity.csv'}")

    return res_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t8(quick=args.quick, full=args.full)
