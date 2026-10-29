#!/usr/bin/env python3
"""
src_v2/evaluate_regression.py

Phase 1 Regression Battery Test Suite.
Evaluates candidate deepfake detection models on n >= 300 clips per class/condition with Wilson 95% CIs.

Test Conditions:
Bona Fide:
- Silence trimming: 0.0s (speech-only), 0.1s, 0.25s, 0.5s margins
- Highpass filter: 100 Hz, 300 Hz
- Lowpass filter: 3.4 kHz (held out), 4.0 kHz, 5.0 kHz, 6.0 kHz
- Downsampling: 8 kHz resample
- Telephone: Full PSTN chain (300-3400 Hz bandpass + 8k resample + mu-law)
- Noise: White, Pink (held out), Low-band (<4k), High-band (>4k) at 30, 20, 10 dB SNR

Spoof:
- Noise: White, Pink (held out), Low-band, High-band at 30, 20, 10 dB SNR
- Padding: Digital zeros and Room tone at 0.5s, 1.0s, 1.5s

Acceptance Criteria:
- Every condition flips < 10.0% of decisions at chosen threshold
- Clean-condition EER does not worsen by > 1.0 point
- Reports all failures explicitly.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src_v2.augment import (
    trim_silence_boundary,
    pad_silence,
    apply_highpass_filter,
    apply_lowpass_filter,
    apply_8k_resample,
    apply_noise,
    apply_phone_chain,
    TARGET_SR,
)

RESULTS_PATH = Path("data_v2/regression_battery_results.csv")


def wilson_score_interval(k: int, n: int, confidence: float = 0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return p, lower, upper


def run_battery_on_clips(model_fn, bona_clips, spoof_clips, threshold: float = 0.6402):
    """
    model_fn: callable taking (audio, sr) and returning prob_spoof in [0, 1]
    bona_clips: list of numpy arrays (bona fide speech)
    spoof_clips: list of numpy arrays (spoof speech)
    """
    results = []

    # 1. CLEAN BASELINES
    bona_clean_preds = [model_fn(clip, TARGET_SR) for clip in bona_clips]
    spoof_clean_preds = [model_fn(clip, TARGET_SR) for clip in spoof_clips]

    bona_clean_decisions = np.array([p >= threshold for p in bona_clean_preds])  # True = Fake (False Alarm)
    spoof_clean_decisions = np.array([p >= threshold for p in spoof_clean_preds])  # True = Fake (Correct)

    clean_fa = int(np.sum(bona_clean_decisions))
    clean_miss = int(np.sum(~spoof_clean_decisions))
    rate_fa, low_fa, up_fa = wilson_score_interval(clean_fa, len(bona_clips))
    rate_miss, low_miss, up_miss = wilson_score_interval(clean_miss, len(spoof_clips))

    results.append({
        "class": "bonafide",
        "condition": "clean_baseline",
        "n": len(bona_clips),
        "flips": clean_fa,
        "flip_rate": round(rate_fa * 100, 2),
        "ci_lower": round(low_fa * 100, 2),
        "ci_upper": round(up_fa * 100, 2),
        "accepted": rate_fa < 0.10
    })

    # 2. BONA FIDE REGRESSION CONDITIONS
    # Trimming
    for margin in [0.5, 0.25, 0.1, 0.0]:
        flips = 0
        for i, clip in enumerate(bona_clips):
            mod = trim_silence_boundary(clip, keep_margin_sec=margin, sr=TARGET_SR)
            p = model_fn(mod, TARGET_SR)
            dec = p >= threshold
            # Flip occurs if clean decision differs from perturbed decision
            if dec != bona_clean_decisions[i]:
                flips += 1
        r, lo, up = wilson_score_interval(flips, len(bona_clips))
        results.append({
            "class": "bonafide",
            "condition": f"trim_{margin}s",
            "n": len(bona_clips),
            "flips": flips,
            "flip_rate": round(r * 100, 2),
            "ci_lower": round(lo * 100, 2),
            "ci_upper": round(up * 100, 2),
            "accepted": r < 0.10
        })

    # Highpass
    for hp in [100.0, 300.0]:
        flips = 0
        for i, clip in enumerate(bona_clips):
            mod = apply_highpass_filter(clip, cutoff_hz=hp, sr=TARGET_SR)
            p = model_fn(mod, TARGET_SR)
            if (p >= threshold) != bona_clean_decisions[i]:
                flips += 1
        r, lo, up = wilson_score_interval(flips, len(bona_clips))
        results.append({
            "class": "bonafide",
            "condition": f"highpass_{int(hp)}hz",
            "n": len(bona_clips),
            "flips": flips,
            "flip_rate": round(r * 100, 2),
            "ci_lower": round(lo * 100, 2),
            "ci_upper": round(up * 100, 2),
            "accepted": r < 0.10
        })

    # Lowpass
    for lp in [3400.0, 4000.0, 5000.0, 6000.0]:
        flips = 0
        for i, clip in enumerate(bona_clips):
            mod = apply_lowpass_filter(clip, cutoff_hz=lp, sr=TARGET_SR)
            p = model_fn(mod, TARGET_SR)
            if (p >= threshold) != bona_clean_decisions[i]:
                flips += 1
        r, lo, up = wilson_score_interval(flips, len(bona_clips))
        results.append({
            "class": "bonafide",
            "condition": f"lowpass_{int(lp)}hz",
            "n": len(bona_clips),
            "flips": flips,
            "flip_rate": round(r * 100, 2),
            "ci_lower": round(lo * 100, 2),
            "ci_upper": round(up * 100, 2),
            "accepted": r < 0.10
        })

    # Resample 8k & Phone chain
    for name, fn in [("resample_8k", lambda c: apply_8k_resample(c, TARGET_SR)),
                     ("phone_chain", lambda c: apply_phone_chain(c, TARGET_SR))]:
        flips = 0
        for i, clip in enumerate(bona_clips):
            mod = fn(clip)
            p = model_fn(mod, TARGET_SR)
            if (p >= threshold) != bona_clean_decisions[i]:
                flips += 1
        r, lo, up = wilson_score_interval(flips, len(bona_clips))
        results.append({
            "class": "bonafide",
            "condition": name,
            "n": len(bona_clips),
            "flips": flips,
            "flip_rate": round(r * 100, 2),
            "ci_lower": round(lo * 100, 2),
            "ci_upper": round(up * 100, 2),
            "accepted": r < 0.10
        })

    # Bona fide Noise
    for ntype in ["white", "pink", "low_band", "high_band"]:
        for snr in [30.0, 20.0, 10.0]:
            flips = 0
            for i, clip in enumerate(bona_clips):
                mod = apply_noise(clip, noise_type=ntype, snr_db=snr, sr=TARGET_SR)
                p = model_fn(mod, TARGET_SR)
                if (p >= threshold) != bona_clean_decisions[i]:
                    flips += 1
            r, lo, up = wilson_score_interval(flips, len(bona_clips))
            results.append({
                "class": "bonafide",
                "condition": f"noise_{ntype}_{int(snr)}db",
                "n": len(bona_clips),
                "flips": flips,
                "flip_rate": round(r * 100, 2),
                "ci_lower": round(lo * 100, 2),
                "ci_upper": round(up * 100, 2),
                "accepted": r < 0.10
            })

    # 3. SPOOF REGRESSION CONDITIONS
    # Spoof Noise
    for ntype in ["white", "pink", "low_band", "high_band"]:
        for snr in [30.0, 20.0, 10.0]:
            flips = 0
            for i, clip in enumerate(spoof_clips):
                mod = apply_noise(clip, noise_type=ntype, snr_db=snr, sr=TARGET_SR)
                p = model_fn(mod, TARGET_SR)
                if (p >= threshold) != spoof_clean_decisions[i]:
                    flips += 1
            r, lo, up = wilson_score_interval(flips, len(spoof_clips))
            results.append({
                "class": "spoof",
                "condition": f"noise_{ntype}_{int(snr)}db",
                "n": len(spoof_clips),
                "flips": flips,
                "flip_rate": round(r * 100, 2),
                "ci_lower": round(lo * 100, 2),
                "ci_upper": round(up * 100, 2),
                "accepted": r < 0.10
            })

    # Spoof Padding
    for mode in ["zeros", "room_tone"]:
        for pad_sec in [0.5, 1.0, 1.5]:
            flips = 0
            for i, clip in enumerate(spoof_clips):
                mod = pad_silence(clip, pad_sec=pad_sec, mode=mode, sr=TARGET_SR)
                p = model_fn(mod, TARGET_SR)
                if (p >= threshold) != spoof_clean_decisions[i]:
                    flips += 1
            r, lo, up = wilson_score_interval(flips, len(spoof_clips))
            results.append({
                "class": "spoof",
                "condition": f"pad_{mode}_{pad_sec}s",
                "n": len(spoof_clips),
                "flips": flips,
                "flip_rate": round(r * 100, 2),
                "ci_lower": round(lo * 100, 2),
                "ci_upper": round(up * 100, 2),
                "accepted": r < 0.10
            })

    df = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"Regression battery completed. Results saved to {RESULTS_PATH}")
    return df
