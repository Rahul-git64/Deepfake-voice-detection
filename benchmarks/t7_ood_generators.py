import os
import sys
import time
import asyncio
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import edge_tts
from gtts import gTTS

from benchmarks.utils import (
    get_wav2vec2,
    get_classifier,
    preprocess_audio_bytes,
    extract_embedding_from_waveform,
    predict_from_embedding,
    FROZEN_DEV_THRESHOLD,
    OOD_SPOOF_DIR,
    OOD_REAL_DIR,
    RESULTS_DIR,
    DEV_DIR,
    TARGET_SR,
    RANDOM_SEED,
)

PHRASES = [
    "Your bank account has been compromised, please transfer your funds immediately to secure them.",
    "Hello grandma, this is your grandson calling, I am in trouble in another city and need bail money.",
    "This is Officer Miller from the Internal Revenue Service regarding an unpaid tax penalty.",
    "Please confirm your social security number and one-time password to restore online access.",
    "Good morning, I am calling from technical support regarding a virus detected on your personal computer.",
    "Your package delivery has been placed on hold due to missing address verification details.",
    "We noticed unusual transaction attempts on your credit card totaling over eight hundred dollars.",
    "This urgent notice is regarding the expiration of your automobile extended warranty coverage.",
    "Congratulations, you have been selected for an all-expenses-paid Caribbean cruise vacation.",
    "Please press one now to speak with a customer representative about lowering your interest rates.",
    "Hey there, could you wire me five hundred dollars right now? I will pay you back tomorrow morning.",
    "The weather today is cloudy with a high chance of intermittent thunderstorms in the afternoon.",
    "Artificial intelligence models can generate synthetic speech with remarkable acoustic fidelity.",
    "Please verify your identity by speaking your full date of birth after the tone.",
    "Our records indicate an urgent outstanding balance on your electric and utility invoice.",
    "Emergency response teams have arrived at the central district hospital following the tremor.",
    "The quick brown fox jumps over the lazy sleeping golden retriever in the autumn sun.",
    "Security alerts should never be ignored when personal information is potentially exposed.",
    "Financial institutions will never request your secret PIN code over an unsolicited telephone call.",
    "Machine learning classifiers must generalize to unseen accents, acoustic environments, and vocoders.",
]

EDGE_VOICES = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-IN-NeerjaNeural",
    "en-IN-PrabhatNeural",
]

async def generate_edge_tts(text: str, voice: str, out_path: Path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))

def generate_gtts(text: str, out_path: Path):
    tts = gTTS(text=text, lang="en")
    tts.save(str(out_path))

def run_t7(quick: bool = False, full: bool = False):
    print("=" * 60)
    print("Running T7: Out-of-Domain Modern Voice Generators")
    print("=" * 60)

    OOD_SPOOF_DIR.mkdir(parents=True, exist_ok=True)
    OOD_REAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize spoof clips if not already present
    # Target: at least 40 edge-tts + 20 gtts = 60 clips
    edge_count = 0
    gtts_count = 0

    print("Checking and generating modern TTS synthetic samples...")
    # Generate Edge-TTS
    phrases_to_use = PHRASES[:5] if quick else PHRASES
    for p_idx, text in enumerate(phrases_to_use):
        for v_idx, voice in enumerate(EDGE_VOICES):
            if quick and v_idx >= 2:
                continue
            fname = f"edge_{voice}_{p_idx:02d}.mp3"
            fpath = OOD_SPOOF_DIR / fname
            if not fpath.exists():
                try:
                    asyncio.run(generate_edge_tts(text, voice, fpath))
                except Exception as e:
                    print(f"Edge TTS error on {voice}: {e}")
            if fpath.exists():
                edge_count += 1

    # Generate gTTS
    for p_idx, text in enumerate(phrases_to_use):
        fname = f"gtts_en_{p_idx:02d}.mp3"
        fpath = OOD_SPOOF_DIR / fname
        if not fpath.exists():
            try:
                generate_gtts(text, fpath)
            except Exception as e:
                print(f"gTTS error on phrase {p_idx}: {e}")
        if fpath.exists():
            gtts_count += 1

    print(f"Total OOD Spoof clips available: {edge_count} Edge-TTS, {gtts_count} gTTS (Total: {edge_count + gtts_count})")

    # 2. Check for real audio in ood_real/
    real_files = list(OOD_REAL_DIR.glob("*.wav")) + list(OOD_REAL_DIR.glob("*.mp3")) + list(OOD_REAL_DIR.glob("*.flac"))
    real_clips = []
    used_dev_fallback = False

    if len(real_files) >= 10 and not quick:
        print(f"Found {len(real_files)} user-provided real voice recordings in {OOD_REAL_DIR}")
        for rf in real_files:
            real_clips.append((rf.name, rf, "user_ood_real"))
    else:
        print(f"Notice: User provided {len(real_files)} clips in {OOD_REAL_DIR}. Falling back to 40 dev bona fide clips.")
        used_dev_fallback = True
        from benchmarks.utils import get_dev_protocol_df
        dev_df = get_dev_protocol_df()
        bona_sample = dev_df[dev_df["label"] == "bonafide"].sample(n=10 if quick else 40, random_state=RANDOM_SEED)
        for _, row in bona_sample.iterrows():
            fn = row["filename"]
            flac_path = DEV_DIR / f"{fn}.flac"
            if flac_path.exists():
                real_clips.append((fn, flac_path, "dev_bonafide_in_domain_baseline"))

    # 3. Feature extraction & prediction
    wav2vec2 = get_wav2vec2()
    classifier = get_classifier()

    spoof_records = []
    spoof_files = sorted(list(OOD_SPOOF_DIR.glob("*.mp3")) + list(OOD_SPOOF_DIR.glob("*.wav")))
    for sf_path in spoof_files:
        try:
            with open(sf_path, "rb") as f:
                raw_bytes = f.read()
            waveform = preprocess_audio_bytes(raw_bytes)
            emb = extract_embedding_from_waveform(waveform, wav2vec2_model=wav2vec2)
            probs, spoof_prob = predict_from_embedding(emb, classifier=classifier)
            gen = "edge_tts" if "edge_" in sf_path.name else ("gtts" if "gtts_" in sf_path.name else "other")
            spoof_records.append({
                "filename": sf_path.name,
                "label": 1,
                "generator": gen,
                "spoof_prob": spoof_prob,
                "bonafide_prob": float(probs[0]),
                "pred_at_frozen": 1 if spoof_prob >= FROZEN_DEV_THRESHOLD else 0,
                "pred_at_05": 1 if spoof_prob >= 0.5 else 0,
            })
        except Exception as e:
            print(f"Error predicting spoof {sf_path.name}: {e}")

    real_records = []
    for fn, rf_path, source in real_clips:
        try:
            with open(rf_path, "rb") as f:
                raw_bytes = f.read()
            waveform = preprocess_audio_bytes(raw_bytes)
            emb = extract_embedding_from_waveform(waveform, wav2vec2_model=wav2vec2)
            probs, spoof_prob = predict_from_embedding(emb, classifier=classifier)
            real_records.append({
                "filename": fn,
                "label": 0,
                "source": source,
                "spoof_prob": spoof_prob,
                "bonafide_prob": float(probs[0]),
                "pred_at_frozen": 1 if spoof_prob >= FROZEN_DEV_THRESHOLD else 0,
                "pred_at_05": 1 if spoof_prob >= 0.5 else 0,
            })
        except Exception as e:
            print(f"Error predicting real {fn}: {e}")

    df_spoofs = pd.DataFrame(spoof_records)
    df_reals = pd.DataFrame(real_records)

    # Calculate Detection Metrics
    edge_mask = df_spoofs["generator"] == "edge_tts"
    gtts_mask = df_spoofs["generator"] == "gtts"

    edge_det_frozen = df_spoofs.loc[edge_mask, "pred_at_frozen"].mean() if sum(edge_mask) > 0 else 0.0
    gtts_det_frozen = df_spoofs.loc[gtts_mask, "pred_at_frozen"].mean() if sum(gtts_mask) > 0 else 0.0
    total_spoof_det_frozen = df_spoofs["pred_at_frozen"].mean()

    edge_det_05 = df_spoofs.loc[edge_mask, "pred_at_05"].mean() if sum(edge_mask) > 0 else 0.0
    gtts_det_05 = df_spoofs.loc[gtts_mask, "pred_at_05"].mean() if sum(gtts_mask) > 0 else 0.0
    total_spoof_det_05 = df_spoofs["pred_at_05"].mean()

    real_frr_frozen = df_reals["pred_at_frozen"].mean()  # False alarm (human flagged as spoof)
    real_frr_05 = df_reals["pred_at_05"].mean()

    print("\n--- T7 Modern OOD Voice Results ---")
    print(f"Edge-TTS Detection Rate:   {edge_det_frozen*100:.2f}% (at 0.6402) | {edge_det_05*100:.2f}% (at 0.5)")
    print(f"gTTS Detection Rate:       {gtts_det_frozen*100:.2f}% (at 0.6402) | {gtts_det_05*100:.2f}% (at 0.5)")
    print(f"Overall Spoof Detection:   {total_spoof_det_frozen*100:.2f}% (at 0.6402)")
    print(f"Real Human False Alarm:    {real_frr_frozen*100:.2f}% ({'Dev In-Domain Fallback' if used_dev_fallback else 'User OOD Real'})")

    all_records = pd.concat([df_spoofs, df_reals], ignore_index=True)
    all_records.to_csv(RESULTS_DIR / "t7_ood_generators.csv", index=False)
    print(f"Detailed predictions saved to {RESULTS_DIR / 't7_ood_generators.csv'}")

    # Plot Score Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 25)

    if sum(edge_mask) > 0:
        ax.hist(df_spoofs.loc[edge_mask, "spoof_prob"], bins=bins, alpha=0.6, label=f"Edge-TTS (n={sum(edge_mask)})", color="#3182ce")
    if sum(gtts_mask) > 0:
        ax.hist(df_spoofs.loc[gtts_mask, "spoof_prob"], bins=bins, alpha=0.6, label=f"gTTS (n={sum(gtts_mask)})", color="#805ad5")
    ax.hist(df_reals["spoof_prob"], bins=bins, alpha=0.6, label=f"Real Audio ({'Dev Baseline' if used_dev_fallback else 'OOD'}) (n={len(df_reals)})", color="#38a169")

    ax.axvline(FROZEN_DEV_THRESHOLD, color="#e53e3e", linestyle="--", lw=2, label=f"Frozen Threshold ({FROZEN_DEV_THRESHOLD})")
    ax.set_xlabel("Predicted Spoof Probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Score Distribution on Modern Out-of-Domain Speech", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "t7_ood_generators.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"OOD score histogram saved to {plot_path}")

    return {
        "edge_tts_detection_rate": float(edge_det_frozen),
        "gtts_detection_rate": float(gtts_det_frozen),
        "overall_spoof_detection_rate": float(total_spoof_det_frozen),
        "real_false_alarm_rate": float(real_frr_frozen),
        "used_dev_fallback": used_dev_fallback,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run_t7(quick=args.quick, full=args.full)
