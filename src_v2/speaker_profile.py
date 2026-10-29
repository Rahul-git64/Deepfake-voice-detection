#!/usr/bin/env python3
"""
src_v2/speaker_profile.py

Extracts acoustic profile (pitch F0, gender, loudness/RMS, duration, clip counts)
for all 40 speakers in the ASVspoof 2019 LA train and dev sets.
Saves results deterministically to data_v2/speaker_profiles.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

DATA_ROOT = Path(r"C:\Users\rahul\Downloads\AVSspoof_LA\LA")
OUTPUT_PATH = Path("data_v2/speaker_profiles.csv")

SPLITS = {
    "train": {
        "protocol": DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "audio_dir": DATA_ROOT / "ASVspoof2019_LA_train/flac",
    },
    "dev": {
        "protocol": DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_dir": DATA_ROOT / "ASVspoof2019_LA_dev/flac",
    },
}

KNOWN_DEV_FEMALE = {"LA_0069", "LA_0072", "LA_0074", "LA_0075", "LA_0077", "LA_0078"}
KNOWN_DEV_MALE = {"LA_0070", "LA_0071", "LA_0073", "LA_0076"}

def extract_speaker_profiles():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles = []

    for split_name, cfg in SPLITS.items():
        protocol_path = cfg["protocol"]
        audio_dir = cfg["audio_dir"]

        # Parse protocol
        speaker_clips = {}
        with open(protocol_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                spk, fn, _, _, key = parts[0], parts[1], parts[2], parts[3], parts[4]
                if spk not in speaker_clips:
                    speaker_clips[spk] = {"bonafide": [], "spoof": []}
                speaker_clips[spk][key].append(fn)

        print(f"[{split_name}] Found {len(speaker_clips)} speakers.")

        for spk in sorted(speaker_clips.keys()):
            bona_fns = speaker_clips[spk]["bonafide"]
            spoof_fns = speaker_clips[spk]["spoof"]

            # Analyze bona fide clips for intrinsic speaker properties (F0, loudness)
            f0_medians = []
            rms_list = []
            durations = []

            # Sample up to 10 bona fide clips for robust acoustic characterization
            sample_fns = bona_fns[:10] if len(bona_fns) >= 10 else bona_fns
            for fn in sample_fns:
                flac_path = audio_dir / f"{fn}.flac"
                if not flac_path.exists():
                    continue
                audio, sr = sf.read(str(flac_path))
                dur = len(audio) / sr
                durations.append(dur)

                # Loudness (RMS in dBFS)
                rms = np.sqrt(np.mean(audio ** 2))
                dbfs = 20 * np.log10(rms + 1e-9)
                rms_list.append(dbfs)

                # Pitch (F0 via yin)
                f0 = librosa.yin(audio, fmin=65, fmax=400, sr=sr)
                voiced = f0[(f0 >= 65) & (f0 <= 400)]
                if len(voiced) > 0:
                    f0_medians.append(float(np.median(voiced)))

            mean_f0 = float(np.mean(f0_medians)) if f0_medians else 0.0
            std_f0 = float(np.std(f0_medians)) if f0_medians else 0.0
            mean_loudness = float(np.mean(rms_list)) if rms_list else -99.0
            std_loudness = float(np.std(rms_list)) if rms_list else 0.0
            mean_dur = float(np.mean(durations)) if durations else 0.0

            # Determine gender: protocol ground truth if known, else acoustic F0 threshold (165 Hz)
            if spk in KNOWN_DEV_FEMALE:
                gender = "F"
            elif spk in KNOWN_DEV_MALE:
                gender = "M"
            else:
                gender = "F" if mean_f0 >= 165.0 else "M"

            profiles.append({
                "speaker_id": spk,
                "split": split_name,
                "gender": gender,
                "mean_f0_hz": round(mean_f0, 2),
                "std_f0_hz": round(std_f0, 2),
                "mean_loudness_dbfs": round(mean_loudness, 2),
                "std_loudness_dbfs": round(std_loudness, 2),
                "mean_duration_sec": round(mean_dur, 2),
                "n_bonafide": len(bona_fns),
                "n_spoof": len(spoof_fns),
                "n_total": len(bona_fns) + len(spoof_fns),
            })

    df = pd.DataFrame(profiles)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved speaker profiles to {OUTPUT_PATH} (n={len(df)} speakers)")
    return df

if __name__ == "__main__":
    df = extract_speaker_profiles()
    print(df.to_string())
