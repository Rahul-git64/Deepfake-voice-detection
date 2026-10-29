#!/usr/bin/env python3
"""
src_v2/build_manifest.py

Builds data_v2/manifest.csv cataloging all audio assets across splits and sources:
Columns: path, label, source, speaker, generator, text, language, duration, split

Sources indexed:
1. ASVspoof 2019 LA:
   - Train (25,380 clips): split = train
   - Dev (24,844 clips): split = dev
   - Eval (71,237 clips): split = heldout_eval
2. Local Multi-Accent TTS (data_v2/tts/):
   - Edge TTS (en-US, en-GB): split = train
   - Edge TTS (en-IN, hi-IN, te-IN), gTTS: split = dev
   - Edge TTS (en-US-GuyNeural): split = heldout_eval (strictly held-out generator)
3. Simulated Real Microphone/Browser Audio (benchmarks/data/my_recordings/):
   - split = heldout_eval
4. Out-of-Domain Spoof Audio (benchmarks/data/ood_spoof/):
   - split = heldout_eval
"""

import os
import struct
from pathlib import Path
import pandas as pd
import soundfile as sf

DATA_ROOT = Path(r"C:\Users\rahul\Downloads\AVSspoof_LA\LA")
OUTPUT_MANIFEST = Path("data_v2/manifest.csv")
TTS_METADATA = Path("data_v2/tts/tts_metadata.csv")
MY_RECORDINGS_DIR = Path("benchmarks/data/my_recordings")
OOD_SPOOF_DIR = Path("benchmarks/data/ood_spoof")


def get_flac_duration_fast(path: Path) -> float:
    """Read STREAMINFO from FLAC header in 36 bytes."""
    try:
        with open(path, "rb") as f:
            data = f.read(36)
            if data[:4] != b"fLaC":
                return 3.0
            sr = (data[18] << 12) | (data[19] << 4) | (data[20] >> 4)
            samples = ((data[21] & 0x0F) << 32) | (data[22] << 24) | (data[23] << 16) | (data[24] << 8) | data[25]
            return round(samples / sr, 2) if sr > 0 else 3.0
    except Exception:
        return 3.0


def build_manifest():
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1. ASVspoof 2019 LA Protocols
    protocol_configs = [
        {
            "split_name": "train",
            "protocol": DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
            "audio_rel_dir": "ASVspoof2019_LA_train/flac",
        },
        {
            "split_name": "dev",
            "protocol": DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            "audio_rel_dir": "ASVspoof2019_LA_dev/flac",
        },
        {
            "split_name": "heldout_eval",
            "protocol": DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
            "audio_rel_dir": "ASVspoof2019_LA_eval/flac",
        },
    ]

    for p_cfg in protocol_configs:
        split_name = p_cfg["split_name"]
        p_path = p_cfg["protocol"]
        rel_dir = p_cfg["audio_rel_dir"]

        if not p_path.exists():
            print(f"Warning: Protocol not found: {p_path}")
            continue

        print(f"Indexing ASVspoof protocol {p_path.name} -> {split_name} ...", flush=True)
        count = 0
        with open(p_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                spk = parts[0]
                fn = parts[1]
                sys_id = parts[3]
                key = parts[4]

                label = "bonafide" if key.lower() == "bonafide" else "spoof"
                generator = "human" if label == "bonafide" else sys_id
                rel_path = f"{rel_dir}/{fn}.flac"

                # Duration lookup (standard mean ~3.25s for ASVspoof, fast fallback)
                duration = 3.25

                rows.append({
                    "path": rel_path,
                    "label": label,
                    "source": "asvspoof2019_la",
                    "speaker": spk,
                    "generator": generator,
                    "text": "-",
                    "language": "en",
                    "duration": duration,
                    "split": split_name,
                })
                count += 1
        print(f"  Indexed {count} clips for {split_name}.")

    # 2. Local Multi-Accent TTS Clips
    if TTS_METADATA.exists():
        print(f"Indexing local TTS metadata from {TTS_METADATA} ...")
        df_tts = pd.read_csv(TTS_METADATA)
        for _, r in df_tts.iterrows():
            rows.append({
                "path": str(r["path"]).replace("\\", "/"),
                "label": "spoof",
                "source": r["source"],
                "speaker": r["speaker"],
                "generator": r["generator"],
                "text": str(r["text"]),
                "language": str(r["language"]),
                "duration": float(r["duration"]),
                "split": str(r["split"]),
            })
        print(f"  Indexed {len(df_tts)} TTS clips.")

    # 3. Simulated Real Microphone Audio (my_recordings)
    if MY_RECORDINGS_DIR.exists():
        my_files = [f for f in MY_RECORDINGS_DIR.glob("*.wav")]
        print(f"Indexing {len(my_files)} simulated real clips from {MY_RECORDINGS_DIR} ...")
        for f in my_files:
            try:
                info = sf.info(str(f))
                dur = round(info.duration, 2)
            except Exception:
                dur = 3.0

            rows.append({
                "path": str(f).replace("\\", "/"),
                "label": "bonafide",
                "source": "my_recordings",
                "speaker": "user_simulated",
                "generator": "human",
                "text": "-",
                "language": "en",
                "duration": dur,
                "split": "heldout_eval",
            })

    # 4. Out-of-Domain Spoof Audio
    if OOD_SPOOF_DIR.exists():
        ood_files = [f for f in OOD_SPOOF_DIR.glob("*.mp3")]
        print(f"Indexing {len(ood_files)} OOD spoof clips from {OOD_SPOOF_DIR} ...")
        for f in ood_files:
            try:
                info = sf.info(str(f))
                dur = round(info.duration, 2)
            except Exception:
                dur = 3.0

            gen_prefix = f.name.split("_")[0]
            rows.append({
                "path": str(f).replace("\\", "/"),
                "label": "spoof",
                "source": "ood_spoof",
                "speaker": f"ood_{gen_prefix}",
                "generator": gen_prefix,
                "text": "-",
                "language": "en",
                "duration": dur,
                "split": "heldout_eval",
            })

    df_manifest = pd.DataFrame(rows)
    df_manifest.to_csv(OUTPUT_MANIFEST, index=False)
    print(f"\nManifest built successfully: {OUTPUT_MANIFEST}")
    print(f"Total rows: {len(df_manifest)}")
    print(df_manifest.groupby(["source", "split", "label"]).size())
    return df_manifest


if __name__ == "__main__":
    build_manifest()
