import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ---------------------------------------------------------------------------
# CONFIG — edit this to match your machine
# ---------------------------------------------------------------------------
try:
    import config
    DATA_ROOT = Path(getattr(config, "LA_DATA_ROOT", r"C:\Users\rahul\Downloads\AVSspoof_LA\LA"))
except ImportError:
    DATA_ROOT = Path(r"C:\Users\rahul\Downloads\AVSspoof_LA\LA")

OUTPUT_DIR = Path("data/embeddings")               # relative to repo root
MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000

SPLIT_CONFIG = {
    "train": {
        "audio_dir": "ASVspoof2019_LA_train/flac",
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    },
    "dev": {
        "audio_dir": "ASVspoof2019_LA_dev/flac",
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    },
    "eval": {
        "audio_dir": "ASVspoof2019_LA_eval/flac",
        "protocol": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
    },
}

LABEL_MAP = {"bonafide": 0, "spoof": 1}


def parse_protocol(protocol_path: Path):
    """
    Each line in a CM protocol file looks like:
        LA_0079 LA_T_1138215 - A01 spoof
        LA_0079 LA_T_1271820 - -   bonafide
    Columns: speaker_id, filename, -, attack_id, label
    Returns list of (filename, label_str)
    """
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            filename = parts[1]
            label = parts[-1]
            entries.append((filename, label))
    return entries


def load_audio(path: Path, target_sr: int = TARGET_SR):
    try:
        data, sr = sf.read(str(path))
        if data.ndim > 1:  # stereo -> mono
            data = np.mean(data, axis=1)
        waveform = torch.from_numpy(data).float().unsqueeze(0)
    except Exception:
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:  # stereo -> mono
            waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0)  # 1D tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "dev", "eval"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of files, for a quick test run")
    args = parser.parse_args()

    cfg = SPLIT_CONFIG[args.split]
    audio_dir = DATA_ROOT / cfg["audio_dir"]
    protocol_path = DATA_ROOT / cfg["protocol"]

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    entries = parse_protocol(protocol_path)
    if args.limit:
        entries = entries[: args.limit]
    print(f"[{args.split}] {len(entries)} entries found in protocol file", flush=True)

    print(f"Loading {MODEL_NAME} ...", flush=True)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(2)
    model.to(device)
    print(f"Using device: {device}", flush=True)

    embeddings = []
    labels = []
    filenames = []
    skipped = 0

    with torch.no_grad():
        for i, (filename, label_str) in enumerate(entries):
            flac_path = audio_dir / f"{filename}.flac"
            if not flac_path.exists():
                skipped += 1
                continue

            waveform = load_audio(flac_path)
            inputs = feature_extractor(
                waveform.numpy(), sampling_rate=TARGET_SR, return_tensors="pt"
            )
            input_values = inputs.input_values.to(device)

            outputs = model(input_values)
            # mean-pool over time to get a fixed-size vector
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()

            embeddings.append(embedding)
            labels.append(LABEL_MAP[label_str])
            filenames.append(filename)

            if (i + 1) % 50 == 0 or (i + 1) == len(entries):
                print(f"  processed {i + 1}/{len(entries)}", flush=True)

    if skipped:
        print(f"Warning: skipped {skipped} files not found on disk", flush=True)

    embeddings_tensor = torch.stack(embeddings)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.split}_embeddings.pt"
    torch.save(
        {
            "embeddings": embeddings_tensor,
            "labels": labels_tensor,
            "filenames": filenames,
        },
        out_path,
    )
    print(f"Saved {embeddings_tensor.shape[0]} embeddings to {out_path}", flush=True)
    print(f"Embedding dim: {embeddings_tensor.shape[1]}", flush=True)
    print(f"Label counts: bonafide={sum(l == 0 for l in labels)}, spoof={sum(l == 1 for l in labels)}", flush=True)


if __name__ == "__main__":
    main()
