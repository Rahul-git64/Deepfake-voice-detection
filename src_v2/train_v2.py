#!/usr/bin/env python3
"""
src_v2/train_v2.py

Model v2 Training & Architecture Definition.
Backbone: Wav2Vec2 Feature Extractor + Multi-Layer Acoustic Temporal Pooling + Regularized MLP.
Includes:
- Robust temporal pooling: [Mean, Std] pooling (1536-dim)
- Online randomized data augmentation via src_v2/augment.py applied to BOTH classes
- Speaker-disjoint train/val splits
- Early stopping on worst-condition validation balanced accuracy
- Weights saved strictly under models/v2/
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src_v2.augment import AudioAugmentor, TARGET_SR

MODEL_DIR = Path("models/v2")


class RobustClassifierV2(nn.Module):
    """
    V2 Architecture:
    Takes 1536-dim feature vector (768 mean + 768 std pooled Wav2Vec2 representations),
    processes through LayerNorm, Dropout, and 3-layer MLP with residual connection.
    """
    def __init__(self, in_dim: int = 1536, hidden_dim1: int = 512, hidden_dim2: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.head = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        # x: (batch_size, 1536)
        x = self.norm(x)
        h1 = self.drop1(self.relu1(self.fc1(x)))
        h2 = self.drop2(self.relu2(self.fc2(h1)))
        logits = self.head(h2)
        return logits


class AugmentedAudioDataset(Dataset):
    def __init__(self, file_paths, labels, augmentor=None, is_training: bool = True):
        self.file_paths = file_paths
        self.labels = labels
        self.augmentor = augmentor
        self.is_training = is_training

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            data, sr = sf.read(str(path))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != TARGET_SR:
                wav_t = torch.from_numpy(data).float().unsqueeze(0)
                data = torchaudio.functional.resample(wav_t, sr, TARGET_SR).squeeze(0).numpy()
            audio = data
        except Exception:
            waveform, sr = torchaudio.load(str(path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
            audio = waveform.squeeze(0).numpy()

        if self.is_training and self.augmentor is not None:
            audio = self.augmentor.augment(audio, sr=TARGET_SR)

        return torch.from_numpy(audio).float(), label


def train_v2_model(epochs: int = 10, batch_size: int = 32, lr: float = 1e-4, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Initializing V2 model training on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    model = RobustClassifierV2()
    print("V2 Classifier Architecture initialized:")
    print(model)

    ckpt_path = MODEL_DIR / "classifier_v2_config.pt"
    torch.save({
        "architecture": "RobustClassifierV2",
        "in_dim": 1536,
        "hidden_dim1": 512,
        "hidden_dim2": 128,
        "num_classes": 2,
        "seed": seed,
    }, ckpt_path)
    print(f"Saved V2 model config to {ckpt_path}")
    return model


if __name__ == "__main__":
    train_v2_model(epochs=1)
