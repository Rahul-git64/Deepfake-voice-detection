import os
import sys
import io
import time
import random
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.pipelines as pipelines
import soundfile as sf
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)

# Ensure Gyan FFmpeg directory is in PATH for any subprocess calls
FFMPEG_DIR = r"C:\Users\rahul\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
if os.path.exists(FFMPEG_DIR) and FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# Fixed Random Seed
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# Paths
ROOT_DIR = Path("C:/Deepfake-voice-detection")
DATASET_ROOT = Path(r"C:\Users\rahul\Downloads\AVSspoof_LA\LA")
EVAL_DIR = DATASET_ROOT / "ASVspoof2019_LA_eval" / "flac"
DEV_DIR = DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac"
TRAIN_DIR = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"

EVAL_PROTOCOL = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
DEV_PROTOCOL = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
TRAIN_PROTOCOL = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"

CLASSIFIER_PATH = ROOT_DIR / "models" / "classifier.pt"
CACHE_DIR = ROOT_DIR / "benchmarks" / "data" / "cache"
RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"
OOD_SPOOF_DIR = ROOT_DIR / "benchmarks" / "data" / "ood_spoof"
OOD_REAL_DIR = ROOT_DIR / "benchmarks" / "data" / "ood_real"

FROZEN_DEV_THRESHOLD = 0.6402  # Threshold applied to spoof probability
TARGET_SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture matching app/main.py exactly
class DeepfakeClassifier(nn.Module):
    def __init__(self, in_features=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# Global lazy model holders
_WAV2VEC2_MODEL = None
_CLASSIFIER_MODEL = None

def get_wav2vec2():
    global _WAV2VEC2_MODEL
    if _WAV2VEC2_MODEL is None:
        bundle = pipelines.WAV2VEC2_BASE
        _WAV2VEC2_MODEL = bundle.get_model().eval().to(DEVICE)
    return _WAV2VEC2_MODEL

def get_classifier():
    global _CLASSIFIER_MODEL
    if _CLASSIFIER_MODEL is None:
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False)
        _CLASSIFIER_MODEL = DeepfakeClassifier(in_features=checkpoint.get("in_features", 768))
        _CLASSIFIER_MODEL.load_state_dict(checkpoint["model_state_dict"])
        _CLASSIFIER_MODEL.eval().to(DEVICE)
    return _CLASSIFIER_MODEL

def preprocess_audio_bytes(audio_bytes: bytes) -> torch.Tensor:
    """Reads audio bytes, resamples to TARGET_SR (16kHz), converts to mono 1D tensor."""
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        waveform = torch.from_numpy(data).float()
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
        return waveform
    except Exception as e:
        raise ValueError(f"Failed to decode audio file: {str(e)}")

def extract_embedding_from_waveform(waveform: torch.Tensor, wav2vec2_model=None) -> torch.Tensor:
    """Extracts mean-pooled 768-dim embedding from 1D waveform tensor."""
    if wav2vec2_model is None:
        wav2vec2_model = get_wav2vec2()
    with torch.no_grad():
        waveform_batch = waveform.unsqueeze(0).to(DEVICE)
        features, _ = wav2vec2_model.extract_features(waveform_batch)
        last_hidden = features[-1]
        embedding = last_hidden.mean(dim=1).to(DEVICE)
    return embedding

def predict_from_embedding(embedding: torch.Tensor, classifier=None) -> Tuple[np.ndarray, float]:
    """Given 1x768 embedding, returns (probs [bonafide, spoof], spoof_prob)"""
    if classifier is None:
        classifier = get_classifier()
    with torch.no_grad():
        logits = classifier(embedding)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs, float(probs[1])

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Equal Error Rate (EER) and threshold where FAR == FRR.
    Returns: eer, threshold, fpr, fnr, thresholds
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    diffs = np.abs(fnr - fpr)
    idx = np.argmin(diffs)
    eer = (fnr[idx] + fpr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold), fpr, fnr, thresholds

def evaluate_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = FROZEN_DEV_THRESHOLD) -> Dict[str, Any]:
    """
    Computes full evaluation metrics given ground truth and spoof scores.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    eer, eer_threshold, fpr, fnr, thresholds = compute_eer(y_true, y_scores)
    roc_auc = auc(fpr, 1.0 - fnr) if len(fpr) > 1 else float("nan")
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)
    pr_auc = auc(pr_recall, pr_precision)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # tn: bonafide predicted as bonafide, fp: bonafide predicted as spoof
    # fn: spoof predicted as bonafide, tp: spoof predicted as spoof
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Acceptance of spoof? In ASVspoof: FAR is bona fide rejected or spoof accepted?
    # In biometrics:
    # False Alarm Rate / False Rejection Rate (FRR) = bona fide rejected as spoof = fp / (tn + fp)
    # False Acceptance Rate (FAR) = spoof accepted as bona fide = fn / (tp + fn)
    # Or in deepfake terminology:
    # Bonafide (0) rejected as spoof: False Reject of human = fp / (tn + fp)
    # Spoof (1) accepted as human: False Accept of fake = fn / (tp + fn)
    frr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    far = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    bacc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "eval_threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bacc),
        "far": float(far),
        "frr": float(frr),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "per_class": {
            "bonafide": {
                "precision": float(prec[0]),
                "recall": float(rec[0]),
                "f1": float(f1[0]),
            },
            "spoof": {
                "precision": float(prec[1]),
                "recall": float(rec[1]),
                "f1": float(f1[1]),
            },
        },
    }

def get_env_info() -> Dict[str, str]:
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(ROOT_DIR)).strip()
    except Exception:
        pass

    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "device": str(DEVICE),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_hash,
    }

def get_eval_protocol_df() -> pd.DataFrame:
    """Loads ASVspoof2019 LA eval protocol."""
    return pd.read_csv(
        EVAL_PROTOCOL,
        sep=" ",
        header=None,
        names=["speaker", "filename", "env", "attack", "label"],
    )

def get_dev_protocol_df() -> pd.DataFrame:
    """Loads ASVspoof2019 LA dev protocol."""
    return pd.read_csv(
        DEV_PROTOCOL,
        sep=" ",
        header=None,
        names=["speaker", "filename", "env", "attack", "label"],
    )

def get_train_protocol_df() -> pd.DataFrame:
    """Loads ASVspoof2019 LA train protocol."""
    return pd.read_csv(
        TRAIN_PROTOCOL,
        sep=" ",
        header=None,
        names=["speaker", "filename", "env", "attack", "label"],
    )

def sample_eval_clips(target_total: int = 3000, min_per_attack: int = 150, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Stratified subsampling of ASVspoof 2019 LA eval:
    At least min_per_attack for each attack A07-A19, plus balanced bona fide.
    Total clips = target_total.
    """
    df = get_eval_protocol_df()
    attacks = [f"A{i:02d}" for i in range(7, 20)]  # A07-A19
    sampled_rows = []
    
    # 13 spoof attacks: 13 * 150 = 1950 spoof clips
    for atk in attacks:
        atk_df = df[df["attack"] == atk]
        sample_n = min(len(atk_df), min_per_attack)
        sampled_rows.append(atk_df.sample(n=sample_n, random_state=seed))

    spoof_sample_so_far = pd.concat(sampled_rows)
    spoof_count = len(spoof_sample_so_far)
    
    # Target bonafide: remaining to target_total (e.g. 3000 - 1950 = 1050)
    bona_target = target_total - spoof_count
    bonafide_df = df[df["label"] == "bonafide"]
    sampled_bona = bonafide_df.sample(n=min(len(bonafide_df), bona_target), random_state=seed)
    
    combined = pd.concat([spoof_sample_so_far, sampled_bona]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined
