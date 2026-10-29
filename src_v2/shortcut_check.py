#!/usr/bin/env python3
"""
src_v2/shortcut_check.py

Trains a tiny Gradient Boosting classifier solely on trivial acoustic descriptors:
- duration
- rms
- peak
- silence_share
- bandwidth_rolloff (spectral rolloff 99%)
- noise_floor
- sample_rate
- codec_id

Outputs:
1. data_v2/shortcut_check_v2.csv:
   Per-feature AUCs and combined GBDT AUC on validation split (before and after augmentation)
   with 2,000-resample within-class bootstrap 95% CIs.
2. data_v2/shortcut_check_results.csv:
   Summary table (raw_unaugmented vs rebalanced_augmented).
3. data_v2/shortcut_descriptors_before_after.csv:
   Per-clip descriptor values before and after augmentation for distribution plotting.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from src_v2.augment import AudioAugmentor, TARGET_SR

RESULTS_PATH = Path("data_v2/shortcut_check_results.csv")
V2_DETAIL_PATH = Path("data_v2/shortcut_check_v2.csv")
DESCRIPTORS_DUMP_PATH = Path("data_v2/shortcut_descriptors_before_after.csv")
DATA_ROOT = Path(r"C:\Users\rahul\Downloads\AVSspoof_LA\LA")


def extract_trivial_descriptors(audio: np.ndarray, sr: int = TARGET_SR, codec_id: int = 0) -> dict:
    if len(audio) == 0:
        return {
            "duration": 0.0, "rms": 0.0, "peak": 0.0, "silence_share": 1.0,
            "bandwidth_rolloff": 0.0, "noise_floor": 0.0, "sample_rate": float(sr), "codec_id": float(codec_id)
        }

    dur = float(len(audio)) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    if len(audio) >= frame_len:
        frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
        silence_thresh = max(1e-5, 0.01 * np.max(frame_rms))
        silence_share = float(np.mean(frame_rms < silence_thresh))
        noise_floor = float(np.percentile(frame_rms, 10))
    else:
        silence_share = 0.0
        noise_floor = rms

    try:
        spec = np.abs(librosa.stft(audio, n_fft=512, hop_length=hop_len))
        rolloff = librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=0.99)
        bandwidth_rolloff = float(np.median(rolloff))
    except Exception:
        bandwidth_rolloff = float(sr / 2)

    return {
        "duration": dur,
        "rms": rms,
        "peak": peak,
        "silence_share": silence_share,
        "bandwidth_rolloff": bandwidth_rolloff,
        "noise_floor": noise_floor,
        "sample_rate": float(sr),
        "codec_id": float(codec_id),
    }


def load_sample_clips(split: str, n_per_class: int = 200, seed: int = 42):
    rng = np.random.RandomState(seed)
    protocol = DATA_ROOT / f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.{'trn' if split=='train' else 'trl'}.txt"
    flac_dir = DATA_ROOT / f"ASVspoof2019_LA_{split}/flac"

    bona_files = []
    spoof_files = []

    with open(protocol, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            fn = parts[1]
            key = parts[4].lower()
            if key == "bonafide":
                bona_files.append(fn)
            else:
                spoof_files.append(fn)

    rng.shuffle(bona_files)
    rng.shuffle(spoof_files)

    selected_bona = bona_files[:n_per_class]
    selected_spoof = spoof_files[:n_per_class]

    clips = []
    for fn in selected_bona:
        clips.append((flac_dir / f"{fn}.flac", 0, fn))
    for fn in selected_spoof:
        clips.append((flac_dir / f"{fn}.flac", 1, fn))

    return clips


def compute_single_feature_auc(y_true: np.ndarray, values: np.ndarray) -> float:
    if np.all(values == values[0]):
        return 0.5
    auc = float(roc_auc_score(y_true, values))
    return max(auc, 1.0 - auc)


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, is_single_feature: bool = False, n_boot: int = 2000, seed: int = 42):
    if np.all(scores == scores[0]):
        return 0.5, 0.5, 0.5
    rng = np.random.RandomState(seed)
    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]
    n0, n1 = len(idx0), len(idx1)

    point_auc = compute_single_feature_auc(y_true, scores) if is_single_feature else float(roc_auc_score(y_true, scores))
    # Determine polarity on full sample if single feature
    raw_auc = float(roc_auc_score(y_true, scores))
    invert = is_single_feature and (raw_auc < 0.5)

    boot_aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s0 = rng.choice(idx0, size=n0, replace=True)
        s1 = rng.choice(idx1, size=n1, replace=True)
        b_idx = np.concatenate([s0, s1])
        b_y = y_true[b_idx]
        b_s = -scores[b_idx] if invert else scores[b_idx]
        if np.all(b_s == b_s[0]):
            boot_aucs[b] = 0.5
        else:
            boot_aucs[b] = roc_auc_score(b_y, b_s)

    low, high = np.percentile(boot_aucs, [2.5, 97.5])
    return float(point_auc), float(low), float(high)


def evaluate_shortcut_suite(n_per_class: int = 200, seed: int = 42):
    print(f"Loading clips for shortcut check (n={n_per_class} per class/split)...", flush=True)
    train_clips = load_sample_clips("train", n_per_class=n_per_class, seed=seed)
    dev_clips = load_sample_clips("dev", n_per_class=n_per_class, seed=seed + 1)

    augmentor = AudioAugmentor(seed=seed)

    feature_cols = [
        "duration", "rms", "peak", "silence_share",
        "bandwidth_rolloff", "noise_floor", "sample_rate", "codec_id"
    ]

    # 1. EXTRACT RAW DESCRIPTORS
    print("Extracting raw trivial descriptors...", flush=True)
    X_train_raw, y_train_raw = [], []
    for path, label, _ in train_clips:
        audio, sr = sf.read(str(path))
        desc = extract_trivial_descriptors(audio, sr=sr, codec_id=0)
        X_train_raw.append([desc[c] for c in feature_cols])
        y_train_raw.append(label)

    X_dev_raw, y_dev_raw = [], []
    dump_rows = []
    for path, label, fn in dev_clips:
        audio, sr = sf.read(str(path))
        desc = extract_trivial_descriptors(audio, sr=sr, codec_id=0)
        X_dev_raw.append([desc[c] for c in feature_cols])
        y_dev_raw.append(label)
        row = {"clip_id": fn, "split": "dev", "label": label, "stage": "before_augmentation"}
        row.update(desc)
        dump_rows.append(row)

    X_train_raw = np.array(X_train_raw, dtype=np.float64)
    y_train_raw = np.array(y_train_raw, dtype=np.int64)
    X_dev_raw = np.array(X_dev_raw, dtype=np.float64)
    y_dev_raw = np.array(y_dev_raw, dtype=np.int64)

    gbdt_raw = HistGradientBoostingClassifier(random_state=seed, max_depth=5, min_samples_leaf=15)
    gbdt_raw.fit(X_train_raw, y_train_raw)
    dev_probs_raw = gbdt_raw.predict_proba(X_dev_raw)[:, 1]
    auc_raw, raw_ci_low, raw_ci_high = bootstrap_auc_ci(y_dev_raw, dev_probs_raw, is_single_feature=False, n_boot=2000, seed=seed)
    bacc_raw = balanced_accuracy_score(y_dev_raw, gbdt_raw.predict(X_dev_raw))
    print(f"[RAW Data] Validation AUC: {auc_raw:.4f} [{raw_ci_low:.4f}, {raw_ci_high:.4f}], Balanced Acc: {bacc_raw:.4f}", flush=True)

    # 2. EXTRACT AUGMENTED DESCRIPTORS
    print("Extracting augmented trivial descriptors with src_v2/augment.py ...", flush=True)
    X_train_aug, y_train_aug = [], []
    for path, label, _ in train_clips:
        audio, sr = sf.read(str(path))
        aug_audio = augmentor.augment(audio, sr=sr)
        desc = extract_trivial_descriptors(aug_audio, sr=sr, codec_id=0)
        X_train_aug.append([desc[c] for c in feature_cols])
        y_train_aug.append(label)

    X_dev_aug, y_dev_aug = [], []
    for path, label, fn in dev_clips:
        audio, sr = sf.read(str(path))
        aug_audio = augmentor.augment(audio, sr=sr)
        desc = extract_trivial_descriptors(aug_audio, sr=sr, codec_id=0)
        X_dev_aug.append([desc[c] for c in feature_cols])
        y_dev_aug.append(label)
        row = {"clip_id": fn, "split": "dev", "label": label, "stage": "after_augmentation"}
        row.update(desc)
        dump_rows.append(row)

    X_train_aug = np.array(X_train_aug, dtype=np.float64)
    y_train_aug = np.array(y_train_aug, dtype=np.int64)
    X_dev_aug = np.array(X_dev_aug, dtype=np.float64)
    y_dev_aug = np.array(y_dev_aug, dtype=np.int64)

    gbdt_aug = HistGradientBoostingClassifier(random_state=seed, max_depth=5, min_samples_leaf=15)
    gbdt_aug.fit(X_train_aug, y_train_aug)
    dev_probs_aug = gbdt_aug.predict_proba(X_dev_aug)[:, 1]
    auc_aug, aug_ci_low, aug_ci_high = bootstrap_auc_ci(y_dev_aug, dev_probs_aug, is_single_feature=False, n_boot=2000, seed=seed)
    bacc_aug = balanced_accuracy_score(y_dev_aug, gbdt_aug.predict(X_dev_aug))
    print(f"[AUGMENTED Data] Validation AUC: {auc_aug:.4f} [{aug_ci_low:.4f}, {aug_ci_high:.4f}], Balanced Acc: {bacc_aug:.4f}", flush=True)

    # 3. BUILD PER-FEATURE + GBDT TABLE (shortcut_check_v2.csv)
    v2_rows = []
    for col_idx, col_name in enumerate(feature_cols):
        f_bef, f_bef_lo, f_bef_hi = bootstrap_auc_ci(
            y_dev_raw, X_dev_raw[:, col_idx], is_single_feature=True, n_boot=2000, seed=seed + col_idx
        )
        f_aft, f_aft_lo, f_aft_hi = bootstrap_auc_ci(
            y_dev_aug, X_dev_aug[:, col_idx], is_single_feature=True, n_boot=2000, seed=seed + 100 + col_idx
        )
        v2_rows.append({
            "feature": col_name,
            "model_type": "single_feature_univariate",
            "n_train": len(y_train_raw),
            "n_val": len(y_dev_raw),
            "auc_before_aug": round(f_bef, 4),
            "auc_before_ci_low": round(f_bef_lo, 4),
            "auc_before_ci_high": round(f_bef_hi, 4),
            "auc_after_aug": round(f_aft, 4),
            "auc_after_ci_low": round(f_aft_lo, 4),
            "auc_after_ci_high": round(f_aft_hi, 4),
            "delta_auc": round(f_aft - f_bef, 4),
            "status_after_aug": "PASS (<0.60)" if f_aft < 0.60 else "EXPLOITABLE (>=0.60)"
        })

    v2_rows.append({
        "feature": "GBDT_all_8_features",
        "model_type": "HistGradientBoostingClassifier",
        "n_train": len(y_train_raw),
        "n_val": len(y_dev_raw),
        "auc_before_aug": round(auc_raw, 4),
        "auc_before_ci_low": round(raw_ci_low, 4),
        "auc_before_ci_high": round(raw_ci_high, 4),
        "auc_after_aug": round(auc_aug, 4),
        "auc_after_ci_low": round(aug_ci_low, 4),
        "auc_after_ci_high": round(aug_ci_high, 4),
        "delta_auc": round(auc_aug - auc_raw, 4),
        "status_after_aug": "PASS (<0.60)" if auc_aug < 0.60 else "EXPLOITABLE (>=0.60)"
    })

    df_v2 = pd.DataFrame(v2_rows)
    V2_DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_v2.to_csv(V2_DETAIL_PATH, index=False)

    # Save descriptor dump
    pd.DataFrame(dump_rows).to_csv(DESCRIPTORS_DUMP_PATH, index=False)

    # Save legacy summary
    results = [
        {
            "condition": "raw_unaugmented",
            "n_train": len(y_train_raw),
            "n_val": len(y_dev_raw),
            "val_auc": round(float(auc_raw), 4),
            "val_auc_ci_low": round(float(raw_ci_low), 4),
            "val_auc_ci_high": round(float(raw_ci_high), 4),
            "val_balanced_acc": round(float(bacc_raw), 4),
            "status": "FAIL_SHORTCUT_EXPLOITABLE" if auc_raw > 0.70 else "PASS",
            "action_taken": "Identified high trivial feature correlation; implemented src_v2/augment.py",
        },
        {
            "condition": "rebalanced_augmented",
            "n_train": len(y_train_aug),
            "n_val": len(y_dev_aug),
            "val_auc": round(float(auc_aug), 4),
            "val_auc_ci_low": round(float(aug_ci_low), 4),
            "val_auc_ci_high": round(float(aug_ci_high), 4),
            "val_balanced_acc": round(float(bacc_aug), 4),
            "status": "PASS_SHORTCUT_NEUTRALIZED" if auc_aug < 0.60 else ("ACCEPTABLE" if auc_aug <= 0.70 else "FAIL"),
            "action_taken": "Applied class-independent random silence trimming, padding, HP/LP filtering, and noise injection.",
        }
    ]
    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_PATH, index=False)

    print(f"\nSaved {V2_DETAIL_PATH}:\n", df_v2.to_string())
    return df_v2


if __name__ == "__main__":
    evaluate_shortcut_suite(n_per_class=200, seed=42)
