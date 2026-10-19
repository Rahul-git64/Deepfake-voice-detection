import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import pearsonr
import torch

from benchmarks.utils import (
    get_train_protocol_df,
    get_dev_protocol_df,
    TRAIN_DIR,
    DEV_DIR,
    get_classifier,
    extract_embedding_from_waveform,
    get_wav2vec2,
)

def extract_clip_descriptors(flac_path):
    y, sr = sf.read(str(flac_path))
    if y.ndim > 1:
        y = y.mean(axis=-1)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    
    silence_thresh = max(1e-5, peak * 0.01)
    is_silence = frame_rms < silence_thresh
    
    lead_sil = 0
    for s in is_silence:
        if s: lead_sil += 1
        else: break
    leading_silence = lead_sil * hop_len / sr
    
    trail_sil = 0
    for s in reversed(is_silence):
        if s: trail_sil += 1
        else: break
    trailing_silence = trail_sil * hop_len / sr
    
    ro_99 = float(np.median(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99, hop_length=hop_len)))
    ro_95 = float(np.median(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95, hop_length=hop_len)))
    noise_floor = float(np.percentile(frame_rms, 10))
    silence_share = float(np.mean(is_silence))
    
    return {
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'noise_floor': noise_floor,
        'rolloff_99': ro_99,
        'rolloff_95': ro_95,
        'leading_silence': leading_silence,
        'trailing_silence': trailing_silence,
        'silence_share': silence_share,
    }

def main():
    print("Extracting acoustic shortcut features...")
    train_df = get_train_protocol_df()
    dev_df = get_dev_protocol_df()

    # Subsample 500 train (250 bona, 250 spoof) and 500 dev (250 bona, 250 spoof) for rapid accurate analysis
    tr_samp = pd.concat([
        train_df[train_df['label'] == 'bonafide'].sample(n=250, random_state=42),
        train_df[train_df['label'] == 'spoof'].sample(n=250, random_state=42),
    ]).sample(frac=1.0, random_state=42)

    dev_samp = pd.concat([
        dev_df[dev_df['label'] == 'bonafide'].sample(n=250, random_state=42),
        dev_df[dev_df['label'] == 'spoof'].sample(n=250, random_state=42),
    ]).sample(frac=1.0, random_state=42)

    clf = get_classifier()
    wav2vec2 = get_wav2vec2()

    tr_records = []
    for _, row in tr_samp.iterrows():
        p = TRAIN_DIR / f"{row['filename']}.flac"
        d = extract_clip_descriptors(p)
        d['label'] = 1 if row['label'] == 'spoof' else 0
        d['attack'] = row['attack']
        d['speaker'] = row['speaker']
        d['filename'] = row['filename']
        
        # Get model score
        aud, _ = sf.read(str(p))
        if aud.ndim > 1: aud = aud.mean(axis=-1)
        emb = extract_embedding_from_waveform(torch.from_numpy(aud).float(), wav2vec2)
        with torch.no_grad():
            prob = torch.softmax(clf(emb), dim=-1)[0, 1].item()
        d['model_score'] = prob
        tr_records.append(d)

    dev_records = []
    for _, row in dev_samp.iterrows():
        p = DEV_DIR / f"{row['filename']}.flac"
        d = extract_clip_descriptors(p)
        d['label'] = 1 if row['label'] == 'spoof' else 0
        d['attack'] = row['attack']
        d['speaker'] = row['speaker']
        d['filename'] = row['filename']
        
        aud, _ = sf.read(str(p))
        if aud.ndim > 1: aud = aud.mean(axis=-1)
        emb = extract_embedding_from_waveform(torch.from_numpy(aud).float(), wav2vec2)
        with torch.no_grad():
            prob = torch.softmax(clf(emb), dim=-1)[0, 1].item()
        d['model_score'] = prob
        dev_records.append(d)

    df_tr = pd.DataFrame(tr_records)
    df_dev = pd.DataFrame(dev_records)

    df_tr.to_csv("benchmarks/data/cache/shortcut_train.csv", index=False)
    df_dev.to_csv("benchmarks/data/cache/shortcut_dev.csv", index=False)

    features = ['rolloff_99', 'rolloff_95', 'noise_floor', 'leading_silence', 'trailing_silence', 'silence_share', 'duration', 'rms', 'peak']

    print("\n--- AUC of Single Descriptors (Predicting Spoof vs Bona Fide) ---")
    for feat in features:
        # Train AUC
        score_tr = df_tr[feat].values
        auc_tr = roc_auc_score(df_tr['label'], score_tr)
        if auc_tr < 0.5:
            auc_tr = 1.0 - auc_tr
        
        # Dev AUC
        score_dev = df_dev[feat].values
        auc_dev = roc_auc_score(df_dev['label'], score_dev)
        if auc_dev < 0.5:
            auc_dev = 1.0 - auc_dev
        print(f"{feat:<18} | Train AUC: {auc_tr:.4f} | Dev AUC: {auc_dev:.4f}")

    # Gradient boosting on all descriptors (trained on train, tested on dev)
    hgb = HistGradientBoostingClassifier(random_state=42)
    hgb.fit(df_tr[features], df_tr['label'])
    hgb_probs = hgb.predict_proba(df_dev[features])[:, 1]
    hgb_auc = roc_auc_score(df_dev['label'], hgb_probs)
    print(f"\nGradient-boosting on all descriptors (speaker-disjoint dev test): AUC = {hgb_auc:.4f}")

    # Per-attack AUC for A01-A06 vs bona fide in dev
    bona_mask = df_dev['label'] == 0
    print("\nPer-attack AUC for A01-A06 vs bona fide (using all descriptors via GBDT):")
    for atk in [f"A{i:02d}" for i in range(1, 7)]:
        atk_mask = df_dev['attack'] == atk
        pair_mask = bona_mask | atk_mask
        if sum(atk_mask) > 0:
            sub_y = df_dev.loc[pair_mask, 'label']
            sub_probs = hgb_probs[pair_mask]
            atk_auc = roc_auc_score(sub_y, sub_probs)
            print(f"  {atk} (n={sum(atk_mask)}): AUC = {atk_auc:.4f}")

    # Within-class partial correlations of classifier.pt's score with each descriptor
    print("\nWithin-class Pearson correlation with classifier.pt spoof score:")
    for cls_name, cls_label in [("Bona Fide (class 0)", 0), ("Spoof (class 1)", 1)]:
        sub = df_dev[df_dev['label'] == cls_label]
        print(f"\n  [{cls_name}] (n={len(sub)}):")
        for feat in features:
            r, p = pearsonr(sub[feat], sub['model_score'])
            print(f"    {feat:<18}: r = {r:+.4f} (p = {p:.4e})")

if __name__ == "__main__":
    main()
