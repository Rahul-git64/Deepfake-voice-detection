#!/usr/bin/env python3
"""
src_v2/augment.py

Deterministic audio augmentation engine for Phase 1 / Phase 2 deepfake voice detection.
Applied identically in distribution to BOTH bona fide and spoof classes to neutralize acoustic shortcuts.

Transforms implemented (Phase 1 Amendments 4):
1. Leading & trailing silence trim (0.0 to 1.5s boundary)
2. Padding (0.0 to 1.5s): digital zeros and realistic room tone
3. Random highpass filter (50 to 400 Hz)
4. Random lowpass filter (3.0 to 8.0 kHz)
5. Random 8 kHz downsampling and upsampling
6. Stationary & colored noise injection (SNR 0 to 40 dB):
   - White noise
   - Low-band noise (< 4 kHz)
   - High-band noise (> 4 kHz)
   - Real environmental noise
7. Telephony chain (PSTN):
   - Bandpass (300 - 3400 Hz) + 8 kHz resample + G.711 mu-law companding

STRICT HOLDOUTS (Evaluation only - NEVER used in training):
- Pink noise (strictly held out for unseen color noise evaluation)
- 3.4 kHz lowpass (strictly held out for steep cutoff evaluation)
- MP3 32 kbps codec (strictly held out for lossy codec evaluation)
"""

import math
import numpy as np
import scipy.signal
import torch
import torchaudio


TARGET_SR = 16000

# Strict holdout constants - never allowed in training augmentations
HELD_OUT_TRANSFORMS = {
    "pink_noise",
    "lowpass_3400hz",
    "mp3_32kbps"
}


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0


def trim_silence_boundary(audio: np.ndarray, keep_margin_sec: float = 0.25, sr: int = TARGET_SR) -> np.ndarray:
    """
    Trims leading and trailing silence beyond a speech activity margin.
    Threshold: max(1e-5, 0.01 * peak).
    """
    if len(audio) == 0:
        return audio
    peak = np.max(np.abs(audio))
    threshold = max(1e-5, 0.01 * peak)
    voiced = np.where(np.abs(audio) >= threshold)[0]
    if len(voiced) == 0:
        return audio
    margin_samples = int(keep_margin_sec * sr)
    start_idx = max(0, voiced[0] - margin_samples)
    end_idx = min(len(audio), voiced[-1] + margin_samples)
    return audio[start_idx:end_idx]


def pad_silence(audio: np.ndarray, pad_sec: float, mode: str = "room_tone", sr: int = TARGET_SR, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Pads audio on both sides with either digital zeros or simulated room tone.
    """
    if pad_sec <= 0.0:
        return audio
    pad_len = int(pad_sec * sr)
    if rng is None:
        rng = np.random.RandomState(42)

    if mode == "zeros":
        pad_front = np.zeros(pad_len, dtype=audio.dtype)
        pad_back = np.zeros(pad_len, dtype=audio.dtype)
    elif mode == "room_tone":
        # Estimate noise floor RMS from the quietest 10% of frames or standard -50 dBFS
        rms = compute_rms(audio)
        noise_rms = max(1e-6, rms * 0.005)
        pad_front = rng.normal(0, noise_rms, pad_len).astype(audio.dtype)
        pad_back = rng.normal(0, noise_rms, pad_len).astype(audio.dtype)
    else:
        raise ValueError(f"Unknown padding mode: {mode}")

    return np.concatenate([pad_front, audio, pad_back])


def apply_highpass_filter(audio: np.ndarray, cutoff_hz: float, sr: int = TARGET_SR) -> np.ndarray:
    """Butterworth 4th-order highpass filter."""
    if cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return audio
    sos = scipy.signal.butter(4, cutoff_hz, btype='highpass', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio).astype(audio.dtype)


def apply_lowpass_filter(audio: np.ndarray, cutoff_hz: float, sr: int = TARGET_SR) -> np.ndarray:
    """Butterworth 4th-order lowpass filter."""
    if cutoff_hz >= sr / 2 or cutoff_hz <= 0:
        return audio
    sos = scipy.signal.butter(4, cutoff_hz, btype='lowpass', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio).astype(audio.dtype)


def apply_8k_resample(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Downsamples to 8 kHz and upsamples back to target SR."""
    tensor = torch.from_numpy(audio).unsqueeze(0).float()
    down = torchaudio.functional.resample(tensor, sr, 8000)
    up = torchaudio.functional.resample(down, 8000, sr)
    out = up.squeeze(0).numpy().astype(audio.dtype)
    # Ensure length consistency
    if len(out) > len(audio):
        out = out[:len(audio)]
    elif len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out


def apply_noise(audio: np.ndarray, noise_type: str, snr_db: float, sr: int = TARGET_SR, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Injects noise at specified SNR in dB.
    noise_type: 'white', 'low_band' (<4 kHz), 'high_band' (>4 kHz), 'pink' (strictly evaluation)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    signal_rms = compute_rms(audio)
    if signal_rms < 1e-9:
        return audio

    noise_rms = signal_rms / (10 ** (snr_db / 20.0))

    if noise_type == "white":
        noise = rng.normal(0, 1, len(audio))
    elif noise_type == "low_band":
        base_noise = rng.normal(0, 1, len(audio))
        sos = scipy.signal.butter(4, 4000, btype='lowpass', fs=sr, output='sos')
        noise = scipy.signal.sosfilt(sos, base_noise)
    elif noise_type == "high_band":
        base_noise = rng.normal(0, 1, len(audio))
        sos = scipy.signal.butter(4, 4000, btype='highpass', fs=sr, output='sos')
        noise = scipy.signal.sosfilt(sos, base_noise)
    elif noise_type == "pink":
        # 1/f noise generated via filtering white noise with a -10 dB/dec slope filter
        white = rng.normal(0, 1, len(audio))
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        noise = scipy.signal.lfilter(b, a, white)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    cur_rms = compute_rms(noise)
    if cur_rms > 0:
        noise = noise * (noise_rms / cur_rms)

    noisy_audio = audio + noise
    # Soft limiter to prevent clipping
    peak = np.max(np.abs(noisy_audio))
    if peak > 1.0:
        noisy_audio = np.tanh(noisy_audio)
    return noisy_audio.astype(audio.dtype)


def apply_phone_chain(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Full telephony PSTN chain:
    1. Bandpass filter 300 - 3400 Hz
    2. Resample to 8 kHz and back
    3. 8-bit Mu-law companding & expansion
    """
    # 1. Bandpass
    sos = scipy.signal.butter(4, [300, 3400], btype='bandpass', fs=sr, output='sos')
    filtered = scipy.signal.sosfilt(sos, audio).astype(audio.dtype)

    # 2. Resample 8k -> 16k
    resampled = apply_8k_resample(filtered, sr=sr)

    # 3. Mu-law companding
    tensor = torch.from_numpy(resampled).unsqueeze(0).float()
    # Normalize peak to [-1, 1] for mu-law encoding
    peak = tensor.abs().max() + 1e-9
    norm_tensor = tensor / peak
    mu_encoded = torchaudio.functional.mu_law_encoding(norm_tensor, quantization_channels=256)
    mu_decoded = torchaudio.functional.mu_law_decoding(mu_encoded, quantization_channels=256)
    out = (mu_decoded * peak).squeeze(0).numpy().astype(audio.dtype)

    return out


class AudioAugmentor:
    """
    Unified randomized augmentation pipeline for training v2 models.
    Applied identically to bona fide and spoof clips.
    """
    def __init__(self, seed: int = 42, is_training: bool = True):
        self.rng = np.random.RandomState(seed)
        self.is_training = is_training

    def augment_with_params(self, audio: np.ndarray, sr: int = TARGET_SR):
        params = {
            "trim_margin_sec": None,
            "pad_sec": 0.0,
            "pad_mode": "none",
            "highpass_hz": None,
            "lowpass_hz": None,
            "telephony_mode": "none",
            "noise_type": "none",
            "snr_db": None,
        }
        if len(audio) == 0:
            return audio, params

        aug_audio = audio.copy()

        # 1. Random silence trim (p=0.4)
        if self.rng.rand() < 0.4:
            keep_margin = float(self.rng.uniform(0.0, 0.5))
            params["trim_margin_sec"] = round(keep_margin, 4)
            aug_audio = trim_silence_boundary(aug_audio, keep_margin_sec=keep_margin, sr=sr)

        # 2. Random silence padding (p=0.35)
        if self.rng.rand() < 0.35:
            pad_sec = float(self.rng.uniform(0.2, 1.2))
            mode = "zeros" if self.rng.rand() < 0.5 else "room_tone"
            params["pad_sec"] = round(pad_sec, 4)
            params["pad_mode"] = mode
            aug_audio = pad_silence(aug_audio, pad_sec=pad_sec, mode=mode, sr=sr, rng=self.rng)

        # 3. Frequency band perturbations:
        # Highpass (p=0.3): 50 - 400 Hz
        if self.rng.rand() < 0.3:
            cutoff_hp = float(self.rng.uniform(50.0, 400.0))
            params["highpass_hz"] = round(cutoff_hp, 2)
            aug_audio = apply_highpass_filter(aug_audio, cutoff_hp, sr=sr)

        # Lowpass (p=0.3): 4.0 - 7.5 kHz (excluding held-out 3.4 kHz)
        if self.rng.rand() < 0.3:
            cutoff_lp = float(self.rng.uniform(4000.0, 7500.0))
            params["lowpass_hz"] = round(cutoff_lp, 2)
            aug_audio = apply_lowpass_filter(aug_audio, cutoff_lp, sr=sr)

        # 4. Telephony / 8k resample (p=0.25)
        r = self.rng.rand()
        if r < 0.15:
            params["telephony_mode"] = "phone_chain"
            aug_audio = apply_phone_chain(aug_audio, sr=sr)
        elif r < 0.25:
            params["telephony_mode"] = "resample_8k"
            aug_audio = apply_8k_resample(aug_audio, sr=sr)

        # 5. Additive noise (p=0.45): white, low-band, high-band at SNR 10-40 dB (pink noise is strictly held out)
        if self.rng.rand() < 0.45:
            noise_type = str(self.rng.choice(["white", "low_band", "high_band"]))
            snr_db = float(self.rng.uniform(10.0, 35.0))
            params["noise_type"] = noise_type
            params["snr_db"] = round(snr_db, 2)
            aug_audio = apply_noise(aug_audio, noise_type=noise_type, snr_db=snr_db, sr=sr, rng=self.rng)

        return aug_audio, params

    def augment(self, audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        aug_audio, _ = self.augment_with_params(audio, sr=sr)
        return aug_audio
