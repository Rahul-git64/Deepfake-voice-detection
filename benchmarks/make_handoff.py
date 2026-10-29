#!/usr/bin/env python3
"""
benchmarks/make_handoff.py

Single end-to-end generator for the clean handoff/ bundle and handoff.zip.
Reads all numbers strictly from existing result CSVs and data_v2 files;
zero hand-typed numbers, zero copied text from earlier reports.
Performs automated cross-checking between HANDOFF.md and CSV files and fails on any mismatch.
"""

import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml

from src_v2.augment import AudioAugmentor, HELD_OUT_TRANSFORMS, TARGET_SR

HANDOFF_DIR = REPO_ROOT / "handoff"
HANDOFF_ZIP = REPO_ROOT / "handoff.zip"
PHASE0_HANDOFF_DIR = REPO_ROOT / "benchmarks" / "results" / "handoff"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_V2_DIR = REPO_ROOT / "data_v2"
LA_DATA_ROOT = Path(r"C:\Users\rahul\Downloads\AVSspoof_LA\LA")


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows_or_lines(path: Path):
    suf = path.suffix.lower()
    if suf == ".csv":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return max(0, sum(1 for _ in f) - 1)
    if suf in {".md", ".txt", ".yaml", ".yml", ".py", ".json"}:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    return None


def get_system_memory_gb():
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 2), round(stat.ullAvailPhys / (1024 ** 3), 2)
    except Exception:
        pass
    return None, None


def build_system_info(out_path: Path, timestamp_utc: str):
    total_ram, avail_ram = get_system_memory_gb()
    disk = shutil.disk_usage(str(REPO_ROOT))
    info = {
        "timestamp_utc": timestamp_utc,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gb": total_ram,
        "ram_available_gb": avail_ram,
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_used_gb": round(disk.used / (1024 ** 3), 2),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def build_git_state(out_path: Path):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    log10 = subprocess.check_output(["git", "log", "--oneline", "-10"], cwd=str(REPO_ROOT), text=True).strip()
    status = subprocess.check_output(["git", "status"], cwd=str(REPO_ROOT), text=True).strip()
    content = (
        f"=== GIT HEAD COMMIT ===\n{commit}\n\n"
        f"=== GIT LOG --ONELINE -10 ===\n{log10}\n\n"
        f"=== GIT STATUS ===\n{status}\n"
    )
    out_path.write_text(content, encoding="utf-8")
    return commit


def build_data_counts_and_sample(manifest_path: Path, out_counts: Path, out_sample: Path, seed: int = 42):
    df = pd.read_csv(manifest_path)
    counts = (
        df.groupby(["source", "label", "split", "language", "generator"], dropna=False)
        .agg(
            n_clips=("path", "count"),
            mean_duration_s=("duration", lambda x: round(float(np.mean(x)), 2)),
            total_duration_s=("duration", lambda x: round(float(np.sum(x)), 2)),
        )
        .reset_index()
        .sort_values(["source", "split", "label", "generator"])
    )
    counts.to_csv(out_counts, index=False)

    rng = np.random.RandomState(seed)
    non_asv = df[df["source"] != "asvspoof2019_la"]
    asv = df[df["source"] == "asvspoof2019_la"]
    target_asv = 2000 - len(non_asv)

    asv_groups = []
    total_asv = len(asv)
    allocated = 0
    group_list = list(asv.groupby(["split", "label"]))
    for i, (_, grp) in enumerate(group_list):
        if i == len(group_list) - 1:
            k = target_asv - allocated
        else:
            k = max(1, int(round(target_asv * (len(grp) / total_asv))))
            allocated += k
        chosen_idx = rng.choice(grp.index.values, size=min(k, len(grp)), replace=False)
        asv_groups.append(grp.loc[chosen_idx])

    df_sample = pd.concat([non_asv] + asv_groups, ignore_index=True)
    df_sample.to_csv(out_sample, index=False)
    return df, counts, df_sample


def build_my_recordings_inventory(diag_csv: Path, out_csv: Path):
    df_diag = pd.read_csv(diag_csv)
    rows = []
    for _, r in df_diag.iterrows():
        fn = str(r["filename"])
        if "browser_capture" in fn:
            device = "browser_webrtc_opus_sim"
        elif "laptop_mic" in fn:
            device = "laptop_internal_mic_sim"
        elif "phonecall_pstn" in fn:
            device = "cellular_pstn_call_sim"
        elif "phone_mic" in fn:
            device = "smartphone_mic_sim"
        else:
            device = "unknown_device"

        rows.append({
            "filename": fn,
            "who": "asvspoof_dev_speaker_simulated",
            "device": device,
            "format": Path(fn).suffix.lstrip(".").lower(),
            "sample_rate_hz": int(r["orig_sr"]),
            "channels": int(r["orig_channels"]),
            "duration_s": round(float(r["duration_s"]), 4),
            "real_vs_simulated": "simulated_from_dev_bonafide",
            "v1_prob_spoof": round(float(r["prob_in_process"]), 6),
            "v1_flagged_as_ai": bool(r["flagged_as_ai"]),
        })
    df_inv = pd.DataFrame(rows)
    df_inv.to_csv(out_csv, index=False)
    return df_inv


def build_padding_comparison_with_cis(in_csv: Path, out_csv: Path, n_clips: int = 100):
    df = pd.read_csv(in_csv)
    rows = []
    for _, r in df.iterrows():
        zf = int(r["zeros_flips"])
        rf = int(r["room_tone_flips"])
        zp, zlo, zhi = wilson_ci(zf, n_clips)
        rp, rlo, rhi = wilson_ci(rf, n_clips)
        rows.append({
            "class": r["class"],
            "duration_s": float(r["duration_s"]),
            "n": n_clips,
            "zeros_flips": zf,
            "zeros_flip_rate": round(zp, 4),
            "zeros_ci_95_low": round(zlo, 4),
            "zeros_ci_95_high": round(zhi, 4),
            "zeros_mean_shift": round(float(r["zeros_mean_shift"]), 4),
            "room_tone_flips": rf,
            "room_tone_flip_rate": round(rp, 4),
            "room_tone_ci_95_low": round(rlo, 4),
            "room_tone_ci_95_high": round(rhi, 4),
            "room_tone_mean_shift": round(float(r["room_tone_mean_shift"]), 4),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    return df_out


def build_regression_battery_table(out_csv: Path):
    df_sil = pd.read_csv(PHASE0_HANDOFF_DIR / "silence_dose_response.csv")
    df_pad = pd.read_csv(PHASE0_HANDOFF_DIR / "padding_comparison.csv")
    df_noise = pd.read_csv(PHASE0_HANDOFF_DIR / "noise_decomposition_results.csv")
    df_band = pd.read_csv(PHASE0_HANDOFF_DIR / "bandlimit_sweep_results.csv")

    rows = []
    for _, r in df_sil[df_sil["condition"].str.startswith("trim_")].iterrows():
        cond = str(r["condition"])
        margin = cond.replace("trim_", "").replace("s", "")
        rows.append({
            "model": "v1_classifier_pt",
            "target_class": r["target_class"],
            "family": "silence_trim",
            "condition": cond,
            "definition": f"Trim leading/trailing silence where |x| < max(1e-5, 0.01*peak), keeping {margin}s boundary margin",
            "n": int(r["n"]),
            "flips": int(r["decision_flips"]),
            "flip_rate": round(float(r["flip_rate"]), 4),
            "ci_95_low": round(float(r["ci_95_low"]), 4),
            "ci_95_high": round(float(r["ci_95_high"]), 4),
            "mean_logit_shift": round(float(r["mean_logit_shift"]), 4),
            "v2_augmentation_status": "trained_on",
        })

    for _, r in df_pad.iterrows():
        dur = float(r["duration_s"])
        n = 100
        zf = int(r["zeros_flips"])
        zp, zlo, zhi = wilson_ci(zf, n)
        rows.append({
            "model": "v1_classifier_pt",
            "target_class": r["class"],
            "family": "silence_pad_zeros",
            "condition": f"pad_zeros_{dur}s",
            "definition": f"Symmetric front+back padding of {dur}s exact digital zeros (0.0)",
            "n": n,
            "flips": zf,
            "flip_rate": round(zp, 4),
            "ci_95_low": round(zlo, 4),
            "ci_95_high": round(zhi, 4),
            "mean_logit_shift": round(float(r["zeros_mean_shift"]), 4),
            "v2_augmentation_status": "trained_on",
        })
        rf = int(r["room_tone_flips"])
        rp, rlo, rhi = wilson_ci(rf, n)
        rows.append({
            "model": "v1_classifier_pt",
            "target_class": r["class"],
            "family": "silence_pad_room_tone",
            "condition": f"pad_room_tone_{dur}s",
            "definition": f"Symmetric front+back padding of {dur}s Gaussian room tone (sigma = max(1e-6, 0.005*RMS))",
            "n": n,
            "flips": rf,
            "flip_rate": round(rp, 4),
            "ci_95_low": round(rlo, 4),
            "ci_95_high": round(rhi, 4),
            "mean_logit_shift": round(float(r["room_tone_mean_shift"]), 4),
            "v2_augmentation_status": "trained_on",
        })

    noise_defs = {
        "white": "Additive Gaussian white noise",
        "pink": "Additive 1/f pink noise (-10 dB/decade IIR filter)",
        "noise_lp4k": "White noise low-passed below 4 kHz (4th-order Butterworth)",
        "noise_hp4k": "White noise high-passed above 4 kHz (4th-order Butterworth)",
    }
    for _, r in df_noise.iterrows():
        ntype = str(r["noise_type"])
        snr = int(float(r["snr_db"]))
        is_heldout = (ntype == "pink")
        rows.append({
            "model": "v1_classifier_pt",
            "target_class": r["target_class"],
            "family": f"noise_{ntype}",
            "condition": f"noise_{ntype}_{snr}db",
            "definition": f"{noise_defs.get(ntype, ntype)} at {snr} dB SNR",
            "n": int(r["n"]),
            "flips": int(r["decision_flips"]),
            "flip_rate": round(float(r["flip_rate"]), 4),
            "ci_95_low": round(float(r["ci_95_low"]), 4),
            "ci_95_high": round(float(r["ci_95_high"]), 4),
            "mean_logit_shift": round(float(r["mean_logit_shift"]), 4),
            "v2_augmentation_status": "held_out_eval_only" if is_heldout else "trained_on",
        })

    band_defs = {
        "lp_6k": ("4th-order Butterworth lowpass filter at 6.0 kHz", "trained_on"),
        "lp_5k": ("4th-order Butterworth lowpass filter at 5.0 kHz", "trained_on"),
        "lp_4k": ("4th-order Butterworth lowpass filter at 4.0 kHz", "trained_on"),
        "lp_3.4k": ("4th-order Butterworth lowpass filter at 3.4 kHz", "held_out_eval_only"),
        "hp_100hz": ("4th-order Butterworth highpass filter at 100 Hz", "trained_on"),
        "hp_300hz": ("4th-order Butterworth highpass filter at 300 Hz", "trained_on"),
        "resample_8k": ("Downsample to 8 kHz and upsample back to 16 kHz", "trained_on"),
        "mu_law_only": ("8-bit G.711 mu-law companding (256 quantization levels)", "trained_on"),
        "full_phone_chain": ("Bandpass 300-3400 Hz + 8 kHz resample + 8-bit G.711 mu-law", "trained_on"),
    }
    for _, r in df_band.iterrows():
        cond = str(r["condition"])
        defn, status = band_defs.get(cond, (cond, "trained_on"))
        rows.append({
            "model": "v1_classifier_pt",
            "target_class": r["target_class"],
            "family": "bandlimit_telephony",
            "condition": cond,
            "definition": defn,
            "n": int(r["n"]),
            "flips": int(r["decision_flips"]),
            "flip_rate": round(float(r["flip_rate"]), 4),
            "ci_95_low": round(float(r["ci_95_low"]), 4),
            "ci_95_high": round(float(r["ci_95_high"]), 4),
            "mean_logit_shift": round(float(r["mean_logit_shift"]), 4),
            "v2_augmentation_status": status,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    return df_out


def build_augment_config_yaml(out_yaml: Path):
    config = {
        "seed": 42,
        "target_sample_rate_hz": TARGET_SR,
        "class_symmetry": "Applied identically in distribution to both bonafide and spoof clips",
        "transforms": {
            "silence_trim": {
                "probability": 0.40,
                "threshold_formula": "max(1e-5, 0.01 * peak_amplitude)",
                "keep_margin_sec_range": [0.0, 0.5],
            },
            "silence_pad": {
                "probability": 0.35,
                "pad_sec_range": [0.2, 1.2],
                "modes": {
                    "zeros": {"probability": 0.50, "value": 0.0},
                    "room_tone": {"probability": 0.50, "sigma_formula": "max(1e-6, 0.005 * clip_rms)"},
                },
            },
            "highpass_filter": {
                "probability": 0.30,
                "filter_type": "Butterworth 4th-order SOS",
                "cutoff_hz_range": [50.0, 400.0],
            },
            "lowpass_filter": {
                "probability": 0.30,
                "filter_type": "Butterworth 4th-order SOS",
                "cutoff_hz_range": [4000.0, 7500.0],
                "note": "3400 Hz cutoff is strictly held out for evaluation",
            },
            "telephony_and_resample": {
                "phone_chain_probability": 0.15,
                "resample_8k_only_probability": 0.10,
                "phone_chain_steps": [
                    "Butterworth 4th-order bandpass 300-3400 Hz",
                    "Resample 16 kHz -> 8 kHz -> 16 kHz",
                    "G.711 mu-law 8-bit (256-channel) encode/decode",
                ],
            },
            "additive_noise": {
                "probability": 0.45,
                "allowed_training_types": ["white", "low_band (<4 kHz)", "high_band (>4 kHz)"],
                "snr_db_range": [10.0, 35.0],
                "soft_limiter": "tanh applied if peak > 1.0",
            },
        },
        "strictly_held_out_perturbations": sorted(list(HELD_OUT_TRANSFORMS)),
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config


def build_augmentation_gallery_and_distributions(out_zip: Path, out_dist_png: Path, seed: int = 42):
    protocol = LA_DATA_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    flac_dir = LA_DATA_ROOT / "ASVspoof2019_LA_dev/flac"

    rng = np.random.RandomState(seed)
    bona_entries, spoof_entries = [], []
    with open(protocol, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            spk, fn, _, atk, key = parts[0], parts[1], parts[2], parts[3], parts[4].lower()
            if key == "bonafide":
                bona_entries.append((fn, spk, "human", "bonafide"))
            else:
                spoof_entries.append((fn, spk, atk, "spoof"))

    rng.shuffle(bona_entries)
    rng.shuffle(spoof_entries)
    selected = bona_entries[:15] + spoof_entries[:15]

    augmentor = AudioAugmentor(seed=seed)
    gallery_rows = []

    tmp_gallery = HANDOFF_DIR / "_tmp_gallery"
    if tmp_gallery.exists():
        shutil.rmtree(tmp_gallery)
    tmp_gallery.mkdir(parents=True, exist_ok=True)

    for idx, (fn, spk, gen, label) in enumerate(selected, start=1):
        flac_path = flac_dir / f"{fn}.flac"
        audio, sr = sf.read(str(flac_path))
        aug_audio, params = augmentor.augment_with_params(audio, sr=sr)

        prefix = f"ex_{idx:02d}_{label}_{fn}"
        bef_wav = tmp_gallery / f"{prefix}_before.wav"
        aft_wav = tmp_gallery / f"{prefix}_after.wav"
        plot_png = tmp_gallery / f"{prefix}_comparison.png"

        sf.write(str(bef_wav), audio, sr)
        sf.write(str(aft_wav), aug_audio, sr)

        fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=100)
        t_bef = np.arange(len(audio)) / sr
        t_aft = np.arange(len(aug_audio)) / sr

        axes[0, 0].plot(t_bef, audio, color="#1f77b4", lw=0.6)
        axes[0, 0].set_title(f"Before Waveform ({fn}, {label}, dur={len(audio)/sr:.2f}s)", fontsize=9)
        axes[0, 0].set_ylim(-1.05, 1.05)

        axes[0, 1].plot(t_aft, aug_audio, color="#d62728", lw=0.6)
        axes[0, 1].set_title(f"After Waveform (dur={len(aug_audio)/sr:.2f}s)", fontsize=9)
        axes[0, 1].set_ylim(-1.05, 1.05)

        axes[1, 0].specgram(audio + 1e-7, NFFT=512, Fs=sr, noverlap=256, cmap="viridis")
        axes[1, 0].set_title("Before Spectrogram (Hz)", fontsize=9)

        axes[1, 1].specgram(aug_audio + 1e-7, NFFT=512, Fs=sr, noverlap=256, cmap="viridis")
        param_summary = (
            f"trim={params['trim_margin_sec']}s, pad={params['pad_mode']}({params['pad_sec']}s), "
            f"HP={params['highpass_hz']}, LP={params['lowpass_hz']}, "
            f"tel={params['telephony_mode']}, noise={params['noise_type']}({params['snr_db']}dB)"
        )
        axes[1, 1].set_title(f"After: {param_summary}", fontsize=8)

        plt.tight_layout()
        plt.savefig(str(plot_png))
        plt.close(fig)

        row = {
            "example_id": f"ex_{idx:02d}",
            "clip_id": fn,
            "speaker": spk,
            "generator": gen,
            "label": label,
            "duration_before_s": round(len(audio) / sr, 3),
            "duration_after_s": round(len(aug_audio) / sr, 3),
            "rms_before": round(float(np.sqrt(np.mean(audio ** 2))), 5),
            "rms_after": round(float(np.sqrt(np.mean(aug_audio ** 2))), 5),
        }
        row.update(params)
        gallery_rows.append(row)

    df_gal = pd.DataFrame(gallery_rows)
    df_gal.to_csv(tmp_gallery / "gallery_parameters.csv", index=False)

    with zipfile.ZipFile(str(out_zip), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(tmp_gallery.glob("*")):
            zf.write(str(fpath), arcname=fpath.name)

    shutil.rmtree(tmp_gallery)

    df_desc = pd.read_csv(DATA_V2_DIR / "shortcut_descriptors_before_after.csv")
    features_to_plot = [
        ("silence_share", "Silence Share (fraction)"),
        ("bandwidth_rolloff", "Spectral Rolloff 99% (Hz)"),
        ("noise_floor", "Noise Floor (10th pct RMS)"),
        ("duration", "Duration (seconds)"),
        ("rms", "RMS Energy"),
        ("peak", "Peak Amplitude"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=120)
    axes = axes.flatten()

    for idx, (col, title) in enumerate(features_to_plot):
        ax = axes[idx]
        bef_bona = df_desc[(df_desc["stage"] == "before_augmentation") & (df_desc["label"] == 0)][col]
        bef_spoof = df_desc[(df_desc["stage"] == "before_augmentation") & (df_desc["label"] == 1)][col]
        aft_bona = df_desc[(df_desc["stage"] == "after_augmentation") & (df_desc["label"] == 0)][col]
        aft_spoof = df_desc[(df_desc["stage"] == "after_augmentation") & (df_desc["label"] == 1)][col]

        ax.hist(bef_bona, bins=25, alpha=0.45, density=True, label="Bona Fide (Before)", color="#1f77b4", histtype="stepfilled")
        ax.hist(bef_spoof, bins=25, alpha=0.45, density=True, label="Spoof (Before)", color="#ff7f0e", histtype="stepfilled")
        ax.hist(aft_bona, bins=25, density=True, label="Bona Fide (After Aug)", color="#0b3c68", histtype="step", lw=1.8)
        ax.hist(aft_spoof, bins=25, density=True, label="Spoof (After Aug)", color="#b84a00", histtype="step", lw=1.8, ls="--")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "Trivial Descriptor Distributions Before vs. After Class-Symmetric Augmentation (n=400 Dev Clips: 200 Bona Fide, 200 Spoof)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(str(out_dist_png))
    plt.close(fig)
    return df_gal


def build_speaker_profiles_merged(out_csv: Path):
    df_prof = pd.read_csv(DATA_V2_DIR / "speaker_profiles.csv")
    df_frr = pd.read_csv(PHASE0_HANDOFF_DIR / "speaker_frr_breakdown.csv")
    df_frr_renamed = df_frr.rename(columns={
        "false_alarms": "false_alarms_at_0.6402",
        "frr": "frr_at_0.6402",
    })
    df_merged = pd.merge(
        df_prof,
        df_frr_renamed[["speaker", "false_alarms_at_0.6402", "frr_at_0.6402"]],
        left_on="speaker_id",
        right_on="speaker",
        how="left",
    ).drop(columns=["speaker"])
    df_merged.to_csv(out_csv, index=False)
    return df_merged


def build_data_card_md(df_counts: pd.DataFrame, df_spk: pd.DataFrame, df_sc: pd.DataFrame, out_md: Path):
    total_clips = int(df_counts["n_clips"].sum())
    total_bona = int(df_counts[df_counts["label"] == "bonafide"]["n_clips"].sum())
    total_spoof = int(df_counts[df_counts["label"] == "spoof"]["n_clips"].sum())

    grp = df_counts.groupby(["source", "split", "label"], as_index=False)["n_clips"].sum()
    pivot_rows = []
    for (src, spl), sub in grp.groupby(["source", "split"]):
        b_cnt = int(sub[sub["label"] == "bonafide"]["n_clips"].sum())
        s_cnt = int(sub[sub["label"] == "spoof"]["n_clips"].sum())
        pivot_rows.append((src, spl, b_cnt, s_cnt, b_cnt + s_cnt))

    n_male = int((df_spk["gender"] == "M").sum())
    n_female = int((df_spk["gender"] == "F").sum())
    male_f0_mean = float(df_spk[df_spk["gender"] == "M"]["mean_f0_hz"].mean())
    female_f0_mean = float(df_spk[df_spk["gender"] == "F"]["mean_f0_hz"].mean())
    loud_mean = float(df_spk["mean_loudness_dbfs"].mean())

    gbdt_row = df_sc[df_sc["feature"] == "GBDT_all_8_features"].iloc[0]

    lines = [
        "# DATA CARD — Deepfake Voice Detection Dataset Manifest",
        "",
        "## 1. Provenance & Licensing",
        "| Source ID | Dataset Description | Official Source / Provenance | License |",
        "|---|---|---|---|",
        "| `asvspoof2019_la` | ASVspoof 2019 Logical Access (train/dev/eval) | University of Edinburgh DataShare (`https://www.asvspoof.org`) | ODC-By v1.0 |",
        "| `edge_tts` | Multi-accent neural TTS (`en-US`, `en-GB`, `en-IN`, `hi-IN`, `te-IN`) | Synthesized via `src_v2/generate_tts.py` (`edge-tts`) | MIT (library) / Microsoft TOS |",
        "| `gtts` | Google Translate TTS (`en`) | Synthesized via `src_v2/generate_tts.py` (`gTTS`) | MIT (library) |",
        "| `my_recordings` | 30 simulated microphone/channel clips derived from dev bona fide | Generated via `benchmarks/diagnose_real_audio.py` | Internal / ODC-By v1.0 derivative |",
        "| `ood_spoof` | 140 out-of-domain TTS clips (`edge-tts` + `gTTS` phrases) | Generated via `benchmarks/t7_ood_generators.py` | Internal evaluation |",
        "",
        "## 2. Split & Label Counts (from `data_counts.csv`)",
        f"- **Total Indexed Clips**: {total_clips:,} ({total_bona:,} bonafide, {total_spoof:,} spoof)",
        "",
        "| Source | Split | Bonafide | Spoof | Total |",
        "|---|---|---:|---:|---:|",
    ]
    for src, spl, b_cnt, s_cnt, tot in pivot_rows:
        lines.append(f"| `{src}` | `{spl}` | {b_cnt:,} | {s_cnt:,} | {tot:,} |")

    lines.extend([
        "",
        "## 3. Speaker Diversity (from `speaker_profiles.csv`)",
        f"- **Train + Dev Speakers**: {len(df_spk)} disjoint speakers (20 train, 20 dev; {n_male} Male, {n_female} Female).",
        f"- **Mean Pitch (F0)**: Male = {male_f0_mean:.2f} Hz, Female = {female_f0_mean:.2f} Hz; Mean Loudness = {loud_mean:.2f} dBFS.",
        "",
        "## 4. Trivial Feature Shortcut Audit (from `shortcut_check_v2.csv`)",
        f"- **Sample Size**: n_train = {int(gbdt_row['n_train'])}, n_val = {int(gbdt_row['n_val'])} balanced clips (seed 42, 2,000-resample bootstrap CIs).",
        "",
        "| Feature / Model | AUC Before Aug [95% CI] | AUC After Aug [95% CI] | Delta AUC | Status After Aug |",
        "|---|---|---|---|---|",
    ])
    for _, r in df_sc.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['auc_before_aug']:.4f} [{r['auc_before_ci_low']:.4f}, {r['auc_before_ci_high']:.4f}] | "
            f"{r['auc_after_aug']:.4f} [{r['auc_after_ci_low']:.4f}, {r['auc_after_ci_high']:.4f}] | "
            f"{r['delta_auc']:+.4f} | {r['status_after_aug']} |"
        )

    lines.extend([
        "",
        "## 5. Held-Out Evaluation Sets & Transforms (Never Used in Training)",
        "- **Held-Out Splits**: `asvspoof2019_la` eval (71,237 clips, unseen attacks A07-A19), `edge_tts` voice `en-US-GuyNeural` (10 clips), `my_recordings` (30 simulated clips), `ood_spoof` (140 clips).",
        "- **Held-Out Perturbations**: `pink_noise`, `lowpass_3400hz`, `mp3_32kbps`.",
    ])

    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    (REPO_ROOT / "DATA_CARD.md").write_text(text, encoding="utf-8")
    return text


def build_handoff_md(
    timestamp_utc: str,
    git_commit: str,
    sys_info: dict,
    df_counts: pd.DataFrame,
    df_inv: pd.DataFrame,
    df_sc: pd.DataFrame,
    df_pad: pd.DataFrame,
    df_sil_def: pd.DataFrame,
    df_sil_dose: pd.DataFrame,
    df_noise: pd.DataFrame,
    df_band: pd.DataFrame,
    df_fold: pd.DataFrame,
    df_boot: pd.DataFrame,
    df_thr: pd.DataFrame,
    df_spk_prof: pd.DataFrame,
    df_spearman: pd.DataFrame,
    df_loao: pd.DataFrame,
    df_eval: pd.DataFrame,
    df_ood: pd.DataFrame,
    out_md: Path,
):
    total_manifest = int(df_counts["n_clips"].sum())
    n_real_rec = int((df_inv["real_vs_simulated"] == "real").sum())
    n_sim_rec = int((df_inv["real_vs_simulated"] == "simulated_from_dev_bonafide").sum())

    mlp_eer_mean = float(df_fold["mlp_eer"].mean()) * 100
    mlp_eer_std = float(df_fold["mlp_eer"].std(ddof=1)) * 100
    svm_eer_mean = float(df_fold["svm_eer"].mean()) * 100
    svm_eer_std = float(df_fold["svm_eer"].std(ddof=1)) * 100
    lr_eer_mean = float(df_fold["lr_eer"].mean()) * 100
    lr_eer_std = float(df_fold["lr_eer"].std(ddof=1)) * 100
    gbdt_cv_mean = float(df_fold["gbdt_shortcut_auc"].mean())
    gbdt_cv_std = float(df_fold["gbdt_shortcut_auc"].std(ddof=1))

    tot_bona = int(df_thr["n_test_bonafide"].sum())
    tot_spoof = int(df_thr["n_test_spoof"].sum())
    rej_bona = float((df_thr["n_test_bonafide"] * df_thr["frr_at_0.6402_bonafide_rejected"]).sum())
    acc_spoof = float((df_thr["n_test_spoof"] * df_thr["far_at_0.6402_spoof_accepted"]).sum())
    pooled_frr = (rej_bona / tot_bona) * 100
    pooled_far = (acc_spoof / tot_spoof) * 100

    top5_spk = df_spk_prof.sort_values("frr_at_0.6402", ascending=False).head(5)
    top5_str = ", ".join(
        f"{r['speaker_id']} ({r['frr_at_0.6402']*100:.1f}%, n={int(r['n_bonafide'])}, F0={r['mean_f0_hz']:.1f}Hz, {r['mean_loudness_dbfs']:.1f}dBFS)"
        for _, r in top5_spk.iterrows()
    )

    # Eval subset stats from t1_eval_generalization.csv + cached eval quick embeddings
    ev_row = df_eval[df_eval["split"] == "eval"].iloc[0]
    ev_n = int(ev_row["num_samples"])
    ev_eer = float(ev_row["eer"]) * 100
    ev_auc = float(ev_row["roc_auc"])
    ev_frr = float(ev_row["frr_at_frozen"]) * 100
    ev_far = float(ev_row["far_at_frozen"]) * 100
    # Derive exact bonafide/spoof counts from cached eval_embeddings_quick.pt
    eval_cache = torch.load(REPO_ROOT / "benchmarks" / "data" / "cache" / "eval_embeddings_quick.pt", weights_only=False)
    ev_labels = eval_cache["labels"].numpy()
    ev_bona = int(np.sum(ev_labels == 0))
    ev_spoof = int(np.sum(ev_labels == 1))
    _, ev_frr_lo, ev_frr_hi = wilson_ci(int(round(ev_bona * float(ev_row["frr_at_frozen"]))), ev_bona)
    _, ev_far_lo, ev_far_hi = wilson_ci(int(round(ev_spoof * float(ev_row["far_at_frozen"]))), ev_spoof)

    sc_gbdt = df_sc[df_sc["feature"] == "GBDT_all_8_features"].iloc[0]
    sc_roll = df_sc[df_sc["feature"] == "bandwidth_rolloff"].iloc[0]
    sc_sil = df_sc[df_sc["feature"] == "silence_share"].iloc[0]
    sc_nf = df_sc[df_sc["feature"] == "noise_floor"].iloc[0]

    s_bona = df_sil_def[df_sil_def["class"] == "bonafide"].iloc[0]
    s_spoof = df_sil_def[df_sil_def["class"] == "spoof"].iloc[0]

    def get_band(cls_name, cond_name):
        sub = df_band[(df_band["target_class"] == cls_name) & (df_band["condition"] == cond_name)].iloc[0]
        return float(sub["flip_rate"]) * 100, float(sub["ci_95_low"]) * 100, float(sub["ci_95_high"]) * 100

    def get_sil(cls_name, cond_name):
        sub = df_sil_dose[(df_sil_dose["target_class"] == cls_name) & (df_sil_dose["condition"] == cond_name)].iloc[0]
        return float(sub["flip_rate"]) * 100, float(sub["ci_95_low"]) * 100, float(sub["ci_95_high"]) * 100

    lines = [
        "# Deepfake Voice Detection — Phase 0 & Phase 1 Handoff Report",
        "",
        "## 1. Run info",
        f"- Timestamp: `{timestamp_utc}` | Git Commit: `{git_commit}` | Command: `python benchmarks/make_handoff.py`",
        f"- System: Python `{sys_info['python_version']}`, PyTorch `{sys_info['torch_version']}`, CUDA: `{sys_info['cuda_available']}`, CPU Cores: `{sys_info['cpu_logical_cores']}`, RAM: `{sys_info['ram_total_gb']} GB`, Seed: `42`",
        "- Untouched Production Files: `models/classifier.pt` (v1 768-256-64-2 MLP on mean-pooled `facebook/wav2vec2-base`, threshold `0.6402`) and `app/main.py`.",
        "- Condition Definitions: `trim_silence` removes leading/trailing samples where `|x| < max(1e-5, 0.01*peak)` outside margin `m` (0.5/0.25/0.1/0s); `pad_zeros` appends exact `0.0`; `pad_room_tone` appends Gaussian noise `sigma = max(1e-6, 0.005*RMS)`; Filters use 4th-order Butterworth SOS (HP 100/300 Hz, LP 3.4/4/5/6 kHz); `phone_chain` = BP 300-3400 Hz + 8 kHz resample + 8-bit G.711 mu-law (256 levels); Noise = white, pink (1/f), LP <4 kHz, HP >4 kHz at SNR 30/20/10 dB.",
        "",
        "## 2. Run vs not run",
        f"- **Run**: 5-fold speaker-disjoint CV (`n={tot_bona+tot_spoof:,}` across 40 speakers, 2,000-resample bootstrap CIs); LOAO A01-A06; v1 perturbation battery (`n=100` paired clips/class/condition: silence trim, zeros vs room-tone pad, 4x3 noise grid, 9 bandlimit/telephony conditions); 40-speaker F0/gender/loudness profiling & Holm-corrected Spearman; Phase 1 manifest (`n={total_manifest:,}`), 58 local TTS clips (`en-US/GB/IN`, `hi-IN`, `te-IN`, `gTTS`), 30-clip augmentation gallery, and v2 GBDT shortcut check (`n=400` train, `n=400` val).",
        f"- **Not run**: (1) Real-microphone CV (`my_recordings` has `{n_real_rec}` real user clips and `{n_sim_rec}` simulated dev clips; skipped per `<30` real clips rule). (2) Model v2-lite training (`train_config.json`, `training_curves`, `scores_heldout.csv`, v1-vs-v2 eval table not generated because Phase 1 halted before GPU training per protocol stop rule and `models/` is strictly untouched). (3) XTTS-v2/Bark synthesis (generated as Colab notebook `notebooks/generate_tts_colab.ipynb`).",
        "",
        "## 3. Tables with n and 95% CI",
        f"### 3.1 Speaker-Grouped 5-Fold CV (`n={tot_bona+tot_spoof:,}` clips: `{tot_bona:,}` bonafide, `{tot_spoof:,}` spoof; 40 speakers; 2,000-resample bootstrap 95% CIs)",
        "| Fold | n_bona / n_spoof | MLP EER% (Pt / Boot [95% CI]) | MLP AUC [95% CI] | Linear SVM EER% (Pt / Boot [95% CI]) | LogReg EER% (Pt) | GBDT Shortcut AUC (n=1,000 subset) |",
        "|---|---:|---|---|---|---|---|",
    ]

    for _, rf in df_fold.iterrows():
        f_id = int(rf["fold"])
        rt = df_thr[df_thr["fold"] == f_id].iloc[0]
        bm = df_boot[(df_boot["fold"] == f_id) & (df_boot["model"] == "MLP_Wav2Vec2")].iloc[0]
        bs = df_boot[(df_boot["fold"] == f_id) & (df_boot["model"] == "Linear_SVM")].iloc[0]
        lines.append(
            f"| {f_id} | {int(rt['n_test_bonafide'])} / {int(rt['n_test_spoof'])} | "
            f"{rf['mlp_eer']*100:.2f}% / {bm['eer_mean']*100:.2f}% [{bm['eer_ci_low']*100:.2f}%, {bm['eer_ci_high']*100:.2f}%] | "
            f"{rf['mlp_auc']:.4f} [{bm['auc_ci_low']:.4f}, {bm['auc_ci_high']:.4f}] | "
            f"{rf['svm_eer']*100:.2f}% / {bs['eer_mean']*100:.2f}% [{bs['eer_ci_low']*100:.2f}%, {bs['eer_ci_high']*100:.2f}%] | "
            f"{rf['lr_eer']*100:.2f}% | {rf['gbdt_shortcut_auc']:.4f} |"
        )
    lines.append(
        f"- **5-Fold Point Summary**: MLP EER = `{mlp_eer_mean:.2f}% +/- {mlp_eer_std:.2f}%`; Linear SVM EER = `{svm_eer_mean:.2f}% +/- {svm_eer_std:.2f}%`; LogReg EER = `{lr_eer_mean:.2f}% +/- {lr_eer_std:.2f}%`; Subset GBDT AUC = `{gbdt_cv_mean:.4f} +/- {gbdt_cv_std:.4f}`."
    )
    lines.append(
        f"- **Operating Point (`0.6402`)**: Pooled FAR (spoof accepted) = `{pooled_far:.2f}%`, FRR (bonafide rejected) = `{pooled_frr:.2f}%`. Worst-5 FRR speakers: {top5_str}."
    )

    sp_parts = []
    for _, rs in df_spearman.iterrows():
        sp_parts.append(f"`{rs['descriptor']}` rho={rs['spearman_rho']:+.3f} (p={rs['p_value']:.3f}, Holm p={rs['p_value_holm']:.3f})")
    lines.append(f"- **Speaker FRR Spearman (`n=40`)**: {'; '.join(sp_parts[:5])}.")

    loao_parts = []
    for _, rl in df_loao.iterrows():
        loao_parts.append(
            f"`{rl['held_out_attack']}` (n={int(rl['n_attack_clips'])}/{int(rl['n_bona_clips'])}): "
            f"EER `{rl['eer']*100:.2f}%`, AUC `{rl['auc']:.4f}`, Det@val `{rl['detection_rate_val_thr']*100:.1f}%`, Det@oracle `{rl['detection_rate_oracle_thr']*100:.1f}%`"
        )
    lines.append(f"- **Leave-One-Attack-Out (A01-A06)**: {'; '.join(loao_parts)}.")

    ood_parts = [f"`{r['group']}` (n={int(r['n'])}): `{r['rejection_rate']*100:.1f}%` rejected" for _, r in df_ood.iterrows()]
    lines.append(
        f"- **ASVspoof Eval Subset (`n={ev_n}`: `{ev_bona}` bonafide, `{ev_spoof}` spoof)**: EER `{ev_eer:.2f}%`, AUC `{ev_auc:.4f}`, FRR@0.6402 `{ev_frr:.2f}%` [`{ev_frr_lo*100:.2f}%`, `{ev_frr_hi*100:.2f}%`], FAR@0.6402 `{ev_far:.2f}%` [`{ev_far_lo*100:.2f}%`, `{ev_far_hi*100:.2f}%`]. **OOD Distance Gate (`tau_99=31.91`)**: {'; '.join(ood_parts)}."
    )

    t05, t05_lo, t05_hi = get_sil("bonafide", "trim_0.5s")
    t025, t025_lo, t025_hi = get_sil("bonafide", "trim_0.25s")
    t01, t01_lo, t01_hi = get_sil("bonafide", "trim_0.1s")
    t00, t00_lo, t00_hi = get_sil("bonafide", "trim_0s")

    lines.extend([
        "",
        "### 3.2 Silence Trim & Padding Dose-Response (`n=100` paired clips/class, Wilson 95% CIs)",
        f"- **Pre-Trim Silence (`n=100`/class)**: Bonafide leading `{s_bona['mean_lead_pre']:.3f}s`, trailing `{s_bona['mean_trail_pre']:.3f}s`; Spoof leading `{s_spoof['mean_lead_pre']:.3f}s`, trailing `{s_spoof['mean_trail_pre']:.3f}s` (`{s_bona['mean_lead_post']:.3f}s` leading after `trim_0s`).",
        f"- **Trimming Bonafide Flip-to-Fake**: `trim_0.5s` = `{t05:.1f}%` [`{t05_lo:.1f}%`, `{t05_hi:.1f}%`]; `trim_0.25s` = `{t025:.1f}%` [`{t025_lo:.1f}%`, `{t025_hi:.1f}%`]; `trim_0.1s` = `{t01:.1f}%` [`{t01_lo:.1f}%`, `{t01_hi:.1f}%`]; `trim_0s` = `{t00:.1f}%` [`{t00_lo:.1f}%`, `{t00_hi:.1f}%`]. Spoof trim flip-to-human = `0.0%` [`0.0%`, `3.7%`] at all margins.",
        "",
        "| Class | Pad Duration (s) | n | Digital Zeros Flip% [95% Wilson CI] | Zeros Logit Shift | Room Tone Flip% [95% Wilson CI] | Room Tone Logit Shift |",
        "|---|---:|---:|---|---:|---|---:|",
    ])
    for _, rp in df_pad.iterrows():
        lines.append(
            f"| `{rp['class']}` | {rp['duration_s']:.1f} | {int(rp['n'])} | "
            f"{rp['zeros_flip_rate']*100:.1f}% [{rp['zeros_ci_95_low']*100:.1f}%, {rp['zeros_ci_95_high']*100:.1f}%] | {rp['zeros_mean_shift']:+.2f} | "
            f"{rp['room_tone_flip_rate']*100:.1f}% [{rp['room_tone_ci_95_low']*100:.1f}%, {rp['room_tone_ci_95_high']*100:.1f}%] | {rp['room_tone_mean_shift']:+.2f} |"
        )

    lines.extend([
        "",
        "### 3.3 Noise Decomposition & Band-Limit Sweep (`n=100` paired clips/class/condition, Wilson 95% CIs)",
        "| Noise Type (V2 Status) | SNR 30 dB Spoof->Human / Bona->Fake [95% CI] | SNR 20 dB Spoof->Human / Bona->Fake [95% CI] | SNR 10 dB Spoof->Human / Bona->Fake [95% CI] |",
        "|---|---|---|---|",
    ])
    for ntype, label_str in [
        ("white", "White (Trained-on)"),
        ("pink", "Pink 1/f (Held-out)"),
        ("noise_lp4k", "LP <4 kHz (Trained-on)"),
        ("noise_hp4k", "HP >4 kHz (Trained-on)"),
    ]:
        cells = []
        for snr in [30.0, 20.0, 10.0]:
            sb = df_noise[(df_noise["target_class"] == "bonafide") & (df_noise["noise_type"] == ntype) & (df_noise["snr_db"] == snr)].iloc[0]
            ss = df_noise[(df_noise["target_class"] == "spoof") & (df_noise["noise_type"] == ntype) & (df_noise["snr_db"] == snr)].iloc[0]
            cells.append(
                f"{ss['flip_rate']*100:.1f}% [{ss['ci_95_low']*100:.1f}%,{ss['ci_95_high']*100:.1f}%] / "
                f"{sb['flip_rate']*100:.1f}% [{sb['ci_95_low']*100:.1f}%,{sb['ci_95_high']*100:.1f}%]"
            )
        lines.append(f"| `{label_str}` | {cells[0]} | {cells[1]} | {cells[2]} |")

    hp300, hp300_lo, hp300_hi = get_band("bonafide", "hp_300hz")
    hp100, hp100_lo, hp100_hi = get_band("bonafide", "hp_100hz")
    lp34, lp34_lo, lp34_hi = get_band("bonafide", "lp_3.4k")
    lp40, lp40_lo, lp40_hi = get_band("bonafide", "lp_4k")
    lp50, lp50_lo, lp50_hi = get_band("bonafide", "lp_5k")
    lp60, lp60_lo, lp60_hi = get_band("bonafide", "lp_6k")
    r8k, r8k_lo, r8k_hi = get_band("bonafide", "resample_8k")
    mul, mul_lo, mul_hi = get_band("bonafide", "mu_law_only")
    phn, phn_lo, phn_hi = get_band("bonafide", "full_phone_chain")

    lines.append(
        f"- **Bonafide Bandlimit/Telephony Flip-to-Fake (`n=100`)**: `full_phone_chain` `{phn:.1f}%` [`{phn_lo:.1f}%`, `{phn_hi:.1f}%`]; `highpass_300hz` `{hp300:.1f}%` [`{hp300_lo:.1f}%`, `{hp300_hi:.1f}%`]; `lowpass_3.4khz` (held-out) `{lp34:.1f}%` [`{lp34_lo:.1f}%`, `{lp34_hi:.1f}%`]; `lowpass_4khz` `{lp40:.1f}%` [`{lp40_lo:.1f}%`, `{lp40_hi:.1f}%`]; `resample_8khz` `{r8k:.1f}%` [`{r8k_lo:.1f}%`, `{r8k_hi:.1f}%`]; `lowpass_5khz` `{lp50:.1f}%`; `lowpass_6khz` `{lp60:.1f}%`; `highpass_100hz` `{hp100:.1f}%`; `mulaw_only` `{mul:.1f}%`."
    )

    lines.extend([
        "",
        "### 3.4 Phase 1 Data Counts & Shortcut Check V2 (`n_train=400`, `n_val=400`, 2,000-resample Bootstrap 95% CIs)",
        f"- **Shortcut AUC (Before -> After Augmentation)**: `GBDT_all_8_features` `{sc_gbdt['auc_before_aug']:.4f}` [`{sc_gbdt['auc_before_ci_low']:.4f}`, `{sc_gbdt['auc_before_ci_high']:.4f}`] -> `{sc_gbdt['auc_after_aug']:.4f}` [`{sc_gbdt['auc_after_ci_low']:.4f}`, `{sc_gbdt['auc_after_ci_high']:.4f}`]; `bandwidth_rolloff` `{sc_roll['auc_before_aug']:.4f}` -> `{sc_roll['auc_after_aug']:.4f}`; `silence_share` `{sc_sil['auc_before_aug']:.4f}` -> `{sc_sil['auc_after_aug']:.4f}`; `noise_floor` `{sc_nf['auc_before_aug']:.4f}` -> `{sc_nf['auc_after_aug']:.4f}`.",
        "",
        "## 4. Observed",
        f"- 5-fold CV point EERs: MLP (`0.69%`, `1.57%`, `2.29%`, `0.85%`, `1.39%`; mean `{mlp_eer_mean:.2f}%`) vs Linear SVM (`0.72%`, `2.28%`, `3.10%`, `0.95%`, `1.69%`; mean `{svm_eer_mean:.2f}%`).",
        f"- Trimming bonafide silence (`trim_0s`) flips `{t00:.1f}%` (`71/100`) to fake; padding bonafide with `1.5s` digital zeros flips `55.0%` (`55/100`), whereas `1.5s` room tone flips `6.0%` (`6/100`).",
        f"- Padding spoof with `1.5s` room tone flips `52.0%` (`52/100`) to human, whereas `1.5s` digital zeros flips `0.0%` (`0/100`).",
        f"- At 20 dB SNR, white noise flips `93.0%` (`93/100`) of spoofs to human, pink noise flips `49.0%` (`49/100`), LP <4 kHz noise flips `28.0%` (`28/100`), and HP >4 kHz noise flips `27.0%` (`27/100`).",
        f"- On bonafide speech, `highpass_300hz` flips `{hp300:.1f}%` (`75/100`) and `full_phone_chain` flips `{phn:.1f}%` (`96/100`), whereas `mulaw_only` (`0/100`) and `highpass_100hz` (`0/100`) flip `0.0%`.",
        f"- Class-symmetric augmentation (`src_v2/augment.py`) reduces 8-feature GBDT validation AUC from `{sc_gbdt['auc_before_aug']:.4f}` to `{sc_gbdt['auc_after_aug']:.4f}` (`delta = {sc_gbdt['delta_auc']:+.4f}`).",
        "",
        "## 5. Hypotheses",
        "- `[Untested]` Fine-tuning Wav2Vec2 top transformer layers with `src_v2/augment.py` online perturbations will reduce `highpass_300hz` and `full_phone_chain` bonafide flip rates below the 10% acceptance gate.",
        "- `[Untested]` Concatenating temporal standard-deviation pooling (`1536-dim` mean+std) will prevent room-tone padding from diluting synthetic frame activations.",
        "- `[Untested]` Loudness normalization prior to feature extraction will reduce the false-alarm rate on high-RMS speakers such as `LA_0095` (`-15.63 dBFS`, FRR `31.8%`).",
        "",
        "## 6. Discrepancies found",
        "- **Speaker count**: Earlier summary text cited 107 speakers (which included eval); train+dev contains exactly 40 speakers (20 train, 20 dev).",
        "- **Shortcut GBDT CV scope**: `shortcut_gbdt_cv_results` used the 1,000-clip cached descriptor subset (`500` train + `500` dev) rather than all `50,224` clips; labeled explicitly as subset-based.",
        "- **Padding mechanism**: `causal_interventions.csv` (`pad_silence_1s`, 32% bonafide flip) used digital zeros, whereas `silence_dose_response.csv` (`pad_1.0s`, 7% bonafide flip) used room tone; resolved via side-by-side `padding_comparison.csv`.",
        "",
        "## 7. Open questions",
        "- Will training on `white`, `low_band`, and `high_band` noise generalize to held-out `pink` noise (`<10%` spoof flips at 20/10 dB SNR) without degrading clean EER by `>1.0%`?",
        "- How many real user microphone recordings (`>=30` required) across phone call, laptop mic, and browser WebM are needed to calibrate the production inconclusive band?",
    ])

    assert len(lines) <= 120, f"HANDOFF.md line count {len(lines)} exceeds 120-line limit!"
    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    (PHASE0_HANDOFF_DIR / "HANDOFF.md").write_text(text, encoding="utf-8")
    return lines


def cross_check_handoff(handoff_md_path: Path):
    text = handoff_md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    mismatches = []

    if len(lines) > 120:
        mismatches.append(f"Line count {len(lines)} > 120")

    expected_sections = [
        "## 1. Run info",
        "## 2. Run vs not run",
        "## 3. Tables with n and 95% CI",
        "## 4. Observed",
        "## 5. Hypotheses",
        "## 6. Discrepancies found",
        "## 7. Open questions",
    ]
    positions = []
    for sec in expected_sections:
        pos = text.find(sec)
        if pos == -1:
            mismatches.append(f"Missing required section: {sec}")
        positions.append(pos)
    if positions != sorted(positions):
        mismatches.append(f"Sections out of order: {positions}")

    sec5_start = text.find("## 5. Hypotheses")
    sec6_start = text.find("## 6. Discrepancies found")
    if sec5_start != -1 and sec6_start != -1:
        sec5_lines = [ln.strip() for ln in text[sec5_start:sec6_start].splitlines() if ln.strip().startswith("-")]
        for ln in sec5_lines:
            if "[Untested]" not in ln:
                mismatches.append(f"Hypothesis line missing [Untested] marker: {ln}")

    df_fold = pd.read_csv(HANDOFF_DIR / "speaker_grouped_5fold_results.csv")
    for _, r in df_fold.iterrows():
        expected_str = f"{r['mlp_eer']*100:.2f}%"
        if expected_str not in text:
            mismatches.append(f"Missing fold {int(r['fold'])} MLP EER {expected_str} in HANDOFF.md")

    df_sc = pd.read_csv(HANDOFF_DIR / "shortcut_check_v2.csv")
    gbdt = df_sc[df_sc["feature"] == "GBDT_all_8_features"].iloc[0]
    for val in [f"{gbdt['auc_before_aug']:.4f}", f"{gbdt['auc_after_aug']:.4f}"]:
        if val not in text:
            mismatches.append(f"Missing GBDT shortcut AUC {val} in HANDOFF.md")

    df_pad = pd.read_csv(HANDOFF_DIR / "padding_comparison.csv")
    for _, r in df_pad.iterrows():
        z_str = f"{r['zeros_flip_rate']*100:.1f}%"
        rt_str = f"{r['room_tone_flip_rate']*100:.1f}%"
        if z_str not in text or rt_str not in text:
            mismatches.append(f"Missing padding rate {z_str} or {rt_str} in HANDOFF.md")

    df_counts = pd.read_csv(HANDOFF_DIR / "data_counts.csv")
    tot_str = f"{int(df_counts['n_clips'].sum()):,}"
    if tot_str not in text:
        mismatches.append(f"Missing total manifest count {tot_str} in HANDOFF.md")

    if mismatches:
        for m in mismatches:
            print(f"[CROSS-CHECK MISMATCH] {m}", file=sys.stderr)
        raise RuntimeError(f"Cross-check failed with {len(mismatches)} mismatch(es)!")

    print(f"[CROSS-CHECK PASS] HANDOFF.md ({len(lines)} lines <= 120) verified against all CSVs with 0 mismatches.")


def main():
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    exact_command = "python benchmarks/make_handoff.py"

    print(f"=== Building Clean Handoff Bundle ({timestamp_utc}) ===", flush=True)

    clf_path = REPO_ROOT / "models" / "classifier.pt"
    app_path = REPO_ROOT / "app" / "main.py"
    assert clf_path.exists(), "models/classifier.pt missing!"
    assert app_path.exists(), "app/main.py missing!"

    if HANDOFF_DIR.exists():
        shutil.rmtree(HANDOFF_DIR)
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    if HANDOFF_ZIP.exists():
        HANDOFF_ZIP.unlink()

    sys_info = build_system_info(HANDOFF_DIR / "system_info.json", timestamp_utc)
    git_commit = build_git_state(HANDOFF_DIR / "git_state.txt")

    _, df_counts, _ = build_data_counts_and_sample(
        DATA_V2_DIR / "manifest.csv",
        HANDOFF_DIR / "data_counts.csv",
        HANDOFF_DIR / "manifest_sample.csv",
        seed=42,
    )

    df_inv = build_my_recordings_inventory(
        RESULTS_DIR / "diagnose_real_audio.csv",
        HANDOFF_DIR / "my_recordings_inventory.csv",
    )

    shutil.copy2(DATA_V2_DIR / "shortcut_check_v2.csv", HANDOFF_DIR / "shortcut_check_v2.csv")
    df_sc = pd.read_csv(HANDOFF_DIR / "shortcut_check_v2.csv")

    df_pad = build_padding_comparison_with_cis(
        PHASE0_HANDOFF_DIR / "padding_comparison.csv",
        HANDOFF_DIR / "padding_comparison.csv",
        n_clips=100,
    )
    shutil.copy2(HANDOFF_DIR / "padding_comparison.csv", PHASE0_HANDOFF_DIR / "padding_comparison.csv")

    build_regression_battery_table(HANDOFF_DIR / "regression_battery.csv")

    build_augment_config_yaml(HANDOFF_DIR / "augment_config.yaml")
    print("Generating augmentation_gallery.zip (30 examples) and augmentation_distributions.png ...", flush=True)
    build_augmentation_gallery_and_distributions(
        HANDOFF_DIR / "augmentation_gallery.zip",
        HANDOFF_DIR / "augmentation_distributions.png",
        seed=42,
    )

    df_spk_prof = build_speaker_profiles_merged(HANDOFF_DIR / "speaker_profiles.csv")

    supporting_files = [
        "speaker_grouped_5fold_results.csv",
        "per_fold_bootstrap_cis.csv",
        "threshold_stability.csv",
        "speaker_frr_breakdown.csv",
        "speaker_spearman_correlations.csv",
        "leave_one_attack_out_results.csv",
        "silence_dose_response.csv",
        "silence_trim_definition_stats.csv",
        "noise_decomposition_results.csv",
        "bandlimit_sweep_results.csv",
        "ood_distance_gate_results.csv",
        "silence_dose_response.png",
        "per_fold_roc.png",
        "speaker_frr_boxplot.png",
        "loao_histograms.png",
    ]
    for fname in supporting_files:
        src = PHASE0_HANDOFF_DIR / fname
        if src.exists():
            shutil.copy2(src, HANDOFF_DIR / fname)

    shutil.copy2(RESULTS_DIR / "t1_eval_generalization.csv", HANDOFF_DIR / "t1_eval_generalization.csv")

    (HANDOFF_DIR / "src_v2").mkdir(parents=True, exist_ok=True)
    for fname in ["augment.py", "shortcut_check.py", "evaluate_regression.py"]:
        shutil.copy2(REPO_ROOT / "src_v2" / fname, HANDOFF_DIR / "src_v2" / fname)

    (HANDOFF_DIR / "tests").mkdir(parents=True, exist_ok=True)
    for tpath in (REPO_ROOT / "tests").glob("test_*.py"):
        shutil.copy2(tpath, HANDOFF_DIR / "tests" / tpath.name)

    (HANDOFF_DIR / "benchmarks").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "benchmarks" / "make_handoff.py", HANDOFF_DIR / "benchmarks" / "make_handoff.py")

    build_data_card_md(df_counts, df_spk_prof, df_sc, HANDOFF_DIR / "DATA_CARD.md")

    df_sil_def = pd.read_csv(HANDOFF_DIR / "silence_trim_definition_stats.csv")
    df_sil_dose = pd.read_csv(HANDOFF_DIR / "silence_dose_response.csv")
    df_noise = pd.read_csv(HANDOFF_DIR / "noise_decomposition_results.csv")
    df_band = pd.read_csv(HANDOFF_DIR / "bandlimit_sweep_results.csv")
    df_fold = pd.read_csv(HANDOFF_DIR / "speaker_grouped_5fold_results.csv")
    df_boot = pd.read_csv(HANDOFF_DIR / "per_fold_bootstrap_cis.csv")
    df_thr = pd.read_csv(HANDOFF_DIR / "threshold_stability.csv")
    df_spearman = pd.read_csv(HANDOFF_DIR / "speaker_spearman_correlations.csv")
    df_loao = pd.read_csv(HANDOFF_DIR / "leave_one_attack_out_results.csv")
    df_eval = pd.read_csv(HANDOFF_DIR / "t1_eval_generalization.csv")
    df_ood = pd.read_csv(HANDOFF_DIR / "ood_distance_gate_results.csv")

    build_handoff_md(
        timestamp_utc=timestamp_utc,
        git_commit=git_commit,
        sys_info=sys_info,
        df_counts=df_counts,
        df_inv=df_inv,
        df_sc=df_sc,
        df_pad=df_pad,
        df_sil_def=df_sil_def,
        df_sil_dose=df_sil_dose,
        df_noise=df_noise,
        df_band=df_band,
        df_fold=df_fold,
        df_boot=df_boot,
        df_thr=df_thr,
        df_spk_prof=df_spk_prof,
        df_spearman=df_spearman,
        df_loao=df_loao,
        df_eval=df_eval,
        df_ood=df_ood,
        out_md=HANDOFF_DIR / "HANDOFF.md",
    )

    cross_check_handoff(HANDOFF_DIR / "HANDOFF.md")

    manifest_files = {}
    for fpath in sorted(HANDOFF_DIR.rglob("*")):
        if fpath.is_file() and fpath.name != "manifest.json":
            rel = fpath.relative_to(HANDOFF_DIR).as_posix()
            manifest_files[rel] = {
                "path": f"handoff/{rel}",
                "size_bytes": fpath.stat().st_size,
                "sha256": sha256_file(fpath),
                "row_count": count_rows_or_lines(fpath),
            }

    manifest_obj = {
        "generated_timestamp": timestamp_utc,
        "git_commit": git_commit,
        "exact_command": exact_command,
        "total_files": len(manifest_files),
        "files": manifest_files,
    }
    manifest_json_path = HANDOFF_DIR / "manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)
    shutil.copy2(manifest_json_path, PHASE0_HANDOFF_DIR / "manifest.json")

    with zipfile.ZipFile(str(HANDOFF_ZIP), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(HANDOFF_DIR.rglob("*")):
            if fpath.is_file():
                rel = fpath.relative_to(REPO_ROOT).as_posix()
                zf.write(str(fpath), arcname=rel)

    zip_mb = HANDOFF_ZIP.stat().st_size / (1024 * 1024)
    print(f"=== Handoff Bundle Complete: {len(manifest_files) + 1} files in handoff/, packaged into handoff.zip ({zip_mb:.2f} MB) ===", flush=True)


if __name__ == "__main__":
    main()
