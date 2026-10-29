# DATA CARD — Deepfake Voice Detection Dataset Manifest

## 1. Provenance & Licensing
| Source ID | Dataset Description | Official Source / Provenance | License |
|---|---|---|---|
| `asvspoof2019_la` | ASVspoof 2019 Logical Access (train/dev/eval) | University of Edinburgh DataShare (`https://www.asvspoof.org`) | ODC-By v1.0 |
| `edge_tts` | Multi-accent neural TTS (`en-US`, `en-GB`, `en-IN`, `hi-IN`, `te-IN`) | Synthesized via `src_v2/generate_tts.py` (`edge-tts`) | MIT (library) / Microsoft TOS |
| `gtts` | Google Translate TTS (`en`) | Synthesized via `src_v2/generate_tts.py` (`gTTS`) | MIT (library) |
| `my_recordings` | 30 simulated microphone/channel clips derived from dev bona fide | Generated via `benchmarks/diagnose_real_audio.py` | Internal / ODC-By v1.0 derivative |
| `ood_spoof` | 140 out-of-domain TTS clips (`edge-tts` + `gTTS` phrases) | Generated via `benchmarks/t7_ood_generators.py` | Internal evaluation |

## 2. Split & Label Counts (from `data_counts.csv`)
- **Total Indexed Clips**: 121,689 (12,513 bonafide, 109,176 spoof)

| Source | Split | Bonafide | Spoof | Total |
|---|---|---:|---:|---:|
| `asvspoof2019_la` | `dev` | 2,548 | 22,296 | 24,844 |
| `asvspoof2019_la` | `heldout_eval` | 7,355 | 63,882 | 71,237 |
| `asvspoof2019_la` | `train` | 2,580 | 22,800 | 25,380 |
| `edge_tts` | `dev` | 0 | 18 | 18 |
| `edge_tts` | `heldout_eval` | 0 | 10 | 10 |
| `edge_tts` | `train` | 0 | 20 | 20 |
| `gtts` | `dev` | 0 | 10 | 10 |
| `my_recordings` | `heldout_eval` | 30 | 0 | 30 |
| `ood_spoof` | `heldout_eval` | 0 | 140 | 140 |

## 3. Speaker Diversity (from `speaker_profiles.csv`)
- **Train + Dev Speakers**: 40 disjoint speakers (20 train, 20 dev; 27 Male, 13 Female).
- **Mean Pitch (F0)**: Male = 122.21 Hz, Female = 171.73 Hz; Mean Loudness = -18.81 dBFS.

## 4. Trivial Feature Shortcut Audit (from `shortcut_check_v2.csv`)
- **Sample Size**: n_train = 400, n_val = 400 balanced clips (seed 42, 2,000-resample bootstrap CIs).

| Feature / Model | AUC Before Aug [95% CI] | AUC After Aug [95% CI] | Delta AUC | Status After Aug |
|---|---|---|---|---|
| `duration` | 0.5168 [0.4577, 0.5746] | 0.5364 [0.4787, 0.5911] | +0.0196 | PASS (<0.60) |
| `rms` | 0.5167 [0.4579, 0.5721] | 0.5118 [0.4548, 0.5702] | -0.0049 | PASS (<0.60) |
| `peak` | 0.5525 [0.5074, 0.5975] | 0.5083 [0.4525, 0.5673] | -0.0442 | PASS (<0.60) |
| `silence_share` | 0.7496 [0.7012, 0.7970] | 0.5492 [0.4909, 0.6020] | -0.2004 | PASS (<0.60) |
| `bandwidth_rolloff` | 0.8443 [0.8076, 0.8798] | 0.5969 [0.5400, 0.6522] | -0.2474 | PASS (<0.60) |
| `noise_floor` | 0.5670 [0.5110, 0.6236] | 0.5103 [0.4523, 0.5639] | -0.0568 | PASS (<0.60) |
| `sample_rate` | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] | +0.0000 | PASS (<0.60) |
| `codec_id` | 0.5000 [0.5000, 0.5000] | 0.5000 [0.5000, 0.5000] | +0.0000 | PASS (<0.60) |
| `GBDT_all_8_features` | 0.8357 [0.7953, 0.8718] | 0.5897 [0.5355, 0.6454] | -0.2460 | PASS (<0.60) |

## 5. Held-Out Evaluation Sets & Transforms (Never Used in Training)
- **Held-Out Splits**: `asvspoof2019_la` eval (71,237 clips, unseen attacks A07-A19), `edge_tts` voice `en-US-GuyNeural` (10 clips), `my_recordings` (30 simulated clips), `ood_spoof` (140 clips).
- **Held-Out Perturbations**: `pink_noise`, `lowpass_3400hz`, `mp3_32kbps`.
