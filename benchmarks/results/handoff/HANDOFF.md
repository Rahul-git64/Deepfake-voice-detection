# Deepfake Voice Detection — Phase 0 & Phase 1 Handoff Report

## 1. Run info
- Timestamp: `2026-10-29 10:39:24 UTC` | Git Commit: `7b5fd72247b3d2cd9793ef706d6e6f810f357045` | Command: `python benchmarks/make_handoff.py`
- System: Python `3.13.5`, PyTorch `2.13.0+cpu`, CUDA: `False`, CPU Cores: `16`, RAM: `15.35 GB`, Seed: `42`
- Untouched Production Files: `models/classifier.pt` (v1 768-256-64-2 MLP on mean-pooled `facebook/wav2vec2-base`, threshold `0.6402`) and `app/main.py`.
- Condition Definitions: `trim_silence` removes leading/trailing samples where `|x| < max(1e-5, 0.01*peak)` outside margin `m` (0.5/0.25/0.1/0s); `pad_zeros` appends exact `0.0`; `pad_room_tone` appends Gaussian noise `sigma = max(1e-6, 0.005*RMS)`; Filters use 4th-order Butterworth SOS (HP 100/300 Hz, LP 3.4/4/5/6 kHz); `phone_chain` = BP 300-3400 Hz + 8 kHz resample + 8-bit G.711 mu-law (256 levels); Noise = white, pink (1/f), LP <4 kHz, HP >4 kHz at SNR 30/20/10 dB.

## 2. Run vs not run
- **Run**: 5-fold speaker-disjoint CV (`n=50,224` across 40 speakers, 2,000-resample bootstrap CIs); LOAO A01-A06; v1 perturbation battery (`n=100` paired clips/class/condition: silence trim, zeros vs room-tone pad, 4x3 noise grid, 9 bandlimit/telephony conditions); 40-speaker F0/gender/loudness profiling & Holm-corrected Spearman; Phase 1 manifest (`n=121,689`), 58 local TTS clips (`en-US/GB/IN`, `hi-IN`, `te-IN`, `gTTS`), 30-clip augmentation gallery, and v2 GBDT shortcut check (`n=400` train, `n=400` val).
- **Not run**: (1) Real-microphone CV (`my_recordings` has `0` real user clips and `30` simulated dev clips; skipped per `<30` real clips rule). (2) Model v2-lite training (`train_config.json`, `training_curves`, `scores_heldout.csv`, v1-vs-v2 eval table not generated because Phase 1 halted before GPU training per protocol stop rule and `models/` is strictly untouched). (3) XTTS-v2/Bark synthesis (generated as Colab notebook `notebooks/generate_tts_colab.ipynb`).

## 3. Tables with n and 95% CI
### 3.1 Speaker-Grouped 5-Fold CV (`n=50,224` clips: `5,128` bonafide, `45,096` spoof; 40 speakers; 2,000-resample bootstrap 95% CIs)
| Fold | n_bona / n_spoof | MLP EER% (Pt / Boot [95% CI]) | MLP AUC [95% CI] | Linear SVM EER% (Pt / Boot [95% CI]) | LogReg EER% (Pt) | GBDT Shortcut AUC (n=1,000 subset) |
|---|---:|---|---|---|---|---|
| 1 | 821 / 9492 | 0.69% / 0.76% [0.46%, 1.19%] | 0.9988 [0.9967, 0.9999] | 0.72% / 0.67% [0.37%, 0.97%] | 1.23% | 0.8739 |
| 2 | 1064 / 8916 | 1.57% / 1.70% [1.21%, 2.23%] | 0.9985 [0.9978, 0.9991] | 2.28% / 2.31% [1.93%, 2.73%] | 4.62% | 0.9184 |
| 3 | 1092 / 8916 | 2.29% / 2.25% [1.82%, 2.67%] | 0.9978 [0.9970, 0.9984] | 3.10% / 3.12% [2.57%, 3.76%] | 3.73% | 0.9001 |
| 4 | 1015 / 8916 | 0.85% / 0.87% [0.53%, 1.19%] | 0.9997 [0.9995, 0.9998] | 0.95% / 0.84% [0.40%, 1.28%] | 1.09% | 0.9169 |
| 5 | 1136 / 8856 | 1.39% / 1.36% [1.02%, 1.68%] | 0.9993 [0.9991, 0.9996] | 1.69% / 1.68% [1.27%, 2.13%] | 2.64% | 0.8863 |
- **5-Fold Point Summary**: MLP EER = `1.36% +/- 0.64%`; Linear SVM EER = `1.75% +/- 0.98%`; LogReg EER = `2.66% +/- 1.54%`; Subset GBDT AUC = `0.8991 +/- 0.0193`.
- **Operating Point (`0.6402`)**: Pooled FAR (spoof accepted) = `0.76%`, FRR (bonafide rejected) = `2.79%`. Worst-5 FRR speakers: LA_0095 (31.8%, n=132, F0=102.8Hz, -15.6dBFS), LA_0099 (24.7%, n=77, F0=89.6Hz, -18.8dBFS), LA_0100 (15.6%, n=77, F0=108.3Hz, -20.0dBFS), LA_0096 (13.6%, n=132, F0=94.3Hz, -19.3dBFS), LA_0101 (13.0%, n=77, F0=102.5Hz, -17.5dBFS).
- **Speaker FRR Spearman (`n=40`)**: `rolloff_99` rho=+0.187 (p=0.249, Holm p=1.000); `rolloff_95` rho=+0.118 (p=0.470, Holm p=1.000); `noise_floor` rho=+0.141 (p=0.384, Holm p=1.000); `leading_silence` rho=-0.159 (p=0.328, Holm p=1.000); `trailing_silence` rho=+0.401 (p=0.010, Holm p=0.093).
- **Leave-One-Attack-Out (A01-A06)**: `A01` (n=3716/2548): EER `6.74%`, AUC `0.9824`, Det@val `62.4%`, Det@oracle `93.3%`; `A02` (n=3716/2548): EER `6.98%`, AUC `0.9845`, Det@val `69.5%`, Det@oracle `93.0%`; `A03` (n=3716/2548): EER `1.44%`, AUC `0.9987`, Det@val `94.6%`, Det@oracle `98.6%`; `A04` (n=3716/2548): EER `18.86%`, AUC `0.8867`, Det@val `18.4%`, Det@oracle `81.2%`; `A05` (n=3716/2548): EER `0.43%`, AUC `0.9999`, Det@val `99.7%`, Det@oracle `99.6%`; `A06` (n=3716/2548): EER `38.97%`, AUC `0.6730`, Det@val `3.5%`, Det@oracle `61.0%`.
- **ASVspoof Eval Subset (`n=300`: `105` bonafide, `195` spoof)**: EER `5.68%`, AUC `0.9855`, FRR@0.6402 `0.95%` [`0.17%`, `5.20%`], FAR@0.6402 `14.36%` [`10.13%`, `19.97%`]. **OOD Distance Gate (`tau_99=31.91`)**: `dev_bonafide` (n=2548): `1.0%` rejected; `pstn` (n=20): `100.0%` rejected; `noise` (n=20): `100.0%` rejected; `edge` (n=20): `35.0%` rejected; `my_recordings_simulated` (n=30): `80.0%` rejected.

### 3.2 Silence Trim & Padding Dose-Response (`n=100` paired clips/class, Wilson 95% CIs)
- **Pre-Trim Silence (`n=100`/class)**: Bonafide leading `1.146s`, trailing `0.451s`; Spoof leading `0.630s`, trailing `0.427s` (`0.000s` leading after `trim_0s`).
- **Trimming Bonafide Flip-to-Fake**: `trim_0.5s` = `17.0%` [`10.9%`, `25.5%`]; `trim_0.25s` = `43.0%` [`33.7%`, `52.8%`]; `trim_0.1s` = `66.0%` [`56.3%`, `74.5%`]; `trim_0s` = `71.0%` [`61.5%`, `79.0%`]. Spoof trim flip-to-human = `0.0%` [`0.0%`, `3.7%`] at all margins.

| Class | Pad Duration (s) | n | Digital Zeros Flip% [95% Wilson CI] | Zeros Logit Shift | Room Tone Flip% [95% Wilson CI] | Room Tone Logit Shift |
|---|---:|---:|---|---:|---|---:|
| `bonafide` | 0.5 | 100 | 12.0% [7.0%, 19.8%] | +6.05 | 3.0% [1.0%, 8.5%] | +1.98 |
| `bonafide` | 1.0 | 100 | 32.0% [23.7%, 41.7%] | +8.91 | 7.0% [3.4%, 13.8%] | +4.77 |
| `bonafide` | 1.5 | 100 | 55.0% [45.2%, 64.4%] | +11.28 | 6.0% [2.8%, 12.5%] | +5.16 |
| `spoof` | 0.5 | 100 | 0.0% [0.0%, 3.7%] | -0.36 | 26.0% [18.4%, 35.4%] | -8.18 |
| `spoof` | 1.0 | 100 | 0.0% [0.0%, 3.7%] | -1.76 | 40.0% [30.9%, 49.8%] | -10.61 |
| `spoof` | 1.5 | 100 | 0.0% [0.0%, 3.7%] | -2.78 | 52.0% [42.3%, 61.5%] | -12.39 |

### 3.3 Noise Decomposition & Band-Limit Sweep (`n=100` paired clips/class/condition, Wilson 95% CIs)
| Noise Type (V2 Status) | SNR 30 dB Spoof->Human / Bona->Fake [95% CI] | SNR 20 dB Spoof->Human / Bona->Fake [95% CI] | SNR 10 dB Spoof->Human / Bona->Fake [95% CI] |
|---|---|---|---|
| `White (Trained-on)` | 47.0% [37.5%,56.7%] / 7.0% [3.4%,13.7%] | 93.0% [86.3%,96.6%] / 0.0% [0.0%,3.7%] | 96.0% [90.2%,98.4%] / 0.0% [0.0%,3.7%] |
| `Pink 1/f (Held-out)` | 27.0% [19.3%,36.4%] / 21.0% [14.2%,30.0%] | 49.0% [39.4%,58.7%] / 19.0% [12.5%,27.8%] | 96.0% [90.2%,98.4%] / 0.0% [0.0%,3.7%] |
| `LP <4 kHz (Trained-on)` | 20.0% [13.3%,28.9%] / 60.0% [50.2%,69.1%] | 28.0% [20.1%,37.5%] / 62.0% [52.2%,70.9%] | 93.0% [86.3%,96.6%] / 1.0% [0.2%,5.4%] |
| `HP >4 kHz (Trained-on)` | 27.0% [19.3%,36.4%] / 0.0% [0.0%,3.7%] | 27.0% [19.3%,36.4%] / 7.0% [3.4%,13.7%] | 45.0% [35.6%,54.8%] / 6.0% [2.8%,12.5%] |
- **Bonafide Bandlimit/Telephony Flip-to-Fake (`n=100`)**: `full_phone_chain` `96.0%` [`90.2%`, `98.4%`]; `highpass_300hz` `75.0%` [`65.7%`, `82.5%`]; `lowpass_3.4khz` (held-out) `17.0%` [`10.9%`, `25.5%`]; `lowpass_4khz` `4.0%` [`1.6%`, `9.8%`]; `resample_8khz` `7.0%` [`3.4%`, `13.7%`]; `lowpass_5khz` `2.0%`; `lowpass_6khz` `1.0%`; `highpass_100hz` `0.0%`; `mulaw_only` `0.0%`.

### 3.4 Phase 1 Data Counts & Shortcut Check V2 (`n_train=400`, `n_val=400`, 2,000-resample Bootstrap 95% CIs)
- **Shortcut AUC (Before -> After Augmentation)**: `GBDT_all_8_features` `0.8357` [`0.7953`, `0.8718`] -> `0.5897` [`0.5355`, `0.6454`]; `bandwidth_rolloff` `0.8443` -> `0.5969`; `silence_share` `0.7496` -> `0.5492`; `noise_floor` `0.5670` -> `0.5103`.

## 4. Observed
- 5-fold CV point EERs: MLP (`0.69%`, `1.57%`, `2.29%`, `0.85%`, `1.39%`; mean `1.36%`) vs Linear SVM (`0.72%`, `2.28%`, `3.10%`, `0.95%`, `1.69%`; mean `1.75%`).
- Trimming bonafide silence (`trim_0s`) flips `71.0%` (`71/100`) to fake; padding bonafide with `1.5s` digital zeros flips `55.0%` (`55/100`), whereas `1.5s` room tone flips `6.0%` (`6/100`).
- Padding spoof with `1.5s` room tone flips `52.0%` (`52/100`) to human, whereas `1.5s` digital zeros flips `0.0%` (`0/100`).
- At 20 dB SNR, white noise flips `93.0%` (`93/100`) of spoofs to human, pink noise flips `49.0%` (`49/100`), LP <4 kHz noise flips `28.0%` (`28/100`), and HP >4 kHz noise flips `27.0%` (`27/100`).
- On bonafide speech, `highpass_300hz` flips `75.0%` (`75/100`) and `full_phone_chain` flips `96.0%` (`96/100`), whereas `mulaw_only` (`0/100`) and `highpass_100hz` (`0/100`) flip `0.0%`.
- Class-symmetric augmentation (`src_v2/augment.py`) reduces 8-feature GBDT validation AUC from `0.8357` to `0.5897` (`delta = -0.2460`).

## 5. Hypotheses
- `[Untested]` Fine-tuning Wav2Vec2 top transformer layers with `src_v2/augment.py` online perturbations will reduce `highpass_300hz` and `full_phone_chain` bonafide flip rates below the 10% acceptance gate.
- `[Untested]` Concatenating temporal standard-deviation pooling (`1536-dim` mean+std) will prevent room-tone padding from diluting synthetic frame activations.
- `[Untested]` Loudness normalization prior to feature extraction will reduce the false-alarm rate on high-RMS speakers such as `LA_0095` (`-15.63 dBFS`, FRR `31.8%`).

## 6. Discrepancies found
- **Speaker count**: Earlier summary text cited 107 speakers (which included eval); train+dev contains exactly 40 speakers (20 train, 20 dev).
- **Shortcut GBDT CV scope**: `shortcut_gbdt_cv_results` used the 1,000-clip cached descriptor subset (`500` train + `500` dev) rather than all `50,224` clips; labeled explicitly as subset-based.
- **Padding mechanism**: `causal_interventions.csv` (`pad_silence_1s`, 32% bonafide flip) used digital zeros, whereas `silence_dose_response.csv` (`pad_1.0s`, 7% bonafide flip) used room tone; resolved via side-by-side `padding_comparison.csv`.

## 7. Open questions
- Will training on `white`, `low_band`, and `high_band` noise generalize to held-out `pink` noise (`<10%` spoof flips at 20/10 dB SNR) without degrading clean EER by `>1.0%`?
- How many real user microphone recordings (`>=30` required) across phone call, laptop mic, and browser WebM are needed to calibrate the production inconclusive band?
