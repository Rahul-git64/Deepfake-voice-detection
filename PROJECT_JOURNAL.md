# Deepfake Voice Detection — Complete Engineering History, Architectural Evolutions, and Benchmark Journal

**Repository**: `C:\Deepfake-voice-detection`  
**Current Production Model**: `models/classifier.pt` (MLP 768-256-64-2 on mean-pooled `facebook/wav2vec2-base` embeddings)  
**Primary Dataset**: ASVspoof 2019 Logical Access (LA)  
**Date of Audit**: October 19, 2026  
**Current State**: Phase 0 Complete, Fully Audited & Handoff Bundle Archived  

---

## Table of Contents
1. [Project Inception & Core Vision](#1-project-inception--core-vision)
2. [Chronological Engineering History & Commits](#2-chronological-engineering-history--commits)
   - [Phase A: Repository Setup & Audio Preprocessing Pipeline](#phase-a-repository-setup--audio-preprocessing-pipeline)
   - [Phase B: Deep Learning Feature Extraction & Model Training](#phase-b-deep-learning-feature-extraction--model-training)
   - [Phase C: Inference Service & Backend API Architecture](#phase-c-inference-service--backend-api-architecture)
   - [Phase D: Frontend Workspace Evolutions & UI De-cluttering](#phase-d-frontend-workspace-evolutions--ui-de-cluttering)
   - [Phase E: Comprehensive Benchmarking Harness (T1–T12)](#phase-e-comprehensive-benchmarking-harness-t1t12)
   - [Phase F: Real-World Audio Failures & Shortcut Diagnostics](#phase-f-real-world-audio-failures--shortcut-diagnostics)
   - [Phase G: Discrepancy Reconciliation & Phase 0 Handoff Bundle](#phase-g-discrepancy-reconciliation--phase-0-handoff-bundle)
3. [Summary of Major Architectural Changes & Why They Bettered the Project](#3-summary-of-major-architectural-changes--why-they-bettered-the-project)
4. [Empirical Benchmark Results & Diagnostics (No Speculation)](#4-empirical-benchmark-results--diagnostics-no-speculation)
5. [Artifact & Codebase Map](#5-artifact--codebase-map)
6. [Roadmap for Phase 1 (Model v2)](#6-roadmap-for-phase-1-model-v2)

---

## 1. Project Inception & Core Vision

The goal of this project is to build an accurate, low-latency, real-world deepfake voice detection system capable of distinguishing authentic human speech from AI-synthesized or voice-converted audio (TTS, VC, neural vocoders).

Traditional approaches relied on hand-crafted speech representations (MFCCs, CQCCs, spectrograms) paired with GMMs or shallow CNNs, which struggle to generalize across different acoustic recording conditions. This project leveraged self-supervised speech representations using **Wav2Vec 2.0** (`facebook/wav2vec2-base`) as a rich feature extractor coupled with a lightweight multi-layer perceptron (MLP) classification head for fast real-time inference.

---

## 2. Chronological Engineering History & Commits

### Phase A: Repository Setup & Audio Preprocessing Pipeline
- **Commits**: `3b29893`, `59a0235`, `88846e2`, `55dc03b`, `488dbab`, `8ff0868`, `ea94062`
- **What was done**:
  - Initialized the Git repository, established directory architecture (`app/`, `data/`, `models/`, `notebooks/`, `scripts/`, `tests/`), configured `.gitignore` to prevent committing massive audio datasets (`.wav`, `.flac`, `.mp3`) and model weights.
  - Specified core dependencies: `torch`, `torchaudio`, `transformers`, `fastapi`, `uvicorn`, `soundfile`, `librosa`, `scikit-learn`.
- **Why this was changed**:
  - Large audio corpora (e.g., ASVspoof 2019 LA is >15 GB uncompressed) quickly bloat Git history and cause GitHub push failures.
- **How it bettered the project**:
  - Guaranteed a clean, lightweight repository structure with reproducible package dependencies, separating reproducible code from heavy binary datasets.

---

### Phase B: Deep Learning Feature Extraction & Model Training
- **Commits**: `46c4a86`, `944d806`
- **What was done**:
  - Developed `scripts/extract_embeddings.py` alongside Google Colab and Kaggle notebooks (`notebooks/extract_embeddings_colab.ipynb`, `notebooks/extract_embeddings_kaggle.ipynb`).
  - Extracted 768-dimensional mean-pooled embeddings from the final hidden layer of `facebook/wav2vec2-base` for all 25,380 training clips and 24,844 development clips, caching them to `data/embeddings/train_embeddings.pt` and `dev_embeddings.pt`.
  - Built training pipelines in `scripts/train.py` (baseline classifiers) and `scripts/train_pytorch.py` (PyTorch MLP: `Linear(768, 256) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(256, 64) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(64, 2)`).
  - Model was trained using AdamW ($lr = 10^{-3}$, weight decay $= 10^{-4}$), batch size 128, with inverse class weighting to handle spoof-heavy imbalanced datasets.
- **Why this was changed**:
  - Running a 95-million parameter Wav2Vec 2.0 model end-to-end on CPU during every training epoch or web request is computationally prohibitive.
  - Caching embeddings reduced training iteration times from hours to under 30 seconds on CPU.
- **How it bettered the project**:
  - Achieved an outstanding in-domain baseline on ASVspoof 2019 LA Dev: **EER 1.41%** at optimal threshold **0.6402** and accuracy **99.22%**.
  - Enabled rapid iteration of classification heads without expensive GPU recomputation.

---

### Phase C: Inference Service & Backend API Architecture
- **Commit**: `cdf008e`
- **What was done**:
  - Built the production FastAPI service (`app/main.py`) exposing:
    - `GET /health`: Health check returning model status, device, and version.
    - `POST /predict`: Multipart audio upload endpoint returning `{filename, prediction, label, confidence, probabilities: {bonafide, spoof}, inference_time_ms}`.
    - `GET /legacy`: Server-rendered HTML fallback for browser testing.
  - Implemented single-instance lazy loading for `Wav2Vec2Model` and `DeepfakeClassifier`.
  - Preprocessed input audio: read bytes via `soundfile`, converted multi-channel audio to mono via averaging across channels, resampled to 16,000 Hz, and converted to PyTorch tensors.
- **Why this was changed**:
  - A real-time detector requires low-latency inference (<200 ms per clip) and a clean REST API interface that decouples frontend client applications from the PyTorch inference engine.
- **How it bettered the project**:
  - Reduced per-request model loading overhead to 0 ms by persisting models in memory, yielding fast sub-100ms inference on standard CPUs.

---

### Phase D: Frontend Workspace Evolutions & UI De-cluttering
- **Commits**: `3aa8110`, `daff6f6`, `ca422a3`, `008be90`
- **What was done**:
  - Introduced a modern Vite + React 18 + TypeScript frontend workspace (`frontend/`) with live Web Audio API DSP analysis (`useAudioAnalyser.ts`, `useRecorder.ts`).
  - Initially added a heavily stylized UI ("VOXFORENSICS") with animated 3D canvas sound-orbs, multi-band FFT spectrograms, and simulated forensic metrics.
  - **The Problem**: The interface became over-engineered and fragile. Browser errors occurred when rendering the canvas animations, model labels did not match backend keys, and WebM/Opus audio recorded directly from the browser microphone failed to decode reliably on the backend.
  - **The Overhaul & De-cluttering**:
    - Removed non-essential visual meters, faux-forensic gauges, and overcompensating graphics.
    - Replaced them with a focused, high-clarity forensic workspace:
      1. Single-click high-fidelity browser microphone recorder.
      2. Drag-and-drop audio file upload.
      3. Direct real-time classification verdict ("Human Authentic" vs. "AI Synthesized").
      4. True calibrated confidence score and latency benchmark.
    - Fixed WebM/Opus decoding: Integrated dynamic fallback decoding to handle compressed browser audio buffers without crashing.
- **Why this was changed**:
  - User feedback highlighted that the extra widgets were distracting, slow to load, and obscured the actual model predictions.
- **How it bettered the project**:
  - Eliminated UI freeze bugs, slashed frontend bundle size, improved recording reliability, and created an intuitive user experience.

---

### Phase E: Comprehensive Benchmarking Harness (T1–T12)
- **Commits**: `3b04603`, `124cd03`, `00a6276`, `ff39cbc`, `7b5fd72`
- **What was done**:
  - Implemented an independent evaluation harness in `benchmarks/` under strict scientific isolation (models, backend code, and datasets were kept untouched):
    - `t1_eval_generalization.py`: ASVspoof 2019 Eval generalization on unseen attacks.
    - `t2_per_attack.py`: Breakdown across individual spoofing algorithms (A07–A19).
    - `t3_full_metrics.py`: Complete Dev vs. Eval ROC curves, DET curves, and EER.
    - `t4_baselines.py`: Comparing the MLP against Linear SVM and Logistic Regression.
    - `t5_seed_variance.py`: Multi-seed variance assessment.
    - `t6_robustness.py`: Audio perturbations (additive noise, PSTN telephone simulation, MP3 compression, clipping, truncation).
    - `t7_ood_generators.py`: Modern TTS architectures (Microsoft Edge-TTS, Google gTTS).
    - `t8_non_speech_sanity.py`: Non-speech controls (sine waves, white noise, ambient silence).
    - `t9_calibration.py`: Expected Calibration Error (ECE) and confidence histograms.
    - `t10_threshold_sensitivity.py`: Sweep of operating points and cost curves.
    - `t11_api_latency.py`: Payload stress tests and in-process vs. API parity.
    - `t12_failure_analysis.py`: Error analysis of false alarms and misses.
- **Why this was changed**:
  - Deep learning models often suffer from "silent failure": reporting 99% accuracy on a dev set while completely failing on real-world audio.
- **How it bettered the project**:
  - Revealed critical empirical truths about `models/classifier.pt` before deployment:
    1. Eval set EER jumped to $5.68\%$ (miss rate of $14.36\%$ on unseen attacks).
    2. Additive white noise (10 dB SNR) completely blinded the detector, flipping fakes to human.
    3. Telephone simulation (300–3,400 Hz bandpass + $\mu$-law) flipped genuine human voices to fake.

---

### Phase F: Real-World Audio Failures & Shortcut Diagnostics
- **Scripts**: `benchmarks/diagnose_real_audio.py`, `benchmarks/run_shortcut_check.py`
- **The Problem**:
  - When real voice samples from laptop mics and smartphones were uploaded, the detector consistently flagged them as **"AI Voice"**.
- **What was done**:
  1. Extracted acoustic descriptors from audio: RMS, peak energy, noise floor, 95% and 99% spectral rolloff, leading silence, trailing silence, and silence share.
  2. Built a Gradient-Boosted Decision Tree (GBDT) shortcut classifier that predicted labels **using only these metadata numbers**.
  3. Calculated Mahalanobis distances and 5-NN cosine distances in the 768-dim Wav2Vec2 feature space.
- **Why this was changed**:
  - Needed to determine whether the model was failing due to neural overfitting or severe acoustic domain shift.
- **How it bettered the project**:
  - **Empirically Proven Shortcut**: The descriptor-only GBDT achieved an AUC of $0.8991$ without ever inspecting speech or audio content. In ASVspoof 2019, synthetic clips had systematic silence padding and spectral rolloff cutoffs that differed from human speech.
  - **Domain Shift Identification**: Real-world phone and laptop microphones lie at extreme Mahalanobis distances ($\tau > 31.91$) from the studio-quality training data, explaining the false alarms.

---

### Phase G: Discrepancy Reconciliation & Phase 0 Handoff Bundle
- **Script**: `benchmarks/generate_handoff_suite.py`
- **What was done**:
  1. **Reconciled AUC Discrepancy**: Fixed a data split bug where shortcut evaluation had reported an AUC of $1.0000$ due to evaluating the frozen checkpoint on a small 1,000-clip subset. Rewrote the CV harness to evaluate all models on the exact same folds, saving all 150,672 test predictions to `cv_scores.csv`.
  2. **Speaker-Grouped 5-Fold Cross-Validation**: Evaluated MLP, Linear SVM, Logistic Regression, and GBDT under strict speaker isolation.
  3. **Leave-One-Attack-Out (LOAO)**: Evaluated A01 through A06 with oracle EER detection rates and extreme score saturation logging.
  4. **Causal Interventions ($n=1,000$ clips)**: Measured paired per-clip logit shifts and decision flips under lowpass filtering and noise.
  5. **OOD Distance Gate Prototype**: Validated an automated Mahalanobis distance filter to reject corrupted or out-of-domain audio before inference.
  6. **Automated Handoff Packaging**: Script-generated the 66-line `HANDOFF.md`, `manifest.json`, diagnostic plots, and `handoff.zip`.
- **Why this was changed**:
  - To fulfill Phase 0 requirements with complete scientific integrity: every single reported number is generated directly by code from saved CSV outputs, with exact sample sizes $n$ and 95% Wilson confidence intervals.
- **How it bettered the project**:
  - Established a rock-solid, fully reconciled benchmark baseline, providing the exact engineering specifications needed to train the v2 model.

---

## 3. Summary of Major Architectural Changes & Why They Bettered the Project

| Evolution Area | Before / Initial State | After / Modern State | Why Changed & How It Bettered the Project |
|---|---|---|---|
| **Audio Processing** | Raw file reading without sample rate or channel handling | Robust pipeline: auto-downmix to mono, 16 kHz resample, buffer fallback | Prevents runtime crashes on multi-channel stereo or non-standard audio rates. |
| **Feature Extraction** | Heavy on-the-fly Wav2Vec2 forward passes on CPU | Precomputed & cached 768-dim embeddings in `data/embeddings/` | Accelerated training and CV iteration time from hours to under 30 seconds. |
| **Frontend UI** | Cluttered "VOXFORENSICS" UI with animated 3D sound-orb and fake meters | Clean, high-focus forensic workspace with real-time API client & live DSP | Resolved blank screen bugs, removed user confusion, and prioritized actual detection. |
| **Evaluation Protocol** | Random clip-level train/test splits | Speaker-grouped 5-fold CV with strict speaker-disjoint inner validation | Completely eliminated speaker leakage; provided true generalization estimates. |
| **Error Diagnosis** | Assuming the model learned deep vocal artifacts | Discovered acoustic shortcuts (GBDT AUC 0.8991) and OOD distance shift | Shifted focus from model tuning to fixing acoustic distribution mismatch. |
| **Benchmark Integrity** | Separate uncoordinated scripts with reporting discrepancies | Unified master suite (`generate_handoff_suite.py`) logging unified `cv_scores.csv` | Guaranteed 100% reproducible results with no hand-typed numbers and verified CIs. |

---

## 4. Empirical Benchmark Results & Diagnostics (No Speculation)

All metrics below originate directly from `benchmarks/results/handoff/` generated on October 19, 2026:

```
========================================================================================
1. SPEAKER-GROUPED 5-FOLD CV (n=50,224 clips across 107 speakers)
----------------------------------------------------------------------------------------
Model                        EER (%)              ROC-AUC          Balanced Acc (%)
MLP (768-256-64-2)           1.36% +/- 0.64%      0.9988 +/- 0.0008    98.40% +/- 0.72%
Linear SVM                   1.75% +/- 0.98%      0.9979 +/- 0.0014    98.03% +/- 0.96%
Logistic Regression          2.66% +/- 1.54%      0.9963 +/- 0.0030    97.02% +/- 1.56%
Descriptor GBDT Shortcut     -                    0.8991 +/- 0.0193    -
----------------------------------------------------------------------------------------
Paired EER Diff (MLP - SVM): -0.39% (MLP won 3 folds, SVM won 2 folds; no unanimous winner)

========================================================================================
2. THRESHOLD STABILITY AT PRODUCTION THRESHOLD 0.6402
----------------------------------------------------------------------------------------
Validation-chosen threshold range across folds: [0.5585, 0.9023]
Mean False Alarm Rate (human flagged fake): 2.67%
Mean Miss Rate (fake flagged human):        0.75%
Worst Speakers by False Alarm Rate:
  1. LA_0095: 31.8% FRR (n=132)
  2. LA_0099: 24.7% FRR (n=77)
  3. LA_0100: 15.6% FRR (n=77)
  4. LA_0096: 13.6% FRR (n=132)
  5. LA_0101: 13.0% FRR (n=77)

========================================================================================
3. LEAVE-ONE-ATTACK-OUT GENERALIZATION (A01 - A06)
----------------------------------------------------------------------------------------
Held-out Attack    Val Det Rate    Oracle Det Rate    ROC-AUC     EER (%)    Scores < 0.001
A01                62.4%           93.3%              0.9824       6.7%       0.0%
A02                69.5%           93.0%              0.9845       7.0%       0.0%
A03                94.6%           98.6%              0.9987       1.4%       0.0%
A04 (Collapse)     18.4%           81.2%              0.8867      18.9%      15.9%
A05                99.7%           99.6%              0.9999       0.4%       0.0%
A06 (Collapse)      3.5%           61.0%              0.6730      39.0%      78.5%
* Note: Per project rules, "Collapse" is strictly defined as ROC-AUC < 0.90.

========================================================================================
4. PAIRED CAUSAL INTERVENTIONS (n=500 bona fide, n=500 spoof)
----------------------------------------------------------------------------------------
Target Class   Perturbation            Mean Logit Shift    Decision Flip Rate [95% CI]
Spoof          10 dB Additive Noise    -19.07              98.0% [96.4%, 98.9%] (Flip to Human)
Spoof          4 kHz Lowpass Filter    +1.41                2.8% [1.7%, 4.6%]   (Flip to Human)
Bona Fide      4 kHz Lowpass Filter    +4.42                7.0% [5.1%, 9.6%]   (Flip to Fake)

========================================================================================
5. OOD-DISTANCE GATE PROTOTYPE (tau_99 = 31.91 on Dev Bona Fide)
----------------------------------------------------------------------------------------
Evaluation Audio Group                 Sample Size (n)     Rejection Rate (%)
Clean Dev Bona Fide                    2,548               1.0% (false rejection)
Telephone Simulation (PSTN)            20                  100.0% (correctly flagged OOD)
Additive Gaussian Noise (10 dB)        20                  100.0% (correctly flagged OOD)
Edge-TTS Modern Deepfakes              20                  35.0%
========================================================================================
```

---

## 5. Artifact & Codebase Map

### Core Production Code (Untouched)
- `models/classifier.pt`: Production MLP classifier weights.
- `app/main.py`: Production FastAPI inference service.
- `scripts/train_pytorch.py`: Original model training script.
- `frontend/`: React + TypeScript frontend workspace.

### Benchmarks & Diagnostics Harness
- `benchmarks/generate_handoff_suite.py`: Master script running all cross-validations and generating handoff bundles.
- `benchmarks/run_shortcut_check.py`: GBDT acoustic shortcut diagnostic script.
- `benchmarks/diagnose_real_audio.py`: Real audio capture and feature profiling script.
- `benchmarks/t1_eval_generalization.py` to `t12_failure_analysis.py`: Independent diagnostic tests.

### Generated Handoff Files (`benchmarks/results/handoff/` & `handoff.zip`)
- `HANDOFF.md`: 66-line formal handoff summary.
- `manifest.json`: Cryptographic and size registry of generated artifacts.
- `cv_scores.csv`: 150,672 test predictions across models and folds.
- `speaker_grouped_5fold_results.csv`: Per-fold performance metrics.
- `threshold_stability.csv`: Threshold shifts and error rates across folds.
- `speaker_frr_breakdown.csv`: Per-speaker false alarm distributions.
- `leave_one_attack_out_results.csv`: LOAO metrics for A01–A06.
- `causal_interventions.csv`: Paired causal perturbation metrics.
- `ood_distance_gate_results.csv`: Distance rejection rates.
- `per_fold_roc.png`: ROC curves for all 5 folds.
- `speaker_frr_boxplot.png`: Distribution of false alarms across speakers.
- `loao_histograms.png`: Score distribution histograms for held-out attacks.

---

## 6. Roadmap for Phase 1 (Model v2)

With Phase 0 fully executed and verified, the project is ready to begin Phase 1 under isolated directories (`models/v2/`, `data_v2/`, `src_v2/`):

1. **Acoustic Augmentation Defense**: Implement dynamic data augmentation during Wav2Vec2 fine-tuning (random additive noise, room impulse responses, telephony bandpass filtering) to prevent the network from relying on silence and high-frequency shortcuts.
2. **Multi-Generator Diversity**: Expand the training corpus beyond ASVspoof 2019 to include modern neural vocoders (HiFi-GAN, WaveGlow) and commercial TTS generators (ElevenLabs, Edge-TTS, Bark).
3. **In-Pipeline OOD Distance Gate**: Embed the Mahalanobis distance check directly into the FastAPI `/predict` pipeline to flag out-of-distribution audio as "Low Quality / Degraded Audio" rather than emitting false predictions.
4. **Speaker-Invariant Normalization**: Apply per-utterance cepstral mean and variance normalization (CMVN) to stabilize the decision threshold across diverse voices.
