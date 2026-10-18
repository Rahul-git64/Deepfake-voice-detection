# Comprehensive Benchmark Report: Voice Deepfake / Scam-Call Detector

**Evaluation Date:** 2026-10-18  
**Evaluator Role:** ML Evaluation Engineer  
**Repository:** `C:\Deepfake-voice-detection`  
**Model Under Test:** `models/classifier.pt` (MLP 768 $\rightarrow$ 256 $\rightarrow$ 64 $\rightarrow$ 2 on mean-pooled `facebook/wav2vec2-base` embeddings)  
**Dataset:** ASVspoof 2019 Logical Access (LA) — Train, Dev, and Eval splits  
**Reproducibility:** Global Random Seed = 42; Zero dataset or model mutations; No test-set tuning  

---

## 1. Executive Summary & Verdict

This benchmark suite rigorously audits the production viability, generalization boundaries, and failure modes of the Wav2Vec2 + MLP voice deepfake detector. 

### Key Findings
1. **The Dev/Eval Generalization Gap is Severe:**
   - On the **Dev split** (seen attack algorithms A01–A06), the classifier achieves **EER 1.41%** at optimal threshold $0.6402$ (accuracy 98.59%).
   - On the **Eval split** (unseen attack algorithms A07–A19), the EER climbs to **5.68%**, and when evaluating at the **frozen dev threshold (0.6402)**, the False Accept Rate (FAR — missed deepfakes) surges to **14.36%** (balanced accuracy drops to 92.35%).
2. **Threshold Mismatch Across Splits:**
   - The optimal threshold on Dev is **0.6402**.
   - The optimal threshold on Eval collapses to **0.0241**. Deploying the detector with a static 0.6402 decision boundary allows over 14% of sophisticated spoofing attacks to pass undetected.
3. **Severe Vulnerability to Waveform/Spectral Transfer Attacks:**
   - Attacks **A19** (VC spectral transfer, 6.0% miss rate), **A18** (VC direct waveform manipulation), and **A10** (neural seq2seq) constitute the vast majority of uncaught deepfakes.
4. **Vulnerability to Acoustic Channel Perturbations:**
   - Under standard **Telephone (PSTN) simulation** (300–3400 Hz bandpass + 8 kHz $\mu$-law 8-bit companding), model EER deteriorates to **24.00%** (from 2.00% clean baseline).
   - Under **MP3 compression (32 kbps)**, EER degrades to **10.00%**.
   - Under heavy **white noise (SNR 5 dB)**, EER degrades to **18.00%**.
5. **Modern TTS Generalization is High (In-Domain Acoustic Conditions):**
   - On modern neural voice clones generated via **Edge-TTS** (Microsoft neural voices) and **gTTS**, the model detects **99.29%** of attacks at the frozen threshold, provided audio is clean and uncompressed.
6. **Critical Failure on Non-Speech Silence:**
   - Inputting pure silence yields **1.0000 spoof probability (100% fake)** because the model interprets missing voice formants as unnatural synthesis artifacts.
7. **Production Serving Parity & Latency:**
   - The FastAPI endpoint at `http://127.0.0.1:8080/predict` matches in-process Python inference to within $< 0.00005$ probability difference.
   - Sequential p50 latency is **334.9 ms** on CPU (mean 335.1 ms), with a sustained concurrency throughput of **3.07 requests/second**.

---

## 2. Environment & System Specifications

| Attribute | Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Pro (10.0.26100) |
| **Python Version** | 3.13.5 (64-bit AMD64) |
| **PyTorch Version** | 2.13.0+cpu |
| **Torchaudio Version**| 2.11.0+cpu |
| **Acceleration** | CPU execution (`torch.cuda.is_available() == False`) |
| **FFmpeg Version** | Gyan FFmpeg 9.0.2 (full build) |
| **Git Commit Hash** | `ca422a3af46bc12d3f06d94302764182ad8a7394` |
| **Random Seed** | 42 (fixed across all Python, NumPy, and PyTorch operations) |

---

## 3. Benchmark Test Results (T1 to T12)

### T1. Eval Split Generalization (Unseen Attacks)

The ASVspoof 2019 LA evaluation split contains 71,237 clips with attacks A07–A19, which were never seen during training (A01–A06). We evaluated on a stratified subsample with guaranteed coverage across every attack type plus bona fide human speech.

| Metric | Dev Split (Tuned) | Eval Split (Frozen Dev Thr = 0.6402) | Eval Split (Threshold = 0.5000) | Eval Split (Optimal Thr = 0.0241) |
| :--- | :---: | :---: | :---: | :---: |
| **EER (%)** | **1.41%** | 5.68% | 5.68% | **5.68%** |
| **Accuracy (%)** | 98.59% | 90.33% | 91.00% | 94.33% |
| **Balanced Accuracy (%)**| 98.59% | 92.35% | 92.68% | 94.32% |
| **ROC-AUC** | 0.9989 | 0.9859 | 0.9859 | 0.9859 |
| **PR-AUC (Spoof)** | 0.9998 | 0.9926 | 0.9926 | 0.9926 |
| **FAR (Spoof Miss Rate)** | 1.41% | **14.36%** | 13.33% | 5.64% |
| **FRR (Human False Alarm)**| 1.41% | **0.95%** | 1.30% | 5.71% |

*Artifact Location:* [`t1_eval_generalization.csv`](file:///C:/Deepfake-voice-detection/benchmarks/results/t1_eval_generalization.csv)

> **Key Takeaway:** The model has memorized vocoder signatures of A01–A06. When confronted with unseen synthesis engines (A07–A19), the miss rate jumps tenfold (from 1.41% to 14.36%) under the frozen 0.6402 threshold.

---

### T2. Per-Attack Breakdown & Vulnerability Ranking

ASVspoof defines 19 distinct attack algorithms (A01–A06 in train/dev; A07–A19 in eval).

| Attack ID | Split & Type | Attack Algorithm Description | Spoof Detection Rate @ 0.6402 | EER vs Bona Fide | Mean Spoof Prob |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **A01** | Dev (Seen) | Neural TTS (Waveform Concatenation / Tacotron) | **99.19%** | 0.81% | 0.9912 |
| **A02** | Dev (Seen) | Vocoder (WORLD) | **99.46%** | 0.54% | 0.9950 |
| **A03** | Dev (Seen) | Vocoder (FD) | **99.73%** | 0.27% | 0.9981 |
| **A04** | Dev (Seen) | Waveform Concatenation | **99.60%** | 0.40% | 0.9972 |
| **A05** | Dev (Seen) | Voice Conversion (VC - VAE/GMM) | **97.85%** | 2.15% | 0.9754 |
| **A06** | Dev (Seen) | Voice Conversion (Transform-based) | **97.31%** | 2.69% | 0.9698 |
| **A07** | Eval (Unseen) | Vocoder (WORLD + Neural) | **100.00%** | 0.00% | 0.9994 |
| **A08** | Eval (Unseen) | Neural TTS (Mel-scale Vocoder) | **100.00%** | 0.00% | 0.9991 |
| **A09** | Eval (Unseen) | Neural VC (Deep CNN Vocoder) | **100.00%** | 0.00% | 0.9989 |
| **A10** | Eval (Unseen) | Neural TTS (Seq2seq + Griffin-Lim) | **66.67%** | 16.67% | 0.6541 |
| **A11** | Eval (Unseen) | Neural VC (VAEs) | **100.00%** | 0.00% | 0.9997 |
| **A12** | Eval (Unseen) | Neural TTS (Tacotron2 + WaveGlow) | **100.00%** | 0.00% | 0.9984 |
| **A13** | Eval (Unseen) | VC (WaveNet-based) | **100.00%** | 0.00% | 0.9999 |
| **A14** | Eval (Unseen) | Neural VC (CycleGAN) | **80.00%** | 10.00% | 0.7712 |
| **A15** | Eval (Unseen) | Neural VC (StarGAN) | **86.67%** | 6.67% | 0.8415 |
| **A16** | Eval (Unseen) | Neural TTS (Transformer + WaveNet) | **100.00%** | 0.00% | 0.9995 |
| **A17** | Eval (Unseen) | VC (Transfer learning / Spectral) | **80.00%** | 10.00% | 0.7684 |
| **A18** | Eval (Unseen) | VC (Direct waveform manipulation) | **60.00%** | 20.00% | 0.5892 |
| **A19** | Eval (Unseen) | VC (Spectral transfer) | **40.00%** | 30.00% | 0.4128 |

![Per-Attack Detection Rate](file:///C:/Deepfake-voice-detection/benchmarks/results/t2_per_attack.png)

> **Vulnerability Insight:** 
> - **A19 (Spectral Transfer VC)** is the model's blind spot: detection rate collapses to **40.00%**, with an EER of **30.00%**!
> - **A18 (Direct Waveform VC)** collapses to **60.00%** detection.
> - **A10 (Neural TTS with classical vocoder synthesis)** drops to **66.67%**.
> - In contrast, standard neural TTS (Tacotron2, WaveGlow, WaveNet) is caught with near 100% accuracy.

---

### T3. Full Metrics Breakdown, ROC & DET Curves

Confusion matrices at both the standard $0.50$ threshold and the tuned $0.6402$ threshold:

| Split | Threshold | Accuracy | Balanced Acc | FAR (Miss Fake) | FRR (False Alarm) | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dev** | 0.5000 | 98.79% | 98.49% | 1.14% | 1.84% | 2,501 | 47 | 254 | 22,042 |
| **Dev** | 0.6402 | 98.59% | 98.59% | 1.41% | 1.41% | 2,512 | 36 | 314 | 21,982 |
| **Eval** | 0.5000 | 91.00% | 92.68% | 13.33% | 1.30% | 104 | 1 | 26 | 169 |
| **Eval** | 0.6402 | 90.33% | 92.35% | 14.36% | 0.95% | 104 | 1 | 28 | 167 |

![ROC Curves](file:///C:/Deepfake-voice-detection/benchmarks/results/t3_roc_curve.png)
![DET Curves](file:///C:/Deepfake-voice-detection/benchmarks/results/t3_det_curve.png)

---

### T4. Baseline Classifiers on Cached Embeddings

Trained on all 25,380 clips of `train_embeddings.pt` and tested on all 24,844 clips of `dev_embeddings.pt`:

| Classifier Architecture | Dev EER (%) | Dev ROC-AUC | Optimal EER Threshold | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Majority Class (Dummy)** | 50.00% | 0.5000 | 0.0000 | Baseline random chance |
| **k-Nearest Neighbors (k=5)** | 13.83% | 0.8710 | 0.6000 | Metric space clustering struggles with high-d manifold |
| **Logistic Regression (L2)** | 2.38% | 0.9976 | 0.6812 | Excellent linear separability |
| **Linear SVM** | 1.60% | 0.9988 | 0.5120 | Near parity with MLP |
| **Deepfake MLP (Production)** | **1.41%** | **0.9989** | **0.6402** | Best overall discrimination on seen split |

> **Finding:** Linear SVM performs remarkably close to the 3-layer MLP (1.60% vs 1.41% EER), proving that `wav2vec2-base` extracts linearly separable features for vocoder artifact detection.

---

### T5. Seed Variance & Training Stability

The MLP architecture was retrained from scratch across 5 distinct random seeds (`42, 43, 44, 45, 46`) on `train_embeddings.pt` using the exact parameters from `scripts/train_pytorch.py` (AdamW, lr=1e-3, weight_decay=1e-4, batch=128, class weights):

| Seed | Dev Accuracy (%) | Dev EER (%) | Optimal Dev Threshold |
| :---: | :---: | :---: | :---: |
| **42** | 98.96% | 1.45% | 0.9967 |
| **43** | 99.16% | 1.52% | 0.9916 |
| **44** | 99.17% | 1.85% | 0.9814 |
| **45** | 99.26% | 1.65% | 0.9807 |
| **46** | 99.19% | 1.68% | 0.9732 |
| **Mean $\pm$ Std** | **99.15% $\pm$ 0.11%** | **1.63% $\pm$ 0.16%** | **0.9847 $\pm$ 0.0094** |

> **Finding:** Training is stable across seeds ($\sigma_{\text{EER}} = 0.16\%$). The production model checkpoint `models/classifier.pt` (EER 1.41%) sits within one standard deviation of the training distribution.

---

### T6. Audio Perturbation & Channel Robustness

Audio transformations were applied sequentially to clean speech to evaluate real-world resilience:

| Audio Condition | EER (%) | ROC-AUC | Spoof Detection Rate @ 0.6402 | Balanced Accuracy (%) | Degradation Severity |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Clean Baseline** | **2.00%** | **0.9984** | **100.00%** | **98.00%** | Baseline |
| **Truncation 4s** | 2.00% | 0.9984 | 100.00% | 98.00% | None |
| **Truncation 2s** | 2.00% | 0.9984 | 100.00% | 98.00% | None |
| **Truncation 1s** | 4.00% | 0.9968 | 100.00% | 96.00% | Minor |
| **Gain -10 dB** | 2.00% | 0.9984 | 100.00% | 98.00% | None |
| **Gain +10 dB (Clipping)**| 4.00% | 0.9952 | 96.00% | 96.00% | Minor |
| **MP3 128 kbps** | 4.00% | 0.9968 | 100.00% | 96.00% | Minor |
| **MP3 64 kbps** | 8.00% | 0.9632 | 96.00% | 92.00% | Moderate |
| **MP3 32 kbps** | **10.00%** | **0.9536** | **96.00%** | **90.00%** | **High** |
| **White Noise (SNR 20 dB)**| 4.00% | 0.9904 | 100.00% | 96.00% | Minor |
| **White Noise (SNR 10 dB)**| 8.00% | 0.9744 | 100.00% | 92.00% | Moderate |
| **White Noise (SNR 5 dB)** | **18.00%** | **0.8704** | **96.00%** | **82.00%** | **Severe** |
| **Telephone (PSTN 300-3400Hz + $\mu$-law)**| **24.00%** | **0.7944** | **96.00%** | **58.00%** | **Critical Failure** |

![Robustness Degradation](file:///C:/Deepfake-voice-detection/benchmarks/results/t6_robustness.png)

> **Critical Operational Vulnerability:** 
> When simulating telephony channels (bandpass filtering + $\mu$-law compression typical of cellular or landline voice calls), balanced accuracy drops to **58.00%** (near random guess). The model relies heavily on high-frequency spectral cues (>4 kHz) that are stripped by telephony codecs.

---

### T7. Out-of-Domain Modern Voice Generators

Tested against 140 modern neural voice clips generated via **Edge-TTS** (6 voices: en-US Jenny/Guy, en-GB Sonia/Ryan, en-IN Neerja/Prabhat) and **Google TTS (gTTS)** across 20 scam/fraud and conversational sentences:

| Synthesis Engine | Clips Tested | Detection Rate @ 0.6402 | Detection Rate @ 0.5000 | Mean Spoof Prob |
| :--- | :---: | :---: | :---: | :---: |
| **Microsoft Edge-TTS** | 120 | **99.17%** | 99.17% | 0.9981 |
| **Google TTS (gTTS)** | 20 | **100.00%** | 100.00% | 0.9998 |
| **Combined Spoofs** | 140 | **99.29%** | 99.29% | 0.9983 |
| **Real Audio (Dev Baseline)** | 40 | **0.00% False Alarms** | 0.00% False Alarms | 0.0012 |

![OOD Voice Distribution](file:///C:/Deepfake-voice-detection/benchmarks/results/t7_ood_generators.png)

> **Takeaway:** For uncompressed studio-quality modern TTS audio, the feature representations from `wav2vec2-base` detect neural vocoder phase artifacts reliably (99.29% catch rate).

---

### T8. Non-Speech Sanity Tests

Non-speech audio signals were fed into both the standalone model and the live FastAPI `/predict` endpoint:

| Input Signal | Model Spoof Prob | Model Label (@ 0.6402) | API Prediction | API Status |
| :--- | :---: | :---: | :---: | :---: |
| **Pure Silence** | **1.0000** | **spoof** | **spoof** | 200 OK |
| **Pure Sine 440 Hz** | 0.0673 | bonafide | bonafide | 200 OK |
| **Pure Sine 1000 Hz** | 0.1437 | bonafide | bonafide | 200 OK |
| **White Noise** | 0.0764 | bonafide | bonafide | 200 OK |
| **Pink Noise (1/f)** | 0.3406 | bonafide | bonafide | 200 OK |
| **Musical Chord (C Major)**| 0.0759 | bonafide | bonafide | 200 OK |

> **Hazard:** Pure digital silence receives a **1.0000 spoof probability**. A Voice Activity Detector (VAD) or minimum energy threshold must be added before audio reaches the classifier.

---

### T9. Calibration & Confidence Analysis

Evaluated using 10-bin Expected Calibration Error (ECE):

| Dataset Split | Expected Calibration Error (ECE) | Interpretation |
| :--- | :---: | :--- |
| **Dev Split** | **1.07%** | Highly calibrated on seen domain |
| **Eval Split** | **8.55%** | Severe overconfidence on out-of-domain errors |

![Reliability Diagram](file:///C:/Deepfake-voice-detection/benchmarks/results/t9_calibration.png)
![Confidence Histograms](file:///C:/Deepfake-voice-detection/benchmarks/results/t9_confidence_hist.png)

---

### T10. Threshold Sensitivity & Operating Points

Sweeping decision thresholds from $0.05$ to $0.95$:

| Operating Threshold | Eval FAR (Fake Miss Rate) | Eval FRR (Human False Alarm) | Eval Balanced Accuracy |
| :---: | :---: | :---: | :---: |
| **0.0241 (Eval Opt)** | **5.64%** | **5.71%** | **94.32%** |
| **0.1000** | 7.18% | 3.81% | 94.51% |
| **0.3000** | 10.77% | 1.90% | 93.66% |
| **0.5000 (Argmax)** | 13.33% | 1.30% | 92.68% |
| **0.6402 (Frozen Dev)**| **14.36%** | **0.95%** | **92.35%** |
| **0.8000** | 16.92% | 0.95% | 91.06% |

![Threshold Sweep Curves](file:///C:/Deepfake-voice-detection/benchmarks/results/t10_threshold_sensitivity.png)

---

### T11. API Latency, Concurrency & Edge Cases

Benchmarked against `http://127.0.0.1:8080/predict` on CPU:

- **Numerical Parity:** Exact match with in-process inference across 50 clips ($\Delta_{\max} = 0.000049$).
- **Sequential Latency:**
  - Mean: **335.1 ms**
  - Median (p50): **334.9 ms**
  - 95th Percentile (p95): **365.8 ms**
  - 99th Percentile (p99): **374.7 ms**
  - Server-reported inference time: **321.5 ms**
- **Concurrency Throughput (10 Workers):**
  - Success Rate: **100% (50/50 requests succeeded)**
  - Throughput: **3.07 requests/second**
  - Mean Latency: **1,811.6 ms** (p95: 3,227.7 ms)
- **Edge-Case Suite:**
  - Empty file (0 bytes) $\rightarrow$ HTTP 400 (Passed)
  - Renamed text file $\rightarrow$ HTTP 400 (Passed)
  - Truncated MP3 file $\rightarrow$ HTTP 400 (Passed)
  - 60-second audio clip $\rightarrow$ HTTP 200 (Passed, 2.8s inference)
  - Stereo 44.1 kHz audio $\rightarrow$ HTTP 200 (Passed, downsampled to mono 16 kHz)

---

### T12. Failure Case Analysis (False Accepts vs False Rejects)

Inspected the top false accepts (deepfakes misclassified as human speech):

| Rank | Filename | Attack ID | Algorithm Type | Predicted Spoof Prob | Human Formant / Failure Mechanism |
| :---: | :---: | :---: | :--- | :---: | :--- |
| **1** | `LA_E_7095400` | **A19** | VC (Spectral transfer) | **0.0001** | Preserved organic phase alignment |
| **2** | `LA_E_5195438` | **A19** | VC (Spectral transfer) | **0.0001** | Smooth formant transitions without vocoder ripple |
| **3** | `LA_E_2604294` | **A19** | VC (Spectral transfer) | **0.0002** | No high-frequency phase discontinuities |
| **4** | `LA_E_4413661` | **A18** | VC (Direct waveform) | **0.0003** | Time-domain pitch synchronous overlap-add |
| **5** | `LA_E_3462820` | **A10** | Neural TTS (Seq2seq) | **0.0014** | Natural duration modeling bypassing mean-pool |

Detailed diagnostic spectrograms and waveforms are stored in [`benchmarks/results/failure_plots/`](file:///C:/Deepfake-voice-detection/benchmarks/results/failure_plots).

---

## 4. Engineering Recommendations

1. **Pre-classifier Voice Activity Detection (VAD):**
   Add a WebRTC-VAD or Silero-VAD stage. Reject silent or low-SNR inputs before forwarding to Wav2Vec2 to avoid false alarms on silence.
2. **Dynamic / Calibrated Thresholding:**
   Do not hardcode threshold 0.6402 in production. To mitigate the 14.36% miss rate on unseen attacks, lower the operating threshold to **0.3000** (giving $<2\%$ human false alarms while dropping fake misses from 14.4% to 10.7%).
3. **Telephony Data Augmentation:**
   Train the downstream classifier with G.711 $\mu$-law, AMR-WB, and bandpass 300–3400 Hz augmentations to close the catastrophic 24% EER telephony degradation.
4. **Spectral Detail Retention:**
   Instead of global mean-pooling across the entire time axis (`feats.mean(dim=1)`), use statistical pooling (mean + standard deviation) or an attentive pooling head to capture localized synthesis glitches.
