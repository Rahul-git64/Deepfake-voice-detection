# Deepfake Voice & Audio Scam Detector — Forensics Lab

Deepfake voice calls are a rising threat worldwide, leading to high-stakes impersonation, phishing, and CEO/family emergency scams. This project provides an end-to-end deepfake voice detection pipeline designed to distinguish authentic human speech from AI-synthesized or cloned audio in real time.

---

## 🔬 System Overview & Architecture

- **Frontend**: High-performance interactive Audio Forensics Lab (React 18, 60 FPS Canvas Sound-Orb, WaveSurfer.js waveform visualizer, dynamic probability gauge, 30s microphone recorder, live status polling).
- **Backend / API**: FastAPI service serving predictions with sub-second latency on CPU/CUDA (`app/main.py`).
- **Feature Extraction**: `facebook/wav2vec2-base` (12 transformer blocks, mean-pooled 768-dimensional representations) resampled to 16 kHz mono.
- **Classifier Architecture**: `768 -> Dense(256) [BatchNorm, ReLU, Dropout 0.3] -> Dense(64) [ReLU, Dropout 0.3] -> 2 Classes` with balanced class weights handling the 9:1 spoof imbalance.
- **Dataset**: ASVspoof 2019 Logical Access (LA) benchmark:
  - **Train**: 25,380 clips
  - **Dev / Validation**: 24,844 clips

### 📊 Benchmark Performance (Dev Set)
- **Equal Error Rate (EER)**: **1.41%** (Threshold: `0.6402`)
- **Accuracy**: **99.22%**
- **Inference Latency**: Typical sub-second execution (~250ms–700ms on CPU)

---

## 🚀 Quickstart

### 1. Run with Python Local Environment
```bash
# Activate your environment
miniproject\Scripts\activate

# Launch the FastAPI app with live reload on port 8080
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```
Open your browser at:
- **Modern Forensics Lab Dashboard**: [http://localhost:8080/](http://localhost:8080/)
- **Legacy Fallback Dashboard**: [http://localhost:8080/legacy](http://localhost:8080/legacy)
- **Interactive Swagger Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)
- **Health & Readiness Check**: [http://localhost:8080/health](http://localhost:8080/health)

### 2. Run with Docker
```bash
# Build multi-stage container
docker build -t deepfake-voice-detector .

# Run on port 8080
docker run -p 8080:8080 deepfake-voice-detector
```

---

## 📡 API Endpoints

### `GET /health`
Returns the operational health, hardware acceleration device (`cpu` / `cuda`), and model loading status:
```json
{
  "status": "healthy",
  "device": "cpu",
  "wav2vec2_loaded": true,
  "classifier_loaded": true
}
```

### `POST /predict`
Multipart form upload accepting `.wav`, `.flac`, `.mp3`, and `.m4a` audio files (field name: `file`):
```json
{
  "filename": "audio.wav",
  "prediction": "bonafide",
  "label": 0,
  "confidence": 0.998,
  "probabilities": {
    "bonafide": 0.998,
    "spoof": 0.002
  },
  "inference_time_ms": 276.7
}
```

---

## 🛡️ Limitations & Responsible Use
- Metrics are benchmarked on the **ASVspoof 2019 Logical Access** development dataset.
- Real-world performance may vary on unseen speech generators, severe background noise, lossy compression (e.g., low-bitrate VoIP or telephony codecs), and extremely short audio snippets (< 0.5s).
- This tool is engineered as an automated screening and forensic aid, not definitive legal proof.

---
Built by Rahul
