import io
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.pipelines as pipelines
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CLASSIFIER_PATH = Path("models/classifier.pt")
TEMPLATE_PATH = Path("app/templates/index.html")
LEGACY_TEMPLATE_PATH = Path("app/templates/legacy.html")
SAMPLES_DIR = Path("public/samples")
PUBLIC_DIR = Path("public")
TARGET_SR = 16000

app = FastAPI(
    title="Voice Deepfake / Scam-Call Detector API",
    description="API for detecting AI-synthesized vs bona fide human speech using Wav2Vec2 + PyTorch Classifier",
    version="1.0.0",
)


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


# Global variables for models
bundle = None
wav2vec2_model = None
classifier = None
device = "cuda" if torch.cuda.is_available() else "cpu"


class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    label: int
    confidence: float
    probabilities: dict
    inference_time_ms: float


@app.on_event("startup")
def load_models():
    global bundle, wav2vec2_model, classifier
    print(f"Loading Wav2Vec2 Base model on {device}...")
    bundle = pipelines.WAV2VEC2_BASE
    wav2vec2_model = bundle.get_model()
    wav2vec2_model.eval()
    wav2vec2_model.to(device)
    print("Wav2Vec2 Base loaded.")

    if CLASSIFIER_PATH.exists():
        print(f"Loading classifier from '{CLASSIFIER_PATH}'...")
        checkpoint = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
        classifier = DeepfakeClassifier(in_features=checkpoint.get("in_features", 768))
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.eval()
        classifier.to(device)
        print("Classifier loaded successfully.")
    else:
        print(f"Warning: Classifier model file '{CLASSIFIER_PATH}' not found. Make sure to train a model first.")


@app.get("/", response_class=HTMLResponse)
def serve_home():
    if TEMPLATE_PATH.exists():
        return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Deepfake Voice Detector API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/legacy", response_class=HTMLResponse)
def serve_legacy():
    if LEGACY_TEMPLATE_PATH.exists():
        return HTMLResponse(content=LEGACY_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Legacy Detector</h1><p>Legacy template not found.</p>")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": device,
        "wav2vec2_loaded": wav2vec2_model is not None,
        "classifier_loaded": classifier is not None,
    }


def preprocess_audio_bytes(audio_bytes: bytes) -> torch.Tensor:
    """Reads audio bytes, resamples to TARGET_SR (16kHz), converts to mono 1D tensor."""
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        waveform = torch.from_numpy(data).float()
        if waveform.ndim > 1:  # stereo to mono
            waveform = waveform.mean(dim=-1)
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
        return waveform
    except Exception as e:
        raise ValueError(f"Failed to decode audio file: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    start_time = time.time()

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=f"Classifier model is not loaded. Expected checkpoint at {CLASSIFIER_PATH}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file provided.")

    try:
        waveform = preprocess_audio_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    with torch.no_grad():
        waveform_batch = waveform.unsqueeze(0).to(device)
        features, _ = wav2vec2_model.extract_features(waveform_batch)
        # Final layer hidden state is the last element
        last_hidden = features[-1]
        embedding = last_hidden.mean(dim=1).to(device)

        logits = classifier(embedding)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    pred_label = int(np.argmax(probs))
    label_str = "spoof" if pred_label == 1 else "bonafide"
    confidence = float(probs[pred_label])

    elapsed_ms = (time.time() - start_time) * 1000.0

    return PredictionResponse(
        filename=file.filename or "unknown",
        prediction=label_str,
        label=pred_label,
        confidence=round(confidence, 4),
        probabilities={
            "bonafide": round(float(probs[0]), 4),
            "spoof": round(float(probs[1]), 4),
        },
        inference_time_ms=round(elapsed_ms, 2),
    )


# Mount static assets after all API routes
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

if PUBLIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

