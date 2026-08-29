# Deepfake-voice-detection
Deepfake voice calls are a rising problem in the current world, this has led to many unfortunate incidents and the victims are rising from the voice phishing scams. To help tackle this problem I am making this fake voice detection model which differentiates between the fake voices and real voices, ultimately reducing the victims of this scam
# Voice Deepfake / Scam-Call Detector

Deepfake voice calls are a rising problem in the current world, this has led to many unfortunate incidents and the victims are rising from the voice phishing scams. To help counter this I am making this fake voice detection model which differentiates between the fake voices and real voices, ultimately reducing the victims of this scam.

## Problem Statement

AI voice cloning is increasingly used to impersonate real people in scam phone calls, and most victims have no way to tell a cloned voice from a genuine one in the moment. This is distinct from stylistic voice-changer effects (like those used in reels/social media filters) — this project specifically targets **detecting synthetic/cloned speech used to impersonate real voices**, not casual voice modification.

## Solution Overview

This project builds a binary classifier that takes an audio clip and predicts whether it is a **real (bona fide)** human voice or an **AI-generated (spoofed)** one, along with a confidence score. Rather than training an audio model from scratch, it uses a pretrained wav2vec2 model to extract speech embeddings, then trains a lightweight classifier on top of those embeddings. The trained model is served through an API and deployed publicly so it can be tested with real audio uploads.

## Tech Stack

- **Language/ML:** Python, PyTorch, HuggingFace Transformers (wav2vec2)
- **Classifier:** scikit-learn / XGBoost
- **Experiment tracking:** MLflow
- **Serving:** FastAPI
- **Containerization:** Docker
- **CI:** GitHub Actions
- **Deployment:** Hugging Face Spaces / Render

## Dataset

This project uses the [ASVspoof](https://www.asvspoof.org/) dataset, a public benchmark of real and AI-synthesized speech built specifically for spoof/deepfake detection research.

> Note: raw audio files are not committed to this repository (see `.gitignore`). To reproduce this project, request access to ASVspoof from the official source and place the data as described in `scripts/download_data.py` (once added).

## Project Status / Roadmap

- [x] Repo scaffolding + README
- [ ] ASVspoof dataset access requested
- [ ] Environment setup (`requirements.txt`)
- [ ] Load + inspect audio samples
- [ ] Exploratory audio analysis (waveforms/spectrograms)
- [ ] Extract wav2vec2 embeddings
- [ ] Train baseline classifier
- [ ] Track experiments with MLflow
- [ ] Build FastAPI app
- [ ] Dockerize
- [ ] Set up GitHub Actions CI
- [ ] Deploy publicly
- [ ] (Optional) Simple frontend for demo

## Setup / How to Run

*(To be filled in as each component is built.)*

## Live Demo

*(Link + example usage to be added after deployment.)*

## Results

*(Accuracy, precision/recall, and confusion matrix to be added once a model is trained.)*
