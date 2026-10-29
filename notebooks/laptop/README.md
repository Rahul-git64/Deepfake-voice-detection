# 💻 Laptop Jupyter Notebooks

This folder contains notebooks pre-configured and optimized to run locally on your **laptop** (Windows / macOS / Linux) using CPU or consumer NVIDIA GPU without requiring external cloud/Colab runtimes.

---

## Available Notebooks

| Notebook | Purpose | Device Requirements | Runtime |
|---|---|---|---|
| [`train_v2_laptop.ipynb`](train_v2_laptop.ipynb) | Trains the **Model V2** classifier using cached embeddings, LayerNorm + Dropout regularized MLP, and speaker-disjoint cross-validation. Saves checkpoints strictly to `models/v2/`. | Multi-threaded CPU or local CUDA GPU | ~1–2 minutes |
| [`generate_tts_laptop.ipynb`](generate_tts_laptop.ipynb) | Generates multi-accent local TTS speech clips across en-US, en-GB, en-IN, hi-IN, te-IN, and gTTS. Includes interactive audio player widgets. | Zero GPU overhead (pure Python async CPU) | ~1–2 minutes |
| [`extract_embeddings_laptop.ipynb`](extract_embeddings_laptop.ipynb) | Extracts 768-dim Wav2Vec2 embeddings from local audio files (`.flac` / `.wav` / `.mp3`) in batched mode with progress tracking. | Multi-threaded CPU or local CUDA GPU | Scalable |

---

## How to Launch on Laptop

1. Activate your local virtual environment:
   ```powershell
   .\miniproject\Scripts\activate
   ```
2. Start Jupyter Notebook:
   ```powershell
   jupyter notebook notebooks/laptop
   ```
3. Open any of the `.ipynb` files and run the cells sequentially.
