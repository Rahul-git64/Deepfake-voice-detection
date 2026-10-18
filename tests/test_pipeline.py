import io
import numpy as np
import soundfile as sf
import torch

from app.main import preprocess_audio_bytes, TARGET_SR


def test_audio_preprocessing():
    # Create 1 second 440 Hz sine wave audio at 22050 Hz
    sr_orig = 22050
    t = np.linspace(0, 1.0, sr_orig, endpoint=False)
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Save to in-memory WAV file
    buf = io.BytesIO()
    sf.write(buf, audio_data, sr_orig, format="WAV")
    wav_bytes = buf.getvalue()

    # Preprocess using app main pipeline
    processed_tensor = preprocess_audio_bytes(wav_bytes)

    assert isinstance(processed_tensor, torch.Tensor)
    assert processed_tensor.ndim == 1
    # Check length is approx 16000 samples (1 second at 16kHz)
    assert abs(len(processed_tensor) - TARGET_SR) < 100
    print(f"Audio preprocessing test PASSED. Output shape: {processed_tensor.shape}", flush=True)


if __name__ == "__main__":
    test_audio_preprocessing()
