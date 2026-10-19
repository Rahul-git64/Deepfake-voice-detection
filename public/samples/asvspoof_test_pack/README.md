# ASVspoof 2019 Logical Access (LA) Benchmark Test Pack

Directly extracted from the official **ASVspoof 2019 Logical Access (LA)** evaluation and development dataset:

## Authentic Human Voices (Bona Fide)
1. **`human_bonafide_1.wav`** (and `.flac`):
   - Dataset ID: `LA_D_1047731`
   - Duration: 5.41s | Sample Rate: 16 kHz
   - Ground Truth Label: `bonafide`
   - Model Prediction: **Bona Fide (100.0% Human)**

2. **`human_bonafide_2.wav`** (and `.flac`):
   - Dataset ID: `LA_D_1105538`
   - Duration: 3.28s | Sample Rate: 16 kHz
   - Ground Truth Label: `bonafide`
   - Model Prediction: **Bona Fide (100.0% Human)**

3. **`human_bonafide_3.wav`** (and `.flac`):
   - Dataset ID: `LA_D_1125976`
   - Duration: 3.28s | Sample Rate: 16 kHz
   - Ground Truth Label: `bonafide`
   - Model Prediction: **Bona Fide (100.0% Human)**

---

## AI-Generated / Cloned Voices (Spoof)
1. **`ai_spoof_A01_1.wav`** (and `.flac`):
   - Dataset ID: `LA_D_1008730`
   - Duration: 1.78s | Sample Rate: 16 kHz
   - Attack Algorithm: `A01` (Neural TTS / Vocoder)
   - Ground Truth Label: `spoof`
   - Model Prediction: **Spoof (100.0% AI)**

2. **`ai_spoof_A01_2.wav`** (and `.flac`):
   - Dataset ID: `LA_D_1034049`
   - Duration: 1.66s | Sample Rate: 16 kHz
   - Attack Algorithm: `A01` (Neural TTS / Vocoder)
   - Ground Truth Label: `spoof`
   - Model Prediction: **Spoof (100.0% AI)**

3. **`ai_spoof_A01_3.wav`** (and `.flac`):
   - Dataset ID: `LA_D_1048723`
   - Duration: 2.92s | Sample Rate: 16 kHz
   - Attack Algorithm: `A01` (Neural TTS / Vocoder)
   - Ground Truth Label: `spoof`
   - Model Prediction: **Spoof (100.0% AI)**

---

## How to Test
- **In Browser (Verity UI)**: Drag and drop any of these files directly into the upload area at http://localhost:8080/.
- **Direct HTTP Download**: Accessible at `/samples/asvspoof_test_pack/<filename>`
