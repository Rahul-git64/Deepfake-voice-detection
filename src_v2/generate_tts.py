#!/usr/bin/env python3
"""
src_v2/generate_tts.py

Local multi-engine, multi-accent TTS speech generator for data_v2.
Generates synthetic speech clips using Edge-TTS and gTTS across:
- en-US (American English): en-US-JennyNeural (train/dev), en-US-GuyNeural (HELD OUT EVAL)
- en-GB (British English): en-GB-SoniaNeural, en-GB-RyanNeural
- en-IN (Indian English): en-IN-NeerjaNeural, en-IN-PrabhatNeural
- hi-IN (Hindi): hi-IN-SwaraNeural, hi-IN-MadhurNeural
- te-IN (Telugu): te-IN-ShrutiNeural, te-IN-MohanNeural
- gTTS (Google Translate TTS): English (en)

Metadata is saved to data_v2/tts/tts_metadata.csv.
"""

import os
import sys
import asyncio
from pathlib import Path
import pandas as pd
import soundfile as sf
import librosa
import edge_tts
from gtts import gTTS

TTS_DIR = Path("data_v2/tts")
METADATA_PATH = TTS_DIR / "tts_metadata.csv"

# Diverse phonetically balanced & conversational sentences
PHRASES = [
    # English (Conversational & Financial)
    {"text": "Your account security verification code is four eight two nine one.", "lang": "en"},
    {"text": "Artificial intelligence speech models require robust testing across accents.", "lang": "en"},
    {"text": "The quick brown fox jumps gracefully over the lazy golden retriever.", "lang": "en"},
    {"text": "We noticed an unauthorized login attempt from a new mobile device.", "lang": "en"},
    {"text": "Good morning, thank you for calling customer service, how can I assist you today?", "lang": "en"},
    {"text": "Please confirm your registered phone number before we process the wire transfer.", "lang": "en"},
    {"text": "Heavy rainfall is expected across coastal districts throughout the weekend.", "lang": "en"},
    {"text": "Our team has completed the annual software security and compliance audit.", "lang": "en"},
    {"text": "Could you please send me the updated quarterly budget spreadsheet by tonight?", "lang": "en"},
    {"text": "Voice authentication systems protect sensitive personal identification records.", "lang": "en"},
    # Hindi (Standard conversational)
    {"text": "नमस्ते, आपका बैंक खाता सफलतापूर्वक सत्यापित कर दिया गया है।", "lang": "hi"},
    {"text": "कृत्रिम बुद्धिमत्ता आधुनिक तकनीक को तेज़ी से बदल रही है।", "lang": "hi"},
    {"text": "कृपया अपना पासवर्ड या सुरक्षा कोड किसी के साथ साझा न करें।", "lang": "hi"},
    {"text": "आज का मौसम बहुत सुहावना है और शाम को हल्की बारिश हो सकती है।", "lang": "hi"},
    # Telugu (Standard conversational)
    {"text": "నమస్కారం, మీ ఖాతా వివరాలు విజయవంతంగా ధృవీకరించబడ్డాయి.", "lang": "te"},
    {"text": "కృత్రిమ మేధస్సు భవిష్యత్తు సాంకేతిక పరిజ్ఞానాన్ని నిర్దేశిస్తుంది.", "lang": "te"},
    {"text": "దయచేసి మీ రహస్య పిన్ నంబర్ ఎవరికీ చెప్పవద్దు.", "lang": "te"},
    {"text": "ఈ రోజు వాతావరణం చల్లగా ఉంది మరియు సాయంత్రం వర్షం పడవచ్చు.", "lang": "te"},
]

VOICE_CONFIGS = [
    # Engine, Voice Name, Speaker ID, Split, Lang
    {"engine": "edge_tts", "voice": "en-US-JennyNeural", "speaker": "edge_Jenny", "split": "train", "lang": "en"},
    {"engine": "edge_tts", "voice": "en-GB-SoniaNeural", "speaker": "edge_Sonia", "split": "train", "lang": "en"},
    {"engine": "edge_tts", "voice": "en-IN-NeerjaNeural", "speaker": "edge_Neerja", "split": "dev", "lang": "en"},
    {"engine": "edge_tts", "voice": "hi-IN-SwaraNeural", "speaker": "edge_Swara", "split": "dev", "lang": "hi"},
    {"engine": "edge_tts", "voice": "te-IN-ShrutiNeural", "speaker": "edge_Shruti", "split": "dev", "lang": "te"},
    # Held-out voice family (NEVER in training)
    {"engine": "edge_tts", "voice": "en-US-GuyNeural", "speaker": "edge_Guy", "split": "heldout_eval", "lang": "en"},
    # gTTS engine
    {"engine": "gtts", "voice": "google_en", "speaker": "gtts_default", "split": "dev", "lang": "en"},
]


async def synthesize_edge(text: str, voice: str, out_path: Path):
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(str(out_path))


def synthesize_gtts(text: str, out_path: Path):
    tts = gTTS(text=text, lang="en")
    tts.save(str(out_path))


async def generate_all_tts():
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    print(f"Starting local TTS generation into {TTS_DIR} ...", flush=True)

    for v_cfg in VOICE_CONFIGS:
        engine = v_cfg["engine"]
        voice = v_cfg["voice"]
        speaker = v_cfg["speaker"]
        split = v_cfg["split"]
        v_lang = v_cfg["lang"]

        # Filter phrases by language
        matching_phrases = [p for p in PHRASES if p["lang"] == v_lang]
        if not matching_phrases:
            continue

        print(f"Generating voice {voice} ({len(matching_phrases)} phrases)...", flush=True)

        for idx, item in enumerate(matching_phrases):
            text = item["text"]
            filename = f"{speaker}_{idx:02d}.mp3"
            filepath = TTS_DIR / filename

            if not filepath.exists() or filepath.stat().st_size == 0:
                try:
                    if engine == "edge_tts":
                        await synthesize_edge(text, voice, filepath)
                    elif engine == "gtts":
                        synthesize_gtts(text, filepath)
                except Exception as e:
                    print(f"Warning: Failed to generate {filename}: {e}", file=sys.stderr)
                    continue

            # Check duration via librosa or soundfile
            try:
                dur = librosa.get_duration(path=str(filepath))
            except Exception:
                dur = 3.0

            records.append({
                "path": str(filepath).replace("\\", "/"),
                "label": "spoof",
                "source": engine,
                "speaker": speaker,
                "generator": voice,
                "text": text,
                "language": v_lang,
                "duration": round(float(dur), 2),
                "split": split,
            })

    df = pd.DataFrame(records)
    df.to_csv(METADATA_PATH, index=False)
    print(f"Generated {len(df)} TTS clips. Metadata saved to {METADATA_PATH}")
    return df


def main():
    asyncio.run(generate_all_tts())


if __name__ == "__main__":
    main()
