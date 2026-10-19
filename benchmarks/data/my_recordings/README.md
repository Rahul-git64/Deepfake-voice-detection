# Real Audio Test Collection (`my_recordings/`)

Please place 30+ recordings of genuine human voices here for diagnosis and validation of model generalization.

### Recommended Audio Profiles:
1. **Smartphone Microphone:** Voice memos, conversational speech recorded in quiet and noisy rooms (WAV, FLAC, M4A, or MP3).
2. **Laptop Microphone:** Internal laptop mic recordings with ambient fan noise / reverberation.
3. **Real Telephone Call:** Audio recorded during a live cellular or landline voice call (transcoded by G.711 / AMR).
4. **Browser-recorded WebM/Opus:** Web browser audio recorded using MediaRecorder.

### File Naming Convention:
Name your files descriptively so metadata can be parsed automatically (e.g.):
- `user_phone_quiet_01.wav`
- `friend_laptop_fan_01.m4a`
- `user_cellcall_pstn_01.mp3`
- `user_browser_mic_01.webm`

If this folder is empty, `benchmarks/diagnose_real_audio.py` will generate illustrative real/synthesized test captures and sample bona fide recordings to verify the entire pipeline diagnostic suite.
