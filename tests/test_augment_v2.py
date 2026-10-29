#!/usr/bin/env python3
"""
tests/test_augment_v2.py

Unit tests for src_v2/augment.py augmentation functions and AudioAugmentor pipeline.
Uses standard Python unittest.
"""

import unittest
import numpy as np
from src_v2.augment import (
    compute_rms,
    trim_silence_boundary,
    pad_silence,
    apply_highpass_filter,
    apply_lowpass_filter,
    apply_8k_resample,
    apply_noise,
    apply_phone_chain,
    AudioAugmentor,
    HELD_OUT_TRANSFORMS,
    TARGET_SR,
)


class TestAugmentV2(unittest.TestCase):
    def setUp(self):
        self.sr = TARGET_SR
        t = np.linspace(0, 2.0, int(2.0 * self.sr), endpoint=False)
        # Sine mixture with silent edges
        body = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)
        silence = np.zeros(int(0.5 * self.sr))
        self.signal = np.concatenate([silence, body, silence]).astype(np.float32)

    def test_trim_and_pad(self):
        trimmed = trim_silence_boundary(self.signal, keep_margin_sec=0.1, sr=self.sr)
        self.assertLess(len(trimmed), len(self.signal))
        self.assertFalse(np.isnan(trimmed).any())

        padded_zeros = pad_silence(trimmed, pad_sec=0.5, mode="zeros", sr=self.sr)
        self.assertEqual(len(padded_zeros), len(trimmed) + 2 * int(0.5 * self.sr))

        padded_room = pad_silence(trimmed, pad_sec=0.5, mode="room_tone", sr=self.sr)
        self.assertEqual(len(padded_room), len(trimmed) + 2 * int(0.5 * self.sr))
        self.assertFalse(np.isnan(padded_room).any())

    def test_filters(self):
        hp = apply_highpass_filter(self.signal, cutoff_hz=300.0, sr=self.sr)
        self.assertEqual(len(hp), len(self.signal))
        self.assertFalse(np.isnan(hp).any())

        lp = apply_lowpass_filter(self.signal, cutoff_hz=4000.0, sr=self.sr)
        self.assertEqual(len(lp), len(self.signal))
        self.assertFalse(np.isnan(lp).any())

    def test_resample_and_phone(self):
        resampled = apply_8k_resample(self.signal, sr=self.sr)
        self.assertEqual(len(resampled), len(self.signal))
        self.assertFalse(np.isnan(resampled).any())

        phoned = apply_phone_chain(self.signal, sr=self.sr)
        self.assertEqual(len(phoned), len(self.signal))
        self.assertFalse(np.isnan(phoned).any())

    def test_noise_types(self):
        for ntype in ["white", "low_band", "high_band", "pink"]:
            noisy = apply_noise(self.signal, noise_type=ntype, snr_db=20.0, sr=self.sr)
            self.assertEqual(len(noisy), len(self.signal))
            self.assertFalse(np.isnan(noisy).any())

    def test_holdouts_integrity(self):
        self.assertIn("pink_noise", HELD_OUT_TRANSFORMS)
        self.assertIn("lowpass_3400hz", HELD_OUT_TRANSFORMS)
        self.assertIn("mp3_32kbps", HELD_OUT_TRANSFORMS)

    def test_augmentor_determinism(self):
        aug1 = AudioAugmentor(seed=42)
        aug2 = AudioAugmentor(seed=42)
        out1 = aug1.augment(self.signal, sr=self.sr)
        out2, params2 = aug2.augment_with_params(self.signal, sr=self.sr)
        np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)
        self.assertIn("pad_mode", params2)
        self.assertIn("noise_type", params2)
        self.assertNotEqual(params2["noise_type"], "pink")


if __name__ == "__main__":
    unittest.main()
