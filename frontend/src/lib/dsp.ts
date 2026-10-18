/**
 * Verity Client-Side Digital Signal Processing (DSP) Engine
 *
 * Assumptions & Parameters:
 * - Window Function: Periodic Hann window w[n] = 0.5 * (1 - cos(2*pi*n / (N - 1)))
 * - Default FFT Size (N): 1024 samples
 * - Default Hop Size (H): 256 samples (75% overlap)
 * - Frequency Resolution: sampleRate / N (e.g., 15.625 Hz at 16 kHz, 43.066 Hz at 44.1 kHz)
 * - Decibel Reference: 0 dBFS (full-scale unity amplitude), Floor: -90 dBFS
 */

export interface AudioFileMetadata {
  filename: string;
  sizeBytes: number;
  durationSec: number;
  sampleRate: number;
  channels: number;
  peakAmplitude: number;
  clippedSamples: number;
  dcOffset: number;
}

export interface STFTResult {
  fftSize: number;
  hopSize: number;
  sampleRate: number;
  numFrames: number;
  numBins: number;
  durationSec: number;
  /** Flattened [numFrames * numBins] magnitude in dBFS (-90 to 0) */
  dbFrames: Float32Array;
  /** Flattened [numFrames * numBins] linear magnitude */
  linearFrames: Float32Array;
  /** Frequency in Hz for each bin [numBins] */
  binFrequencies: Float32Array;
}

export interface MelSpectrogramResult {
  numMels: number;
  numFrames: number;
  melDbFrames: Float32Array;
  melCenterFrequencies: Float32Array;
}

export interface AcousticFeatureSummary {
  intervalStartSec: number;
  intervalEndSec: number;
  spectralCentroidHz: number;
  spectralBandwidthHz: number;
  spectralRolloff85Hz: number;
  spectralFlatness: number;
  rmsLinear: number;
  rmsDb: number;
  zeroCrossingRatePerSec: number;
  dominantFrequencyHz: number;
  dominantPeakDb: number;
  mfccMeans: number[];
}

export const DSP_DEFAULTS = {
  FFT_SIZE: 1024,
  HOP_SIZE: 256,
  NUM_MELS: 64,
  DB_FLOOR: -90,
} as const;

export function createHannWindow(size: number): Float32Array {
  const win = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return win;
}

export function radix2FFT(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      const tr = real[j];
      real[j] = real[i];
      real[i] = tr;
      const ti = imag[j];
      imag[j] = imag[i];
      imag[i] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let s = 2; s <= n; s <<= 1) {
    const half = s >> 1;
    const theta = (-2 * Math.PI) / s;
    const wStepR = Math.cos(theta);
    const wStepI = Math.sin(theta);

    for (let k = 0; k < n; k += s) {
      let wR = 1;
      let wI = 0;
      for (let m = 0; m < half; m++) {
        const i1 = k + m;
        const i2 = i1 + half;
        const tR = wR * real[i2] - wI * imag[i2];
        const tI = wR * imag[i2] + wI * real[i2];
        real[i2] = real[i1] - tR;
        imag[i2] = imag[i1] - tI;
        real[i1] += tR;
        imag[i1] += tI;
        const nextWR = wR * wStepR - wI * wStepI;
        wI = wR * wStepI + wI * wStepR;
        wR = nextWR;
      }
    }
  }
}

export function extractMonoSamples(buffer: AudioBuffer): Float32Array {
  const length = buffer.length;
  const channels = buffer.numberOfChannels;
  if (channels === 1) {
    return new Float32Array(buffer.getChannelData(0));
  }
  const mono = new Float32Array(length);
  for (let ch = 0; ch < channels; ch++) {
    const chData = buffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += chData[i] / channels;
    }
  }
  return mono;
}

export function computeSTFT(
  samples: Float32Array,
  sampleRate: number,
  fftSize = DSP_DEFAULTS.FFT_SIZE,
  hopSize = DSP_DEFAULTS.HOP_SIZE
): STFTResult {
  const numBins = fftSize >> 1;
  const numFrames = Math.max(1, Math.floor((samples.length - fftSize) / hopSize) + 1);
  const hann = createHannWindow(fftSize);

  const dbFrames = new Float32Array(numFrames * numBins);
  const linearFrames = new Float32Array(numFrames * numBins);
  const binFrequencies = new Float32Array(numBins);

  for (let b = 0; b < numBins; b++) {
    binFrequencies[b] = (b * sampleRate) / fftSize;
  }

  const real = new Float32Array(fftSize);
  const imag = new Float32Array(fftSize);

  for (let f = 0; f < numFrames; f++) {
    const offset = f * hopSize;
    for (let i = 0; i < fftSize; i++) {
      const idx = offset + i;
      real[i] = (idx < samples.length ? samples[idx] : 0) * hann[i];
      imag[i] = 0;
    }

    radix2FFT(real, imag);

    const rowOffset = f * numBins;
    for (let b = 0; b < numBins; b++) {
      const mag = (Math.sqrt(real[b] * real[b] + imag[b] * imag[b]) * 2) / fftSize;
      linearFrames[rowOffset + b] = mag;
      const db = 20 * Math.log10(Math.max(mag, 1e-5));
      dbFrames[rowOffset + b] = Math.max(DSP_DEFAULTS.DB_FLOOR, Math.min(0, db));
    }
  }

  return {
    fftSize,
    hopSize,
    sampleRate,
    numFrames,
    numBins,
    durationSec: samples.length / sampleRate,
    dbFrames,
    linearFrames,
    binFrequencies,
  };
}
