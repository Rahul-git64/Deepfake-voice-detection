/**
 * Verity Modular Frontend Components Index
 * Separates rendering components from the DSP engine in ../lib/dsp.ts and API service in ../lib/api.ts
 */

export type SpectralViewMode = 'magnitude' | 'mel' | 'average';
export type FrequencyScaleMode = 'linear' | 'log';

export interface TimeRegionSelection {
  startSec: number;
  endSec: number;
}

export interface ExplainableDemoPreset {
  mode: 'human' | 'synthetic' | 'inconclusive';
  isIllustrative: true;
}
