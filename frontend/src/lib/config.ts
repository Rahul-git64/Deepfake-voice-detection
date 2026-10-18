export interface RiskThresholds {
  readonly high: number;
  readonly medium: number;
}

export const RISK_THRESHOLDS: RiskThresholds = {
  high: 0.85,
  medium: 0.6,
} as const;

export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
export const ALLOWED_EXTENSIONS = ['.wav', '.flac', '.mp3', '.m4a'] as const;
export const MIN_ANALYZING_DURATION_MS = 1800;
export const HEALTH_POLL_INTERVAL_MS = 15000;
export const MAX_RECORD_SECONDS = 30;

export const MODEL_FACTS = {
  dataset: 'ASVspoof 2019 Logical Access (LA)',
  trainClips: '25,380',
  devClips: '24,844',
  featureExtractor: 'facebook/wav2vec2-base',
  sampleRate: '16 kHz mono',
  transformerBlocks: 12,
  embeddingDim: 768,
  classifierArch: '768 → Dense 256 (BatchNorm, ReLU, Dropout 0.3) → Dense 64 (ReLU, Dropout 0.3) → 2 classes',
  trainingDetails: 'Balanced class weights (spoof:bonafide ~9:1), AdamW (lr 1e-3), batch size 128',
  devEer: '1.41%',
  eerThreshold: '0.6402',
  devAccuracy: '99.22%',
  typicalLatency: 'A few hundred ms on CPU',
} as const;

export type RiskLevel = 'Authentic' | 'High' | 'Medium' | 'Low';

export function getRiskLabel(prediction: 'bonafide' | 'spoof', spoofProbability: number): RiskLevel {
  if (prediction === 'bonafide') {
    return 'Authentic';
  }
  if (spoofProbability >= RISK_THRESHOLDS.high) {
    return 'High';
  }
  if (spoofProbability >= RISK_THRESHOLDS.medium) {
    return 'Medium';
  }
  return 'Low';
}
