import { ApiErrorKind, PredictionResponse } from '../lib/api';

export type DetectorStatus =
  | 'idle'
  | 'loaded'
  | 'analyzing'
  | 'result_bonafide'
  | 'result_spoof'
  | 'error';

export interface SessionHistoryItem extends PredictionResponse {
  id: string;
  timestamp: string;
  runIndex: number;
}

export interface AudioMetadata {
  durationSec: number;
  sampleRate: number;
  channels: number;
  sizeBytes: number;
}

export interface DetectorState {
  status: DetectorStatus;
  file: File | null;
  audioUrl: string | null;
  audioMeta: AudioMetadata | null;
  analysisStep: 0 | 1 | 2;
  result: PredictionResponse | null;
  errorMsg: string | null;
  errorKind: ApiErrorKind | null;
  history: SessionHistoryItem[];
}

export type DetectorAction =
  | { type: 'LOAD_FILE'; file: File; audioUrl: string; audioMeta: AudioMetadata | null }
  | { type: 'SET_AUDIO_META'; audioMeta: AudioMetadata }
  | { type: 'START_ANALYZING' }
  | { type: 'SET_ANALYSIS_STEP'; step: 0 | 1 | 2 }
  | { type: 'SET_RESULT'; result: PredictionResponse }
  | { type: 'SET_ERROR'; message: string; kind: ApiErrorKind }
  | { type: 'RESET' };

export const initialDetectorState: DetectorState = {
  status: 'idle',
  file: null,
  audioUrl: null,
  audioMeta: null,
  analysisStep: 0,
  result: null,
  errorMsg: null,
  errorKind: null,
  history: [],
};

export function detectorReducer(state: DetectorState, action: DetectorAction): DetectorState {
  switch (action.type) {
    case 'LOAD_FILE':
      return {
        ...state,
        status: 'loaded',
        file: action.file,
        audioUrl: action.audioUrl,
        audioMeta: action.audioMeta,
        result: null,
        errorMsg: null,
        errorKind: null,
      };
    case 'SET_AUDIO_META':
      return {
        ...state,
        audioMeta: action.audioMeta,
      };
    case 'START_ANALYZING':
      return {
        ...state,
        status: 'analyzing',
        analysisStep: 0,
        errorMsg: null,
        errorKind: null,
      };
    case 'SET_ANALYSIS_STEP':
      return {
        ...state,
        analysisStep: action.step,
      };
    case 'SET_RESULT': {
      const nextStatus: DetectorStatus =
        action.result.prediction === 'bonafide' ? 'result_bonafide' : 'result_spoof';
      const entry: SessionHistoryItem = {
        ...action.result,
        id: `${Date.now()}-${state.history.length + 1}`,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
        runIndex: state.history.length + 1,
      };
      return {
        ...state,
        status: nextStatus,
        result: action.result,
        history: [...state.history, entry],
      };
    }
    case 'SET_ERROR':
      return {
        ...state,
        status: 'error',
        errorMsg: action.message,
        errorKind: action.kind,
      };
    case 'RESET':
      return {
        ...state,
        status: 'idle',
        file: null,
        audioUrl: null,
        audioMeta: null,
        result: null,
        errorMsg: null,
        errorKind: null,
      };
    default:
      return state;
  }
}
