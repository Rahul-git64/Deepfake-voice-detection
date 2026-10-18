import React, { useReducer, useState } from 'react';
import { detectorReducer, initialDetectorState } from './state/detectorState';
import { useHealth } from './hooks/useHealth';
import { useAudioAnalyser } from './hooks/useAudioAnalyser';
import { useRecorder } from './hooks/useRecorder';
import { postPredictAudio, DetectorApiError } from './lib/api';
import {
  MODEL_FACTS,
  MAX_FILE_SIZE_BYTES,
  MIN_ANALYZING_DURATION_MS,
  getRiskLabel,
} from './lib/config';

export function App(): React.JSX.Element {
  const [state, dispatch] = useReducer(detectorReducer, initialDetectorState);
  const health = useHealth();
  const analyser = useAudioAnalyser();
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);

  const handleFileSelected = (file: File): void => {
    if (file.size > MAX_FILE_SIZE_BYTES) {
      dispatch({
        type: 'SET_ERROR',
        message: 'Audio file exceeds the 10 MB limit.',
        kind: 'unsupported_file',
      });
      return;
    }
    const url = URL.createObjectURL(file);
    dispatch({
      type: 'LOAD_FILE',
      file,
      audioUrl: url,
      audioMeta: {
        durationSec: 0,
        sampleRate: 16000,
        channels: 1,
        sizeBytes: file.size,
      },
    });
  };

  const recorder = useRecorder(
    handleFileSelected,
    (msg) => dispatch({ type: 'SET_ERROR', message: msg, kind: 'unsupported_file' }),
    analyser.connectStream,
    analyser.getRmsLevel
  );

  const handleAnalyze = async (): Promise<void> => {
    if (!state.file || state.status === 'analyzing') return;
    dispatch({ type: 'START_ANALYZING' });

    const t1 = window.setTimeout(() => dispatch({ type: 'SET_ANALYSIS_STEP', step: 1 }), 600);
    const t2 = window.setTimeout(() => dispatch({ type: 'SET_ANALYSIS_STEP', step: 2 }), 1200);
    const start = performance.now();

    try {
      const result = await postPredictAudio(state.file);
      const elapsed = performance.now() - start;
      const wait = Math.max(0, MIN_ANALYZING_DURATION_MS - elapsed);
      window.setTimeout(() => {
        dispatch({ type: 'SET_RESULT', result });
      }, wait);
    } catch (err: unknown) {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      if (err instanceof DetectorApiError) {
        dispatch({ type: 'SET_ERROR', message: err.message, kind: err.kind });
      } else {
        dispatch({
          type: 'SET_ERROR',
          message: 'Unexpected error during analysis.',
          kind: 'unknown',
        });
      }
    }
  };

  const spoofProb = state.result ? state.result.probabilities.spoof : 0;
  const riskLabel = state.result ? getRiskLabel(state.result.prediction, spoofProb) : 'Authentic';

  return (
    <div className="min-h-screen bg-[#07080d] text-[#e8ecf4]">
      <header className="sticky top-0 z-50 border-b border-white/10 bg-[#07080d]/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-6">
          <span className="font-bold tracking-tight">VOXFORENSICS</span>
          <div className="font-mono text-xs">
            DEVICE: {health.device.toUpperCase()} • {health.online && health.modelReady ? 'Model ready' : 'Offline'}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-12">
        <h1 className="text-center text-5xl font-bold">Can you trust that voice?</h1>
        <p className="mt-4 text-center text-[#8b93a7]">
          Dataset: {MODEL_FACTS.dataset} • EER: {MODEL_FACTS.devEer} • Accuracy: {MODEL_FACTS.devAccuracy}
        </p>

        <div className="mt-8 rounded-2xl border border-white/10 bg-white/[0.04] p-8 backdrop-blur-xl">
          <div className="flex items-center justify-between gap-4">
            <button
              type="button"
              onClick={() => (recorder.isRecording ? recorder.stopRecording() : void recorder.startRecording())}
              className="rounded-lg border border-pink-500/40 bg-pink-500/10 px-4 py-2 text-sm"
            >
              {recorder.isRecording ? `Stop (${recorder.secondsElapsed}s)` : 'Record from mic'}
            </button>
            <button
              type="button"
              onClick={() => setPlaybackRate((r) => (r === 1 ? 1.25 : 1))}
              className="font-mono text-xs text-[#8b93a7]"
            >
              Speed: {playbackRate}x
            </button>
          </div>

          <button
            type="button"
            disabled={!state.file || state.status === 'analyzing'}
            onClick={() => void handleAnalyze()}
            className="mt-6 w-full rounded-xl bg-cyan-400 py-3 font-semibold text-slate-950"
          >
            {state.status === 'analyzing' ? 'Analyzing...' : 'Analyze'}
          </button>

          {state.result && (
            <div className="mt-6 rounded-xl border border-white/15 p-6">
              <h2 className="text-2xl font-bold">
                {state.result.prediction === 'bonafide' ? 'Human voice verified' : 'Synthetic voice detected'}
              </h2>
              <p className="font-mono text-sm">
                Risk: {riskLabel} • Spoof Prob: {(spoofProb * 100).toFixed(2)}% • {state.result.inference_time_ms} ms
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
export default App;
