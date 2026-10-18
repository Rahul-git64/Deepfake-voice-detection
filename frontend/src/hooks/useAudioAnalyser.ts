import { useRef, useCallback, useEffect } from 'react';

export interface AudioAnalyserControls {
  connectMediaElement: (audioEl: HTMLMediaElement) => void;
  connectStream: (stream: MediaStream) => void;
  getFrequencyBars: (barCount: number) => Float32Array;
  getRmsLevel: () => number;
}

export function useAudioAnalyser(): AudioAnalyserControls {
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceMapRef = useRef<WeakMap<HTMLMediaElement, MediaElementAudioSourceNode>>(new WeakMap());
  const streamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const smoothedBarsRef = useRef<Float32Array>(new Float32Array(64));

  const ensureContext = useCallback((): { ctx: AudioContext; analyser: AnalyserNode } => {
    if (!audioCtxRef.current) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      audioCtxRef.current = new AudioCtx();
    }
    const ctx = audioCtxRef.current;
    if (ctx.state === 'suspended') {
      void ctx.resume();
    }
    if (!analyserRef.current) {
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.78;
      analyserRef.current = analyser;
    }
    return { ctx, analyser: analyserRef.current };
  }, []);

  const connectMediaElement = useCallback(
    (audioEl: HTMLMediaElement): void => {
      const { ctx, analyser } = ensureContext();
      let source = sourceMapRef.current.get(audioEl);
      if (!source) {
        source = ctx.createMediaElementSource(audioEl);
        sourceMapRef.current.set(audioEl, source);
      }
      try {
        source.disconnect();
      } catch {
        // Ignore if not connected
      }
      source.connect(analyser);
      analyser.connect(ctx.destination);
    },
    [ensureContext]
  );

  const connectStream = useCallback(
    (stream: MediaStream): void => {
      const { ctx, analyser } = ensureContext();
      if (streamSourceRef.current) {
        try {
          streamSourceRef.current.disconnect();
        } catch {
          // Ignore
        }
      }
      const streamSource = ctx.createMediaStreamSource(stream);
      streamSource.connect(analyser);
      streamSourceRef.current = streamSource;
    },
    [ensureContext]
  );

  const getFrequencyBars = useCallback((barCount = 64): Float32Array => {
    if (smoothedBarsRef.current.length !== barCount) {
      smoothedBarsRef.current = new Float32Array(barCount);
    }
    const smoothed = smoothedBarsRef.current;
    const analyser = analyserRef.current;

    if (!analyser) {
      for (let i = 0; i < barCount; i++) {
        smoothed[i] *= 0.88;
      }
      return smoothed;
    }

    const bufferLength = analyser.frequencyBinCount;
    const rawData = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(rawData);

    for (let i = 0; i < barCount; i++) {
      const binIdx = Math.min(bufferLength - 1, Math.floor((i / barCount) * (bufferLength * 0.75)));
      const target = rawData[binIdx] / 255.0;
      smoothed[i] = smoothed[i] + (target - smoothed[i]) * 0.25; // Lerp smoothing
    }
    return smoothed;
  }, []);

  const getRmsLevel = useCallback((): number => {
    const analyser = analyserRef.current;
    if (!analyser) return 0;
    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);
    let sumSq = 0;
    for (let i = 0; i < data.length; i++) {
      const norm = (data[i] - 128) / 128.0;
      sumSq += norm * norm;
    }
    return Math.min(1, Math.sqrt(sumSq / data.length) * 3.2);
  }, []);

  useEffect(() => {
    return () => {
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        void audioCtxRef.current.close();
      }
    };
  }, []);

  return {
    connectMediaElement,
    connectStream,
    getFrequencyBars,
    getRmsLevel,
  };
}
