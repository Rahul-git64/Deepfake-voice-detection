import { useState, useRef, useCallback, useEffect } from 'react';
import { MAX_RECORD_SECONDS } from '../lib/config';

export interface RecorderControls {
  isRecording: boolean;
  secondsElapsed: number;
  liveLevel: number;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
}

export function useRecorder(
  onRecordingComplete: (file: File) => void,
  onError: (msg: string) => void,
  connectStreamToAnalyser?: (stream: MediaStream) => void,
  getRmsLevel?: () => number
): RecorderControls {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [secondsElapsed, setSecondsElapsed] = useState<number>(0);
  const [liveLevel, setLiveLevel] = useState<number>(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  const stopRecording = useCallback((): void => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (rafRef.current) {
      window.cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setIsRecording(false);
    setLiveLevel(0);
  }, []);

  const startRecording = useCallback(async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (connectStreamToAnalyser) {
        connectStreamToAnalyser(stream);
      }

      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];

      mr.ondataavailable = (e: BlobEvent): void => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mr.onstop = (): void => {
        const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
        const file = new File([blob], `mic_capture_${Date.now()}.wav`, { type: 'audio/wav' });
        onRecordingComplete(file);
        stream.getTracks().forEach((t) => t.stop());
      };

      mr.start();
      setIsRecording(true);
      setSecondsElapsed(0);

      timerRef.current = window.setInterval(() => {
        setSecondsElapsed((prev) => {
          if (prev + 1 >= MAX_RECORD_SECONDS) {
            stopRecording();
            return MAX_RECORD_SECONDS;
          }
          return prev + 1;
        });
      }, 1000);

      const updateLevel = (): void => {
        if (getRmsLevel) {
          setLiveLevel(getRmsLevel());
        }
        rafRef.current = window.requestAnimationFrame(updateLevel);
      };
      rafRef.current = window.requestAnimationFrame(updateLevel);
    } catch {
      onError('Microphone permission was denied or no audio input device was detected.');
    }
  }, [connectStreamToAnalyser, getRmsLevel, onError, onRecordingComplete, stopRecording]);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return {
    isRecording,
    secondsElapsed,
    liveLevel,
    startRecording,
    stopRecording,
  };
}
