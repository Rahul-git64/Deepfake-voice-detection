export interface PredictionProbabilities {
  bonafide: number;
  spoof: number;
}

export interface PredictionResponse {
  filename: string;
  prediction: 'bonafide' | 'spoof';
  label: 0 | 1;
  confidence: number;
  probabilities: PredictionProbabilities;
  inference_time_ms: number;
}

export interface HealthResponse {
  status: string;
  device: string;
  wav2vec2_loaded: boolean;
  classifier_loaded: boolean;
}

export type ApiErrorKind = 'offline' | 'unsupported_file' | 'timeout' | 'unknown';

export class DetectorApiError extends Error {
  public readonly kind: ApiErrorKind;

  constructor(message: string, kind: ApiErrorKind) {
    super(message);
    this.name = 'DetectorApiError';
    this.kind = kind;
  }
}

export function normalizePredictionResponse(raw: unknown): PredictionResponse {
  if (!raw || typeof raw !== 'object') {
    throw new DetectorApiError('Malformed response payload from server.', 'unknown');
  }
  const obj = raw as Record<string, unknown>;
  const prediction = obj.prediction === 'bonafide' ? 'bonafide' : 'spoof';
  const label = obj.label === 0 ? 0 : 1;
  const probsRaw = (obj.probabilities ?? {}) as Record<string, unknown>;

  return {
    filename: typeof obj.filename === 'string' ? obj.filename : 'audio.wav',
    prediction,
    label,
    confidence: Number(obj.confidence ?? 0),
    probabilities: {
      bonafide: Number(probsRaw.bonafide ?? 0),
      spoof: Number(probsRaw.spoof ?? 0),
    },
    inference_time_ms: Number(obj.inference_time_ms ?? 0),
  };
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch('/health', { method: 'GET' });
  if (!res.ok) {
    throw new DetectorApiError('Health check returned non-200 status', 'offline');
  }
  const data = (await res.json()) as Record<string, unknown>;
  return {
    status: typeof data.status === 'string' ? data.status : 'unknown',
    device: typeof data.device === 'string' ? data.device : 'cpu',
    wav2vec2_loaded: Boolean(data.wav2vec2_loaded),
    classifier_loaded: Boolean(data.classifier_loaded),
  };
}

export async function postPredictAudio(file: File, timeoutMs = 25000): Promise<PredictionResponse> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!res.ok) {
      let detail = `Server responded with status ${res.status}`;
      try {
        const errJson = (await res.json()) as Record<string, unknown>;
        if (typeof errJson.detail === 'string') {
          detail = errJson.detail;
        }
      } catch {
        // Ignore JSON parse fallback
      }
      if (res.status === 400 || res.status === 422) {
        throw new DetectorApiError(detail, 'unsupported_file');
      }
      throw new DetectorApiError(detail, 'offline');
    }

    const raw = await res.json();
    return normalizePredictionResponse(raw);
  } catch (err: unknown) {
    if (err instanceof DetectorApiError) {
      throw err;
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new DetectorApiError('Request timed out while analyzing audio clip.', 'timeout');
    }
    throw new DetectorApiError(
      'Unable to reach the FastAPI inference engine. Verify uvicorn is running on port 8080.',
      'offline'
    );
  } finally {
    window.clearTimeout(timer);
  }
}
