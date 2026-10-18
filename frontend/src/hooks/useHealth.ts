import { useState, useEffect } from 'react';
import { fetchHealth } from '../lib/api';
import { HEALTH_POLL_INTERVAL_MS } from '../lib/config';

export interface HealthStatusState {
  online: boolean;
  device: string;
  modelReady: boolean;
  lastChecked: string | null;
}

export function useHealth(): HealthStatusState {
  const [health, setHealth] = useState<HealthStatusState>({
    online: true,
    device: 'cpu',
    modelReady: true,
    lastChecked: null,
  });

  useEffect(() => {
    let mounted = true;

    const check = async (): Promise<void> => {
      try {
        const data = await fetchHealth();
        if (!mounted) return;
        setHealth({
          online: data.status === 'healthy',
          device: data.device,
          modelReady: data.wav2vec2_loaded && data.classifier_loaded,
          lastChecked: new Date().toLocaleTimeString(),
        });
      } catch {
        if (!mounted) return;
        setHealth((prev) => ({
          ...prev,
          online: false,
          modelReady: false,
        }));
      }
    };

    void check();
    const interval = window.setInterval(() => {
      void check();
    }, HEALTH_POLL_INTERVAL_MS);

    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  return health;
}
