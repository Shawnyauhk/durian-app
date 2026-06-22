import { useState, useCallback, useEffect } from 'react';
import {
  classifyDurianRipeness,
  loadVisionModel,
  resetVisionModel,
  isVisionModelReady,
  getVisionModelType,
  getVisionModelStatus,
} from '../utils/tf-helpers';
import type { ModalityResult } from '../types';
import type { ModelStatus } from '../utils/tf-helpers';

export function useVisionModel() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelType, setModelType] = useState<'ai' | 'heuristic'>('heuristic');
  const [modelStatus, setModelStatus] = useState<ModelStatus>(getVisionModelStatus());

  // Attempt model load on mount
  useEffect(() => {
    let cancelled = false;

    const attemptLoad = async () => {
      const loaded = await loadVisionModel();
      if (cancelled) return;
      setModelType(getVisionModelType());
      setModelStatus(getVisionModelStatus());
      if (loaded) {
        console.log('[useVisionModel] AI model ready');
      } else {
        console.log('[useVisionModel] Using heuristic fallback');
      }
    };

    attemptLoad();
    return () => { cancelled = true; };
  }, []);

  /** Re-attempt model load (useful after user deploys model). */
  const retryModelLoad = useCallback(async () => {
    resetVisionModel();
    const loaded = await loadVisionModel();
    setModelType(getVisionModelType());
    setModelStatus(getVisionModelStatus());
    return loaded;
  }, []);

  const classify = useCallback(async (canvas: HTMLCanvasElement): Promise<ModalityResult> => {
    setIsLoading(true);
    setError(null);

    // Try loading model if not yet attempted
    if (!isVisionModelReady()) {
      await loadVisionModel();
      setModelType(getVisionModelType());
      setModelStatus(getVisionModelStatus());
    }

    try {
      const result = await classifyDurianRipeness(canvas);
      return result;
    } catch (err) {
      const msg = err instanceof Error ? err.message : '視覺分析失敗';
      setError(msg);
      return {
        ripeness: 'unknown',
        scores: { unripe: 0.33, ripe: 0.34, overripe: 0.33 },
        confidence: 0,
        available: false,
        error: msg,
      };
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { classify, isLoading, error, modelType, modelStatus, retryModelLoad };
}
