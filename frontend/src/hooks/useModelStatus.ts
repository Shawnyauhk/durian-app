/**
 * useModelStatus.ts
 * 
 * Global hook to track AI model availability for both vision and acoustic modalities.
 * Polls the backend health endpoint and checks frontend model state.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import {
  isVisionModelReady,
  getVisionModelType,
  isAcousticModelReady,
  getAcousticModelType,
} from '../utils/tf-helpers';

const API_BASE = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? 'https://durian-ai-api.onrender.com' : 'http://localhost:8000');

const POLL_INTERVAL_MS = 30_000; // 30s

export interface SystemModelStatus {
  /** Backend reachable */
  backendOnline: boolean;
  /** Backend using real acoustic AI model */
  acousticModelOnBackend: boolean;
  /** Frontend vision model loaded */
  visionModelOnFrontend: boolean;
  /** Frontend acoustic model loaded (optional — normally backend handles this) */
  acousticModelOnFrontend: boolean;
  /** Overall: is the system using real AI or heuristics? */
  usingRealAI: boolean;
  /** Details */
  visionMode: 'ai' | 'heuristic';
  acousticMode: 'ai' | 'heuristic';
  /** Backend feedback count */
  feedbackCount: number;
  /** Acoustic model version if available */
  acousticModelVersion: string | null;
  /** Last polled */
  lastChecked: Date | null;
  /** Loading state */
  checking: boolean;
}

const DEFAULT_STATUS: SystemModelStatus = {
  backendOnline: false,
  acousticModelOnBackend: false,
  visionModelOnFrontend: false,
  acousticModelOnFrontend: false,
  usingRealAI: false,
  visionMode: 'heuristic',
  acousticMode: 'heuristic',
  feedbackCount: 0,
  acousticModelVersion: null,
  lastChecked: null,
  checking: true,
};

export function useModelStatus() {
  const [status, setStatus] = useState<SystemModelStatus>(DEFAULT_STATUS);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const checkStatus = useCallback(async () => {
    setStatus(prev => ({ ...prev, checking: true }));

    let backendOnline = false;
    let acousticModelOnBackend = false;
    let feedbackCount = 0;
    let acousticModelVersion: string | null = null;

    // Check backend health
    try {
      const resp = await fetch(`${API_BASE}/api/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (resp.ok) {
        backendOnline = true;
        const data = await resp.json();
        feedbackCount = data.feedback_count ?? 0;
      }
    } catch {
      backendOnline = false;
    }

    // Check backend model status
    if (backendOnline) {
      try {
        const resp = await fetch(`${API_BASE}/api/model-status`, {
          signal: AbortSignal.timeout(5000),
        });
        if (resp.ok) {
          const data = await resp.json();
          acousticModelOnBackend = data.acoustic_model_loaded === true;
          acousticModelVersion = data.acoustic_model_version ?? null;
        }
      } catch {
        // Model status endpoint may not exist yet
        acousticModelOnBackend = false;
      }
    }

    // Frontend model states
    const visionModelOnFrontend = isVisionModelReady();
    const acousticModelOnFrontend = isAcousticModelReady();
    const visionMode = getVisionModelType();
    const acousticMode = getAcousticModelType();

    const usingRealAI = visionModelOnFrontend || acousticModelOnBackend;

    setStatus({
      backendOnline,
      acousticModelOnBackend,
      visionModelOnFrontend,
      acousticModelOnFrontend,
      usingRealAI,
      visionMode,
      acousticMode,
      feedbackCount,
      acousticModelVersion,
      lastChecked: new Date(),
      checking: false,
    });
  }, []);

  // Initial check + periodic polling
  useEffect(() => {
    checkStatus();
    pollRef.current = setInterval(checkStatus, POLL_INTERVAL_MS);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [checkStatus]);

  return { status, refresh: checkStatus };
}
