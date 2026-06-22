import type { ModalityResult } from '../types';

// Backend URL — fallback to localhost in dev, production URL when deployed
const API_BASE = import.meta.env.VITE_API_URL || 
  (import.meta.env.PROD ? 'https://durian-ai-api.onrender.com' : 'http://localhost:8000');

export async function analyzeAcoustic(audioBlob: Blob): Promise<ModalityResult> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'knock.webm');

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

  try {
    const response = await fetch(`${API_BASE}/api/analyze-acoustic`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    return {
      ripeness: data.ripeness,
      scores: data.scores,
      confidence: data.confidence,
      available: true,
    };
  } catch (err) {
    clearTimeout(timeoutId);
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('請求超時，請重試');
    }
    throw err;
  }
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/api/health`, {
      signal: AbortSignal.timeout(5000),
    });
    return response.ok;
  } catch {
    return false;
  }
}
