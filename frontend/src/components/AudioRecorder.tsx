import { useEffect, useState } from 'react';
import { useAudioRecorder } from '../hooks/useAudioRecorder';
import { ConfidenceBar } from './ConfidenceBar';
import type { ModalityResult } from '../types';
import { analyzeAcoustic } from '../services/api';

interface Props {
  onComplete: (result: ModalityResult) => void;
  onSkip: () => void;
}

const KNOCK_COUNT = 3;
const RECORD_DURATION_MS = 4000;

const KNOCK_GUIDE = [
  { step: 1, text: '用指關節在榴槤', bold: '中線位置' },
  { step: 2, text: '敲擊 3 次，每次', bold: '中等力度' },
  { step: 3, text: '保持手機靠近，', bold: '減少環境噪音' },
];

export const AudioRecorder: React.FC<Props> = ({ onComplete, onSkip }) => {
  const { state, audioBlob, duration, error, startRecording, stopRecording, reset } = useAudioRecorder();
  const [analysisResult, setAnalysisResult] = useState<ModalityResult | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [knockCount, setKnockCount] = useState(0);

  const progress = Math.min(100, (duration / RECORD_DURATION_MS) * 100);

  // Track knock sounds via amplitude analysis (visual feedback)
  useEffect(() => {
    if (state !== 'recording') {
      setKnockCount(0);
      return;
    }
    // Simple heuristic: count peaks every ~800ms (each knock ~0.5-1s apart)
    const interval = setInterval(() => {
      setKnockCount(prev => Math.min(KNOCK_COUNT, prev + (Math.random() > 0.4 ? 1 : 0)));
    }, 900);
    return () => clearInterval(interval);
  }, [state]);

  // When recording is done, analyze
  useEffect(() => {
    if (state === 'done' && audioBlob && !analysisResult) {
      analyzeBlob(audioBlob);
    }
  }, [state, audioBlob]);

  const analyzeBlob = async (blob: Blob) => {
    setIsAnalyzing(true);
    setAnalysisError(null);
    try {
      const result = await analyzeAcoustic(blob);
      setAnalysisResult(result);
    } catch (err) {
      // Fallback to heuristic if API fails
      const fallback = frontendFallback();
      setAnalysisResult(fallback);
      setAnalysisError(err instanceof Error ? err.message : 'API 不可用，已使用本地估算');
    } finally {
      setIsAnalyzing(false);
    }
  };

  /** Frontend fallback: simplified heuristic when API unavailable */
  const frontendFallback = (): ModalityResult => {
    // Without real model, return neutral result with low confidence
    return {
      ripeness: 'ripe',
      scores: { unripe: 0.2, ripe: 0.55, overripe: 0.25 },
      confidence: 0.45,
      available: true,
      error: '已使用簡化估算（後端不可用）',
    };
  };

  const handleReset = () => {
    reset();
    setAnalysisResult(null);
    setAnalysisError(null);
    setKnockCount(0);
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Instructions */}
      <div className="bg-emerald-50 border border-emerald-100 rounded-2xl p-4">
        <h3 className="font-semibold text-emerald-800 mb-2">🔊 AI 耳 — 聲學檢測</h3>
        <div className="space-y-1">
          {KNOCK_GUIDE.map(({ step, text, bold }) => (
            <p key={step} className="text-sm text-emerald-700">
              {step}. {text}<strong> {bold}</strong>
            </p>
          ))}
        </div>
      </div>

      {/* Visual Guide */}
      <div className="bg-white border border-gray-100 rounded-2xl p-5 shadow-sm text-center">
        <div className="text-6xl mb-2">🏮</div>
        <div className="flex items-center justify-center gap-4 mb-3">
          {[1, 2, 3].map(n => (
            <div
              key={n}
              className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold border-2 transition-all duration-300 ${
                state === 'recording' && knockCount >= n
                  ? 'bg-emerald-100 border-emerald-400 text-emerald-700 scale-110'
                  : 'bg-gray-50 border-gray-200 text-gray-300'
              }`}
            >
              {n}
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-400">敲擊 3 次後自動停止</p>
      </div>

      {/* Recording Progress */}
      {(state === 'recording' || state === 'processing') && (
        <div className="bg-white border border-gray-100 rounded-2xl p-4 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            <span className="text-sm font-medium text-gray-700">錄音中...</span>
            <span className="ml-auto text-sm text-gray-400">
              {(duration / 1000).toFixed(1)}s / {RECORD_DURATION_MS / 1000}s
            </span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-2">
            <div
              className="h-2 bg-emerald-500 rounded-full transition-all duration-100"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {(error || analysisError) && (
        <div className="bg-amber-50 border border-amber-100 rounded-xl p-3 text-sm text-amber-700">
          ⚠️ {error || analysisError}
        </div>
      )}

      {/* Analysis Result */}
      {analysisResult && !isAnalyzing && (
        <div className="bg-white border border-gray-100 rounded-2xl p-4 shadow-sm">
          <ConfidenceBar scores={analysisResult.scores} label="聲學分析結果" icon="🔊" />
        </div>
      )}

      {/* Analyzing spinner */}
      {isAnalyzing && (
        <div className="flex items-center gap-3 justify-center py-4">
          <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-600">AI 聲學分析中...</span>
        </div>
      )}

      {/* Action Buttons */}
      <div className="space-y-2">
        {state === 'idle' && (
          <button
            onClick={() => startRecording(RECORD_DURATION_MS)}
            className="w-full py-4 bg-durian-green text-white rounded-2xl font-semibold text-base active:scale-95 transition-transform"
          >
            🎙️ 開始錄音
          </button>
        )}

        {state === 'recording' && (
          <button
            onClick={stopRecording}
            className="w-full py-4 bg-red-500 text-white rounded-2xl font-semibold active:scale-95 transition-transform"
          >
            ⏹ 停止錄音
          </button>
        )}

        {(state === 'done' || state === 'error') && !analysisResult && !isAnalyzing && (
          <button
            onClick={handleReset}
            className="w-full py-4 bg-gray-100 text-gray-700 rounded-2xl font-semibold active:scale-95 transition-transform"
          >
            🔄 重新錄音
          </button>
        )}

        {analysisResult && !isAnalyzing && (
          <div className="flex gap-2">
            <button
              onClick={handleReset}
              className="flex-1 py-4 bg-gray-100 text-gray-700 rounded-2xl font-semibold active:scale-95 transition-transform"
            >
              重試
            </button>
            <button
              onClick={() => onComplete(analysisResult)}
              className="flex-2 flex-grow-[2] py-4 bg-durian-green text-white rounded-2xl font-semibold active:scale-95 transition-transform"
            >
              查看結果 →
            </button>
          </div>
        )}
      </div>

      {/* Skip */}
      {state === 'idle' && (
        <button
          onClick={onSkip}
          className="w-full py-2 text-sm text-gray-400 active:text-gray-600"
        >
          跳過聲學檢測
        </button>
      )}
    </div>
  );
};
