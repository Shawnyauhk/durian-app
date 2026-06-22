import React, { useState } from 'react';
import type { FusionResult, RipenessLevel } from '../types';
import { RIPENESS_LABELS, RIPENESS_EMOJI, RIPENESS_BG } from '../types';
import { ConfidenceBar } from './ConfidenceBar';
import { ModelStatusBadge } from './ModelStatusBadge';
import type { SystemModelStatus } from '../hooks/useModelStatus';

interface Props {
  result: FusionResult;
  onReset: () => void;
  modelStatus?: SystemModelStatus;
}

const PURCHASE_ADVICE: Record<string, { icon: string; text: string; detail: string; color: string }> = {
  unripe: {
    icon: '🚫',
    text: '不建議購買',
    detail: '榴槤尚未成熟，果肉硬、甜度不足，建議等待或選擇其他果實。',
    color: 'text-amber-700',
  },
  ripe: {
    icon: '✅',
    text: '可以購買食用',
    detail: '榴槤已達最佳成熟度，果肉軟糯香甜，建議立即購買。',
    color: 'text-emerald-700',
  },
  overripe: {
    icon: '⚠️',
    text: '不建議購買',
    detail: '榴槤過熟，果肉可能酒味過重或開始腐敗，謹慎購買。',
    color: 'text-red-700',
  },
  unknown: {
    icon: '❓',
    text: '無法判定',
    detail: '數據不足，請重新檢測。',
    color: 'text-gray-600',
  },
};

const RIPENESS_OPTIONS = [
  { value: 'unripe', label: '未熟', emoji: '🟢' },
  { value: 'ripe', label: '成熟', emoji: '🟡' },
  { value: 'overripe', label: '過熟', emoji: '🔴' },
];

type FeedbackState = 'idle' | 'selecting' | 'submitting' | 'done' | 'error';

export const ResultCard: React.FC<Props> = ({ result, onReset, modelStatus }) => {
  const advice = PURCHASE_ADVICE[result.ripeness];
  const bgClass = RIPENESS_BG[result.ripeness];
  const confidencePct = Math.round(result.confidence * 100);
  const confidenceLabel =
    result.confidence >= 0.75 ? '高置信度' :
    result.confidence >= 0.5 ? '中置信度' :
    '低置信度';
  const confidenceColor =
    result.confidence >= 0.75 ? 'text-emerald-600' :
    result.confidence >= 0.5 ? 'text-amber-600' :
    'text-red-500';

  // ── Feedback state ──
  const [feedbackState, setFeedbackState] = useState<FeedbackState>('idle');
  const [selectedLabel, setSelectedLabel] = useState<string>('');

  const handleFeedbackStart = () => {
    setSelectedLabel(result.ripeness); // Pre-select AI's answer
    setFeedbackState('selecting');
  };

  const handleFeedbackSubmit = async () => {
    if (!selectedLabel) return;
    setFeedbackState('submitting');

    try {
      const payload = {
        ai_prediction: result.ripeness,
        user_label: selectedLabel,
        confidence: result.confidence,
        acoustic_scores: result.acoustic?.scores ?? null,
        vision_scores: result.vision?.scores ?? null,
        timestamp: new Date().toISOString(),
      };

      // Try to submit to backend
      const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
      await fetch(`${API_BASE}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      setFeedbackState('done');
    } catch {
      // Even if backend is unavailable, mark as done (offline mode)
      setFeedbackState('done');
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Main Result */}
      <div className={`border-2 rounded-2xl p-5 ${bgClass}`}>
        <div className="text-center mb-4">
          <div className="text-6xl mb-2">{RIPENESS_EMOJI[result.ripeness]}</div>
          <div className="text-3xl font-bold text-gray-800 mb-1">
            {RIPENESS_LABELS[result.ripeness]}
          </div>
          <div className={`text-sm font-medium ${confidenceColor}`}>
            {confidenceLabel} ({confidencePct}%)
          </div>
        </div>

        {/* Confidence Bar - overall */}
        <div className="bg-white/60 rounded-xl p-3 mb-3">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>未熟</span>
            <span>成熟</span>
            <span>過熟</span>
          </div>
          <div className="relative h-3 bg-gradient-to-r from-amber-200 via-emerald-200 to-red-200 rounded-full">
            <div
              className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white border-2 border-gray-400 rounded-full shadow transition-all duration-700"
              style={{ left: `${result.weightedScore * 100}%`, transform: 'translateX(-50%) translateY(-50%)' }}
            />
          </div>
        </div>

        {/* Advice */}
        <div className="flex items-start gap-2">
          <span className="text-xl">{advice.icon}</span>
          <div>
            <p className={`font-semibold text-sm ${advice.color}`}>{advice.text}</p>
            <p className="text-xs text-gray-600 mt-0.5">{advice.detail}</p>
          </div>
        </div>
      </div>

      {/* Warnings */}
      {result.contradictionWarning && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-3 text-sm text-amber-700">
          ⚡ 視覺與聲學結果差異較大，建議再次檢測確認。
        </div>
      )}

      {/* Modality Breakdown */}
      <div className="bg-white border border-gray-100 rounded-2xl p-4 shadow-sm space-y-4">
        <h4 className="text-sm font-semibold text-gray-700">各模態詳情</h4>

        {result.acoustic ? (
          <div>
            <ConfidenceBar scores={result.acoustic.scores} label="AI 耳（聲學）" icon="🔊" />
            {result.acoustic.error && (
              <p className="text-xs text-amber-600 mt-1">⚠️ {result.acoustic.error}</p>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>🔊</span><span>AI 耳（聲學）— 未檢測</span>
          </div>
        )}

        {result.vision ? (
          <div>
            <ConfidenceBar scores={result.vision.scores} label="AI 眼（視覺）" icon="📸" />
            {result.vision.error && (
              <p className="text-xs text-amber-600 mt-1">⚠️ {result.vision.error}</p>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>📸</span><span>AI 眼（視覺）— 未檢測</span>
          </div>
        )}
      </div>

      {/* Weight Legend */}
      <div className="bg-gray-50 rounded-xl p-3">
        <p className="text-xs text-gray-400 text-center">
          融合算法：聲學 ×0.6 + 視覺 ×0.4 | 加權分數 = {(result.weightedScore * 100).toFixed(0)}%
        </p>
      </div>

      {/* Model Status */}
      {modelStatus && (
        <ModelStatusBadge status={modelStatus} variant="card" />
      )}

      {/* ── User Feedback Section ── */}
      {feedbackState === 'idle' && (
        <div className="bg-blue-50 border border-blue-100 rounded-2xl p-4">
          <div className="flex items-start gap-3">
            <span className="text-xl">🎯</span>
            <div className="flex-1">
              <p className="text-sm font-semibold text-blue-800">幫助改善 AI 準確度</p>
              <p className="text-xs text-blue-600 mt-0.5">
                開果後確認實際成熟度，讓 AI 從真實數據中學習。
              </p>
            </div>
          </div>
          <button
            onClick={handleFeedbackStart}
            className="mt-3 w-full py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium active:scale-95 transition-transform"
          >
            ✍️ 確認實際成熟度
          </button>
        </div>
      )}

      {feedbackState === 'selecting' && (
        <div className="bg-blue-50 border border-blue-200 rounded-2xl p-4">
          <p className="text-sm font-semibold text-blue-800 mb-3">開果後，實際成熟度是？</p>
          <div className="grid grid-cols-3 gap-2 mb-3">
            {RIPENESS_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => setSelectedLabel(opt.value)}
                className={`py-3 rounded-xl text-sm font-medium transition-all border-2 ${
                  selectedLabel === opt.value
                    ? 'border-blue-500 bg-blue-100 text-blue-800 scale-105'
                    : 'border-gray-200 bg-white text-gray-600'
                }`}
              >
                <div className="text-xl mb-1">{opt.emoji}</div>
                <div>{opt.label}</div>
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setFeedbackState('idle')}
              className="flex-1 py-2.5 border border-gray-200 rounded-xl text-sm text-gray-500 active:scale-95"
            >
              取消
            </button>
            <button
              onClick={handleFeedbackSubmit}
              disabled={!selectedLabel}
              className="flex-1 py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium active:scale-95 disabled:opacity-50"
            >
              提交反饋
            </button>
          </div>
        </div>
      )}

      {feedbackState === 'submitting' && (
        <div className="bg-blue-50 border border-blue-100 rounded-2xl p-4 text-center">
          <div className="text-2xl animate-spin inline-block mb-2">⏳</div>
          <p className="text-sm text-blue-700">正在提交反饋...</p>
        </div>
      )}

      {feedbackState === 'done' && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-2xl p-4">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🙏</span>
            <div>
              <p className="text-sm font-semibold text-emerald-800">感謝您的反饋！</p>
              <p className="text-xs text-emerald-600 mt-0.5">
                {selectedLabel === result.ripeness
                  ? '✅ AI 判定正確！繼續積累數據。'
                  : `📝 已記錄：AI 判 ${RIPENESS_LABELS[result.ripeness]}，實際 ${RIPENESS_LABELS[selectedLabel as RipenessLevel] ?? selectedLabel}。`
                }
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Reset Button */}
      <button
        onClick={onReset}
        className="w-full py-4 bg-durian-green text-white rounded-2xl font-semibold text-base active:scale-95 transition-transform"
      >
        🔄 重新檢測
      </button>
    </div>
  );
};
