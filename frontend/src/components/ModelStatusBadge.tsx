/**
 * ModelStatusBadge.tsx
 * Compact badge showing current AI model status (AI / Heuristic).
 * Used in App header and result card.
 */
import React from 'react';
import type { SystemModelStatus } from '../hooks/useModelStatus';

interface Props {
  status: SystemModelStatus;
  variant?: 'header' | 'inline' | 'card';
}

export const ModelStatusBadge: React.FC<Props> = ({ status, variant = 'inline' }) => {
  const { checking } = status;

  if (checking && !status.lastChecked) {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-gray-400 px-2 py-0.5 rounded-full bg-gray-100">
        <span className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-pulse" />
        檢測中...
      </span>
    );
  }

  const visionAI = status.visionModelOnFrontend;
  const acousticAI = status.acousticModelOnBackend;
  const anyAI = visionAI || acousticAI;

  if (variant === 'header') {
    return (
      <div className="flex items-center gap-1.5">
        {/* Vision badge */}
        <span
          className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium ${
            visionAI
              ? 'bg-emerald-100 text-emerald-700'
              : 'bg-amber-50 text-amber-600'
          }`}
          title={visionAI ? '視覺 AI 模型已載入' : '視覺：使用色彩分析'}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${visionAI ? 'bg-emerald-500' : 'bg-amber-400'}`} />
          📸 {visionAI ? 'AI' : '啟發式'}
        </span>

        {/* Acoustic badge */}
        <span
          className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium ${
            acousticAI
              ? 'bg-emerald-100 text-emerald-700'
              : status.backendOnline
                ? 'bg-amber-50 text-amber-600'
                : 'bg-gray-100 text-gray-400'
          }`}
          title={
            acousticAI ? '聲學 AI 模型已載入'
            : status.backendOnline ? '聲學：使用頻譜分析'
            : '後端離線'
          }
        >
          <span className={`w-1.5 h-1.5 rounded-full ${
            acousticAI ? 'bg-emerald-500'
            : status.backendOnline ? 'bg-amber-400'
            : 'bg-gray-300'
          }`} />
          🔊 {acousticAI ? 'AI' : status.backendOnline ? '啟發式' : '離線'}
        </span>
      </div>
    );
  }

  if (variant === 'card') {
    return (
      <div className={`rounded-xl px-3 py-2 text-xs ${anyAI ? 'bg-emerald-50 border border-emerald-100' : 'bg-amber-50 border border-amber-100'}`}>
        <div className="flex items-center justify-between mb-1.5">
          <span className={`font-semibold ${anyAI ? 'text-emerald-700' : 'text-amber-700'}`}>
            {anyAI ? '🤖 AI 模型推理中' : '📐 啟發式規則分析'}
          </span>
          {status.feedbackCount > 0 && (
            <span className="text-gray-400">已收集 {status.feedbackCount} 條反饋</span>
          )}
        </div>
        <div className="flex gap-3 text-gray-500">
          <span className={`flex items-center gap-1 ${visionAI ? 'text-emerald-600' : ''}`}>
            📸 視覺：{visionAI ? 'CNN AI' : '色彩分析'}
          </span>
          <span className={`flex items-center gap-1 ${acousticAI ? 'text-emerald-600' : ''}`}>
            🔊 聲學：{
              acousticAI ? 'KnockNet' :
              status.backendOnline ? '頻譜分析' :
              '後端離線'
            }
          </span>
        </div>
        {!anyAI && (
          <p className="text-gray-400 mt-1">
            訓練 AI 模型後，部署即可啟用真正 AI 推理。
          </p>
        )}
      </div>
    );
  }

  // variant === 'inline'
  return (
    <span
      className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
        anyAI ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-50 text-amber-600'
      }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${anyAI ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400'}`} />
      {anyAI ? 'AI 模型' : '啟發式'}
    </span>
  );
};
