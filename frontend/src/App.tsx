import { useState, useCallback } from 'react';
import { ProgressStepper } from './components/ProgressStepper';
import { CameraCapture } from './components/CameraCapture';
import { AudioRecorder } from './components/AudioRecorder';
import { ResultCard } from './components/ResultCard';
import { ModelStatusBadge } from './components/ModelStatusBadge';
import type { AppStep, ModalityResult, FusionResult } from './types';
import { computeFusion } from './utils/fusion';
import { useModelStatus } from './hooks/useModelStatus';

function App() {
  const [step, setStep] = useState<AppStep>('intro');
  const [visionResult, setVisionResult] = useState<ModalityResult | null>(null);
  const [_acousticResult, setAcousticResult] = useState<ModalityResult | null>(null);
  const [fusionResult, setFusionResult] = useState<FusionResult | null>(null);

  const { status: modelStatus } = useModelStatus();

  const handleVisionComplete = useCallback((result: ModalityResult) => {
    setVisionResult(result);
    setStep('acoustic');
  }, []);

  const handleVisionSkip = useCallback(() => {
    setVisionResult(null);
    setStep('acoustic');
  }, []);

  const handleAcousticComplete = useCallback((result: ModalityResult) => {
    setAcousticResult(result);
    const fusion = computeFusion(visionResult, result);
    setFusionResult(fusion);
    setStep('result');
  }, [visionResult]);

  const handleAcousticSkip = useCallback(() => {
    if (!visionResult) return;
    const fusion = computeFusion(visionResult, null);
    setFusionResult(fusion);
    setStep('result');
  }, [visionResult]);

  const handleReset = useCallback(() => {
    setStep('intro');
    setVisionResult(null);
    setAcousticResult(null);
    setFusionResult(null);
  }, []);

  return (
    <div className="min-h-screen bg-durian-bg">
      {/* Header */}
      <header className="sticky top-0 z-10 bg-durian-green shadow-md">
        <div className="max-w-md mx-auto px-4 py-3 flex items-center gap-3">
          <span className="text-2xl">🏮</span>
          <div className="flex-1 min-w-0">
            <h1 className="text-white font-bold text-lg leading-tight">DurianAI</h1>
            <p className="text-emerald-200 text-xs">榴槤成熟度智能檢測</p>
          </div>
          {/* Model status badges */}
          <ModelStatusBadge status={modelStatus} variant="header" />
        </div>
      </header>

      {/* Progress Stepper */}
      {step !== 'intro' && (
        <div className="max-w-md mx-auto bg-white shadow-sm">
          <ProgressStepper currentStep={step} />
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-md mx-auto px-4 py-5">

        {/* ── Intro Screen ── */}
        {step === 'intro' && (
          <div className="flex flex-col items-center gap-6">
            <div className="text-center pt-4">
              <div className="text-8xl mb-4">🏮</div>
              <h2 className="text-2xl font-bold text-gray-800 mb-2">榴槤成熟度檢測</h2>
              <p className="text-gray-500 text-sm leading-relaxed">
                利用手機 AI 雙模態分析，透過視覺和聲學兩種方式判斷榴槤是否達到最佳食用成熟度
              </p>
            </div>

            {/* Feature cards */}
            <div className="w-full grid grid-cols-2 gap-3">
              {[
                { icon: '📸', title: 'AI 眼', desc: '拍攝果殼分析顏色紋理', color: 'bg-amber-50 border-amber-100' },
                { icon: '🔊', title: 'AI 耳', desc: '敲擊聲學分析內部狀態', color: 'bg-emerald-50 border-emerald-100' },
                { icon: '🧠', title: 'AI 融合', desc: '雙模態加權判定結果', color: 'bg-blue-50 border-blue-100' },
                { icon: '⚡', title: '即時結果', desc: '5秒內完成全面分析', color: 'bg-purple-50 border-purple-100' },
              ].map(({ icon, title, desc, color }) => (
                <div key={title} className={`border rounded-2xl p-3 ${color}`}>
                  <div className="text-2xl mb-1">{icon}</div>
                  <div className="font-semibold text-gray-800 text-sm">{title}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{desc}</div>
                </div>
              ))}
            </div>

            {/* Model Status Card */}
            <div className="w-full">
              <ModelStatusBadge status={modelStatus} variant="card" />
            </div>

            {/* Accuracy note */}
            <div className="w-full bg-white border border-gray-100 rounded-2xl p-4 shadow-sm">
              <div className="flex items-start gap-3">
                <span className="text-xl">📊</span>
                <div>
                  <p className="text-sm font-semibold text-gray-700">基於學術研究</p>
                  <p className="text-xs text-gray-500 mt-1">
                    目標：聲學 KnockNet（96.34%）+ 視覺 CNN（95.5%），
                    雙模態融合目標準確率 94-97%。
                    訓練模型部署後自動啟用。
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={() => setStep('vision')}
              className="w-full py-4 bg-durian-green text-white rounded-2xl font-bold text-lg active:scale-95 transition-transform shadow-lg"
            >
              開始檢測 🚀
            </button>

            <p className="text-xs text-gray-400 text-center">
              需要攝像頭和麥克風權限 · 數據僅在確認後才上傳
            </p>
          </div>
        )}

        {/* ── Vision Step ── */}
        {step === 'vision' && (
          <CameraCapture
            onComplete={handleVisionComplete}
            onSkip={handleVisionSkip}
          />
        )}

        {/* ── Acoustic Step ── */}
        {step === 'acoustic' && (
          <AudioRecorder
            onComplete={handleAcousticComplete}
            onSkip={handleAcousticSkip}
          />
        )}

        {/* ── Result Step ── */}
        {step === 'result' && fusionResult && (
          <ResultCard
            result={fusionResult}
            onReset={handleReset}
            modelStatus={modelStatus}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="max-w-md mx-auto px-4 pb-8 pt-2 text-center">
        <p className="text-xs text-gray-300">
          DurianAI · 僅供零售篩選參考 · 最終判斷請結合實際開果
        </p>
      </footer>
    </div>
  );
}

export default App;
