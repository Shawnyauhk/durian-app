import React, { useRef, useState, useCallback, useEffect } from 'react';
import { useVisionModel } from '../hooks/useVisionModel';
import { ConfidenceBar } from './ConfidenceBar';
import type { ModalityResult } from '../types';

interface Props {
  onComplete: (result: ModalityResult) => void;
  onSkip: () => void;
}

type CameraState = 'idle' | 'preview' | 'captured' | 'analyzing' | 'done';

export const CameraCapture: React.FC<Props> = ({ onComplete, onSkip }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cameraState, setCameraState] = useState<CameraState>('idle');
  const [capturedDataUrl, setCapturedDataUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ModalityResult | null>(null);
  const [camError, setCamError] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');

  const { classify } = useVisionModel();

  const startCamera = useCallback(async (facing: 'environment' | 'user' = 'environment') => {
    setCamError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facing,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState('preview');
    } catch (err) {
      const msg = err instanceof Error ? err.message : '無法開啟攝像頭';
      setCamError(msg.includes('denied') ? '請允許訪問攝像頭' : '無法開啟攝像頭');
      setCameraState('idle');
    }
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  const capturePhoto = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    setCapturedDataUrl(dataUrl);
    stopCamera();
    setCameraState('captured');
  }, [stopCamera]);

  const analyzeCapture = useCallback(async () => {
    if (!canvasRef.current) return;
    setCameraState('analyzing');
    const res = await classify(canvasRef.current);
    setResult(res);
    setCameraState('done');
  }, [classify]);

  const retake = useCallback(() => {
    setCapturedDataUrl(null);
    setResult(null);
    setCameraState('idle');
  }, []);

  const switchCamera = useCallback(() => {
    const newFacing = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(newFacing);
    stopCamera();
    startCamera(newFacing);
  }, [facingMode, stopCamera, startCamera]);

  return (
    <div className="flex flex-col gap-4">
      {/* Instructions */}
      <div className="bg-amber-50 border border-amber-100 rounded-2xl p-4">
        <h3 className="font-semibold text-amber-800 mb-1">📸 AI 眼 — 視覺分析</h3>
        <p className="text-sm text-amber-700">
          將榴槤整個果實拍入鏡頭，確保光線充足。系統將分析果殼顏色、棘刺形態和外觀特徵。
        </p>
      </div>

      {/* Camera / Preview Area */}
      <div className="relative bg-black rounded-2xl overflow-hidden aspect-[4/3]">
        {cameraState === 'idle' && !camError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-white">
            <div className="text-5xl">📷</div>
            <p className="text-sm text-gray-300">點擊開啟攝像頭</p>
          </div>
        )}

        {camError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-4">
            <div className="text-3xl">⚠️</div>
            <p className="text-sm text-white text-center">{camError}</p>
          </div>
        )}

        {/* Live preview */}
        <video
          ref={videoRef}
          className={`w-full h-full object-cover ${cameraState === 'preview' ? '' : 'hidden'}`}
          playsInline
          muted
        />

        {/* Captured image */}
        {capturedDataUrl && cameraState !== 'idle' && (
          <img
            src={capturedDataUrl}
            alt="Captured durian"
            className="w-full h-full object-cover"
          />
        )}

        {/* Analyzing overlay */}
        {cameraState === 'analyzing' && (
          <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center gap-2 text-white">
            <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
            <p className="text-sm">AI 視覺分析中...</p>
          </div>
        )}

        {/* Camera guide overlay */}
        {cameraState === 'preview' && (
          <>
            <div className="absolute inset-4 border-2 border-white/40 rounded-xl pointer-events-none" />
            <button
              onClick={switchCamera}
              className="absolute top-3 right-3 bg-black/40 text-white p-2 rounded-full text-sm"
            >
              🔄
            </button>
          </>
        )}
      </div>

      {/* Hidden canvas for image processing */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Action Buttons */}
      <div className="space-y-2">
        {cameraState === 'idle' && (
          <button
            onClick={() => startCamera(facingMode)}
            className="w-full py-4 bg-durian-green text-white rounded-2xl font-semibold text-base active:scale-95 transition-transform"
          >
            📷 開啟攝像頭
          </button>
        )}

        {cameraState === 'preview' && (
          <button
            onClick={capturePhoto}
            className="w-full py-4 bg-durian-green text-white rounded-2xl font-semibold text-base active:scale-95 transition-transform"
          >
            📸 拍攝照片
          </button>
        )}

        {cameraState === 'captured' && (
          <div className="flex gap-2">
            <button
              onClick={retake}
              className="flex-1 py-4 bg-gray-100 text-gray-700 rounded-2xl font-semibold active:scale-95 transition-transform"
            >
              重拍
            </button>
            <button
              onClick={analyzeCapture}
              className="flex-2 flex-grow-[2] py-4 bg-durian-green text-white rounded-2xl font-semibold active:scale-95 transition-transform"
            >
              🔍 分析
            </button>
          </div>
        )}

        {cameraState === 'done' && result && (
          <div className="space-y-3">
            <div className="bg-white border border-gray-100 rounded-2xl p-4 shadow-sm">
              <ConfidenceBar scores={result.scores} label="視覺分析結果" icon="📸" />
            </div>
            <button
              onClick={() => onComplete(result)}
              className="w-full py-4 bg-durian-green text-white rounded-2xl font-semibold active:scale-95 transition-transform"
            >
              下一步：AI 耳 聲學檢測 →
            </button>
          </div>
        )}
      </div>

      {/* Skip button */}
      {cameraState !== 'analyzing' && cameraState !== 'done' && (
        <button
          onClick={onSkip}
          className="w-full py-2 text-sm text-gray-400 active:text-gray-600"
        >
          跳過視覺檢測
        </button>
      )}
    </div>
  );
};
