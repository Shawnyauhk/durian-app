import { useState, useRef, useCallback } from 'react';

export type RecordingState = 'idle' | 'requesting' | 'recording' | 'processing' | 'done' | 'error';

export function useAudioRecorder() {
  const [state, setState] = useState<RecordingState>('idle');
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const durationRef = useRef(0);

  const startRecording = useCallback(async (maxDurationMs = 5000) => {
    setError(null);
    setAudioBlob(null);
    setDuration(0);
    durationRef.current = 0;
    chunksRef.current = [];

    try {
      setState('requesting');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false, // keep raw signal for knock detection
        },
        video: false,
      });
      streamRef.current = stream;

      // Try preferred MIME types
      const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4'];
      const mimeType = mimeTypes.find(m => MediaRecorder.isTypeSupported(m)) || '';

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' });
        setAudioBlob(blob);
        setState('done');
        // Stop all tracks
        streamRef.current?.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      };

      recorder.onerror = () => {
        setError('錄音發生錯誤，請重試');
        setState('error');
      };

      setState('recording');
      recorder.start(100); // collect data every 100ms

      // Auto-stop after maxDuration
      timerRef.current = setInterval(() => {
        durationRef.current += 100;
        setDuration(durationRef.current);
        if (durationRef.current >= maxDurationMs) {
          stopRecording();
        }
      }, 100);

    } catch (err) {
      const msg = err instanceof Error ? err.message : '無法訪問麥克風';
      const friendly = msg.includes('denied') || msg.includes('NotAllowed')
        ? '請允許訪問麥克風後重試'
        : '無法訪問麥克風，請檢查設備權限';
      setError(friendly);
      setState('error');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      setState('processing');
      mediaRecorderRef.current.stop();
    }
  }, []);

  const reset = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    mediaRecorderRef.current = null;
    chunksRef.current = [];
    setState('idle');
    setAudioBlob(null);
    setDuration(0);
    setError(null);
    durationRef.current = 0;
  }, []);

  return {
    state,
    audioBlob,
    duration,
    error,
    startRecording,
    stopRecording,
    reset,
  };
}
