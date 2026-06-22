"""
audio_processor.py — 榴槤敲擊音分析
支持兩種模式：
1. TFLite 模型推理（當模型可用時）
2. 頻譜特徵啟發式（後備方案）
"""
import io
import os
import logging
import numpy as np
import librosa
import soundfile as sf

# pydub for webm/opus decoding (browser-recorded audio)
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

logger = logging.getLogger(__name__)

# TFLite runtime (optional — falls back to heuristic if not installed)
try:
    import tflite_runtime.interpreter as tflite
    HAS_TFLITE = True
except ImportError:
    try:
        import tensorflow as tf
        HAS_TFLITE = True
        tflite = tf.lite
    except ImportError:
        HAS_TFLITE = False

TARGET_SR = 16000  # 16kHz, same as KnockNet paper
N_MFCC = 13
HOP_LENGTH = 256
N_FFT = 512
N_MELS = 40
SEGMENT_DURATION = 2.0  # seconds

# Model paths
ACOUSTIC_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "acoustic", "knocknet_lite.tflite"
)
ACOUSTIC_LABELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "acoustic", "labels.txt"
)

# Global model interpreter (lazy-loaded)
_interpreter = None
_input_details = None
_output_details = None
_model_labels = ["unripe", "ripe", "overripe"]


def _load_model():
    """Lazy-load TFLite acoustic model."""
    global _interpreter, _input_details, _output_details, _model_labels

    if _interpreter is not None:
        return _interpreter is not False  # False = failed to load

    if not HAS_TFLITE:
        print("[AudioProcessor] TFLite not available, using heuristic")
        _interpreter = False
        return False

    model_path = os.environ.get("ACOUSTIC_MODEL_PATH", ACOUSTIC_MODEL_PATH)

    if not os.path.exists(model_path):
        print(f"[AudioProcessor] Model not found at {model_path}, using heuristic")
        _interpreter = False
        return False

    try:
        if hasattr(tflite, 'Interpreter'):
            _interpreter = tflite.Interpreter(model_path=model_path)
        else:
            # tensorflow.lite.Interpreter
            _interpreter = tflite.Interpreter(model_path=model_path)

        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()

        # Load labels
        labels_path = os.environ.get("ACOUSTIC_LABELS_PATH", ACOUSTIC_LABELS_PATH)
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                _model_labels = [line.strip() for line in f if line.strip()]

        print(f"[AudioProcessor] TFLite model loaded: {model_path}")
        print(f"[AudioProcessor] Input shape: {_input_details[0]['shape']}")
        print(f"[AudioProcessor] Labels: {_model_labels}")
        return True

    except Exception as e:
        print(f"[AudioProcessor] Failed to load model: {e}")
        _interpreter = False
        return False


def get_model_info() -> dict:
    """Return current model status information."""
    model_path = os.environ.get("ACOUSTIC_MODEL_PATH", ACOUSTIC_MODEL_PATH)
    loaded = _interpreter is not None and _interpreter is not False

    # Try to read version from metadata file
    version = None
    meta_path = os.path.splitext(model_path)[0] + "_metadata.json"
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            version = meta.get("version") or meta.get("created_at")
        except Exception:
            pass

    return {
        "loaded": loaded,
        "method": "tflite_model" if loaded else "spectral_heuristic",
        "path": model_path if loaded else None,
        "labels": _model_labels,
        "tflite_available": HAS_TFLITE,
        "model_exists": os.path.exists(model_path),
        "version": version,
        "input_shape": (
            _input_details[0]['shape'].tolist()
            if loaded and _input_details else None
        ),
    }


def reload_model() -> bool:
    """Hot-reload the acoustic model (resets global state and re-loads)."""
    global _interpreter, _input_details, _output_details, _model_labels
    # Reset state
    _interpreter = None
    _input_details = None
    _output_details = None
    _model_labels = ["unripe", "ripe", "overripe"]
    # Re-load
    return _load_model()


def load_audio(audio_bytes: bytes) -> np.ndarray:
    """Load audio from bytes, convert to mono 16kHz.
    
    Handles browser-recorded webm/opus via pydub+ffmpeg,
    falls back to soundfile then librosa for WAV/other formats.
    """
    y = None
    
    # 1. Try pydub (handles webm/opus — browser-recorded format)
    if HAS_PYDUB:
        try:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            seg = seg.set_frame_rate(TARGET_SR).set_channels(1)
            wav_buf = io.BytesIO()
            seg.export(wav_buf, format='wav')
            wav_buf.seek(0)
            y, sr = sf.read(wav_buf, always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            logger.info(f"Audio loaded via pydub: {len(y)} samples at {TARGET_SR}Hz")
        except Exception as e:
            logger.warning(f"pydub conversion failed ({e}), trying soundfile directly")
    
    # 2. Try soundfile (handles WAV/FLAC/OGG)
    if y is None:
        try:
            buf = io.BytesIO(audio_bytes)
            y, sr = sf.read(buf, always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        except Exception:
            pass
    
    # 3. Try librosa+ffmpeg (last resort)
    if y is None:
        try:
            buf = io.BytesIO(audio_bytes)
            y, sr = librosa.load(buf, sr=TARGET_SR, mono=True)
        except Exception as e:
            raise ValueError(f"Cannot decode audio: {e}")
    
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    return y.astype(np.float32)


def extract_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """Extract mel spectrogram for TFLite model input."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """Extract MFCC features."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=TARGET_SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mfcc_norm = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
    return mfcc_norm.astype(np.float32)


def _run_model_inference(features: np.ndarray) -> dict | None:
    """Run TFLite model inference."""
    if not _load_model():
        return None

    try:
        # Prepare input based on model's expected shape
        input_shape = _input_details[0]['shape']
        input_dtype = _input_details[0]['dtype']

        # Reshape features to match model input
        if len(input_shape) == 4:
            # CNN expects (1, H, W, C)
            if features.ndim == 2:
                features = features[..., np.newaxis]  # Add channel dim
            features = features[np.newaxis, ...]  # Add batch dim
        elif len(input_shape) == 2:
            # Flatten
            features = features.flatten()[np.newaxis, ...]

        # Cast to expected dtype
        if input_dtype == np.float32:
            features = features.astype(np.float32)
        elif input_dtype == np.int8:
            features = features.astype(np.int8)

        # Set input and run
        _interpreter.set_tensor(_input_details[0]['index'], features)
        _interpreter.invoke()

        # Get output
        output_data = _interpreter.get_tensor(_output_details[0]['index'])
        scores = output_data[0]  # Remove batch dim

        # Map scores to labels
        result_scores = {}
        for i, label in enumerate(_model_labels):
            if i < len(scores):
                result_scores[label] = float(scores[i])

        # Normalize to standard 3-class
        unripe = result_scores.get('unripe', 0)
        ripe = result_scores.get('ripe', 0)
        overripe = result_scores.get('overripe', 0)

        total = unripe + ripe + overripe
        if total > 0:
            unripe /= total
            ripe /= total
            overripe /= total

        scores_dict = {'unripe': float(unripe), 'ripe': float(ripe), 'overripe': float(overripe)}
        ripeness = max(scores_dict, key=lambda k: scores_dict[k])

        sorted_vals = sorted(scores_dict.values(), reverse=True)
        confidence = min(0.98, sorted_vals[0])

        return {
            'ripeness': ripeness,
            'scores': scores_dict,
            'confidence': float(confidence),
            'method': 'tflite_model',
        }

    except Exception as e:
        print(f"[AudioProcessor] Model inference failed: {e}")
        return None


# ============================================================
# Heuristic Fallback (spectral features)
# ============================================================

def extract_spectral_features(y: np.ndarray) -> dict:
    """Extract spectral features for rule-based inference."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    return {
        'centroid_mean': float(centroid.mean()),
        'centroid_std': float(centroid.std()),
        'rolloff_mean': float(rolloff.mean()),
        'flatness_mean': float(flatness.mean()),
        'zcr_mean': float(zcr.mean()),
        'rms_mean': float(rms.mean()),
        'rms_std': float(rms.std()),
    }


def classify_by_features(features: dict) -> dict:
    """Rule-based classifier based on spectral features."""
    centroid = features['centroid_mean']
    zcr = features['zcr_mean']
    flatness = features['flatness_mean']

    centroid_norm = np.clip((centroid - 500) / 3500, 0, 1)
    zcr_norm = np.clip(zcr / 0.2, 0, 1)
    flatness_norm = np.clip(flatness / 0.5, 0, 1)

    unripe_signal = centroid_norm * 0.5 + zcr_norm * 0.3 + (1 - flatness_norm) * 0.2
    ripe_signal = (1 - abs(centroid_norm - 0.35)) * 0.5 + (1 - abs(zcr_norm - 0.3)) * 0.3 + 0.2
    overripe_signal = (1 - centroid_norm) * 0.4 + flatness_norm * 0.4 + (1 - zcr_norm) * 0.2

    total = unripe_signal + ripe_signal + overripe_signal
    if total > 0:
        unripe_score = unripe_signal / total
        ripe_score = ripe_signal / total
        overripe_score = overripe_signal / total
    else:
        unripe_score = ripe_score = overripe_score = 1/3

    scores = {'unripe': float(unripe_score), 'ripe': float(ripe_score), 'overripe': float(overripe_score)}
    ripeness = max(scores, key=lambda k: scores[k])

    sorted_vals = sorted(scores.values(), reverse=True)
    confidence = min(0.85, sorted_vals[0] - sorted_vals[1] + 0.45)

    return {
        'ripeness': ripeness,
        'scores': scores,
        'confidence': float(confidence),
        'method': 'spectral_heuristic',
    }


# ============================================================
# Main Analysis Pipeline
# ============================================================

def analyze_audio(audio_bytes: bytes) -> dict:
    """
    Main analysis pipeline.
    1. Try TFLite model inference (if model available)
    2. Fall back to spectral heuristic
    """
    y = load_audio(audio_bytes)

    # Try TFLite model first
    mel_spec = extract_mel_spectrogram(y)
    model_result = _run_model_inference(mel_spec)

    if model_result:
        model_result['features'] = extract_spectral_features(y)  # include for debugging
        return model_result

    # Fallback: heuristic
    features = extract_spectral_features(y)
    result = classify_by_features(features)
    result['features'] = features

    return result
