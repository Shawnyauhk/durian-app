/**
 * tf-helpers.ts — DurianAI Vision Model Inference
 *
 * Loading priority:
 *   1. TFJS SavedModel JSON format  (/models/vision/model.json)  — preferred
 *   2. Heuristic color analysis                                   — fallback
 *
 * After training, export model from Colab with:
 *   model.save('tfjs_model/')  → upload to frontend/public/models/vision/
 *
 * Model architecture: MobileNetV2 fine-tuned → 3 classes [unripe, ripe, overripe]
 * Input: [1, 224, 224, 3] float32, normalized to [0,1]
 * Output: [1, 3] softmax probabilities
 */
import * as tf from '@tensorflow/tfjs';
import type { ModalityResult } from '../types';

// ============================================================
// Configuration
// ============================================================

const VISION_MODEL_JSON = '/models/vision/model.json';
const VISION_LABELS_PATH = '/models/vision/labels.txt';
const ACOUSTIC_MODEL_JSON = '/models/acoustic/model.json';
const ACOUSTIC_LABELS_PATH = '/models/acoustic/labels.txt';

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;

export type ModelFormat = 'tfjs' | 'none';

export interface ModelStatus {
  loaded: boolean;
  format: ModelFormat;
  labels: string[];
  inputShape: number[] | null;
  version: string | null;
  error: string | null;
}

// ============================================================
// Vision Model State
// ============================================================

let _visionModel: tf.GraphModel | tf.LayersModel | null = null;
let _visionLabels: string[] = ['unripe', 'ripe', 'overripe'];
let _visionLoading = false;
let _visionLoadAttempted = false;
let _visionVersion: string | null = null;

export function getVisionModelStatus(): ModelStatus {
  return {
    loaded: _visionModel !== null,
    format: _visionModel ? 'tfjs' : 'none',
    labels: _visionLabels,
    inputShape: _visionModel ? [1, INPUT_HEIGHT, INPUT_WIDTH, 3] : null,
    version: _visionVersion,
    error: null,
  };
}

export function isVisionModelReady(): boolean {
  return _visionModel !== null;
}

export function getVisionModelType(): 'ai' | 'heuristic' {
  return _visionModel ? 'ai' : 'heuristic';
}

/**
 * Load the TFJS vision model from /public/models/vision/model.json
 * Returns true if model loaded successfully, false if falling back to heuristic.
 */
export async function loadVisionModel(): Promise<boolean> {
  if (_visionModel) return true;
  if (_visionLoading) return false;
  if (_visionLoadAttempted) return false; // Don't retry after failure

  _visionLoading = true;
  _visionLoadAttempted = true;

  try {
    // Check if model file exists first (avoid long timeout errors)
    const modelUrl = `${window.location.origin}${VISION_MODEL_JSON}`;
    const checkResp = await fetch(modelUrl, { method: 'HEAD' });
    if (!checkResp.ok) {
      console.info('[VisionModel] Model not found at', VISION_MODEL_JSON, '— using heuristic');
      return false;
    }

    console.info('[VisionModel] Loading TF.js model from', VISION_MODEL_JSON);

    // Try GraphModel first (from model.save() in TF.js format)
    try {
      _visionModel = await tf.loadGraphModel(modelUrl);
      console.info('[VisionModel] GraphModel loaded ✓');
    } catch {
      // Try LayersModel (from model.save() Keras format)
      _visionModel = await tf.loadLayersModel(modelUrl);
      console.info('[VisionModel] LayersModel loaded ✓');
    }

    // Load labels
    await _loadVisionLabels();

    // Read version from metadata if available
    try {
      const metaUrl = `${window.location.origin}/models/vision/metadata.json`;
      const metaResp = await fetch(metaUrl);
      if (metaResp.ok) {
        const meta = await metaResp.json();
        _visionVersion = meta.version ?? meta.created_at ?? null;
      }
    } catch {
      // Metadata optional
    }

    // Warm up the model with a dummy input
    try {
      const dummy = tf.zeros([1, INPUT_HEIGHT, INPUT_WIDTH, 3]);
      const warmup = _visionModel.predict(dummy) as tf.Tensor;
      warmup.dispose();
      dummy.dispose();
      console.info('[VisionModel] Warm-up complete');
    } catch (e) {
      console.warn('[VisionModel] Warm-up failed (non-critical):', e);
    }

    return true;

  } catch (err) {
    console.warn('[VisionModel] Failed to load model, using heuristic:', err);
    _visionModel = null;
    return false;
  } finally {
    _visionLoading = false;
  }
}

/** Reset model state — allows retry after model file is deployed. */
export function resetVisionModel(): void {
  _visionModel = null;
  _visionLoadAttempted = false;
  _visionLoading = false;
}

async function _loadVisionLabels(): Promise<void> {
  try {
    const labelsUrl = `${window.location.origin}${VISION_LABELS_PATH}`;
    const resp = await fetch(labelsUrl);
    if (resp.ok) {
      const text = await resp.text();
      const parsed = text.trim().split('\n').map(l => l.trim()).filter(Boolean);
      if (parsed.length >= 2) {
        _visionLabels = parsed;
        console.info('[VisionModel] Labels:', _visionLabels);
      }
    }
  } catch {
    console.warn('[VisionModel] Using default labels');
  }
}

// ============================================================
// Vision Model Inference
// ============================================================

/**
 * Run model inference on a canvas element.
 * Returns normalized 3-class probability scores.
 */
async function _runVisionInference(canvas: HTMLCanvasElement): Promise<{
  scores: { unripe: number; ripe: number; overripe: number };
} | null> {
  if (!_visionModel) return null;

  return tf.tidy(() => {
    // Preprocess: crop center, resize to 224×224, normalize to [0,1]
    const raw = tf.browser.fromPixels(canvas);

    // Center crop to square
    const h = raw.shape[0];
    const w = raw.shape[1];
    const side = Math.min(h, w);
    const y0 = Math.floor((h - side) / 2);
    const x0 = Math.floor((w - side) / 2);
    const cropped = tf.slice(raw, [y0, x0, 0], [side, side, 3]);

    // Resize to 224×224
    const resized = tf.image.resizeBilinear(cropped as tf.Tensor3D, [INPUT_HEIGHT, INPUT_WIDTH]);

    // Normalize to [0, 1]
    const normalized = resized.toFloat().div(255.0);
    const batched = normalized.expandDims(0); // [1, 224, 224, 3]

    // Inference
    const output = _visionModel!.predict(batched) as tf.Tensor;
    const probsArray = Array.from(output.dataSync());

    // Map to standard 3-class output
    const rawScores: Record<string, number> = {};
    _visionLabels.forEach((label, i) => {
      if (i < probsArray.length) rawScores[label.toLowerCase()] = probsArray[i];
    });

    // Resolve with label aliases
    const unripe = rawScores['unripe'] ?? rawScores['ripe1'] ?? rawScores['immature'] ?? 0;
    const ripe   = rawScores['ripe']   ?? rawScores['ripe2'] ?? rawScores['mature']    ?? 0;
    const over   = rawScores['overripe'] ?? rawScores['ripe3'] ?? rawScores['ripe4'] ?? rawScores['overmature'] ?? 0;

    // Normalize to sum = 1
    const total = unripe + ripe + over;
    if (total > 0) {
      return { scores: { unripe: unripe / total, ripe: ripe / total, overripe: over / total } };
    }
    return { scores: { unripe: 0.33, ripe: 0.34, overripe: 0.33 } };
  });
}

// ============================================================
// Heuristic Color Fallback
// ============================================================

interface ColorFeatures {
  greenRatio: number;
  yellowRatio: number;
  brownRatio: number;
  darkRatio: number;
  saturationMean: number;
}

function _extractColorFeatures(canvas: HTMLCanvasElement): ColorFeatures {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Cannot get canvas context');

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  let greenCount = 0, yellowCount = 0, brownCount = 0, darkCount = 0;
  let totalSat = 0;
  const pixelCount = data.length / 4;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
    const brightness = (r + g + b) / 3;

    // HSV saturation
    const maxC = Math.max(r, g, b);
    const minC = Math.min(r, g, b);
    const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
    totalSat += sat;

    if (brightness < 0.23) { darkCount++; }
    else if (g > r * 1.2 && g > b * 1.2 && sat > 0.2) { greenCount++; }
    else if (r > 0.70 && g > 0.59 && b < 0.39 && sat > 0.15) { yellowCount++; }
    else if (r > 0.47 && g < 0.39 && b < 0.31 && sat > 0.15) { brownCount++; }
  }

  return {
    greenRatio: greenCount / pixelCount,
    yellowRatio: yellowCount / pixelCount,
    brownRatio: brownCount / pixelCount,
    darkRatio: darkCount / pixelCount,
    saturationMean: totalSat / pixelCount,
  };
}

function _heuristicClassify(f: ColorFeatures): {
  scores: { unripe: number; ripe: number; overripe: number };
  confidence: number;
} {
  const { greenRatio, yellowRatio, brownRatio, darkRatio } = f;

  // Weighted rule-based scoring
  let u = greenRatio * 0.65 + (1 - yellowRatio) * 0.20 + (1 - brownRatio) * 0.15;
  let r = yellowRatio * 0.55 + greenRatio * 0.10 + (1 - darkRatio) * 0.20 + 0.15;
  let o = brownRatio * 0.50 + darkRatio * 0.30 + yellowRatio * 0.20;

  const total = u + r + o || 1;
  u /= total; r /= total; o /= total;

  const sorted = [u, r, o].sort((a, b) => b - a);
  const confidence = Math.min(0.82, sorted[0] - sorted[1] + 0.35);

  return { scores: { unripe: u, ripe: r, overripe: o }, confidence };
}

// ============================================================
// Public API: classifyDurianRipeness
// ============================================================

/**
 * Main classification function.
 * Uses TF.js model if loaded, otherwise heuristic color analysis.
 */
export async function classifyDurianRipeness(canvas: HTMLCanvasElement): Promise<ModalityResult> {

  // ── AI Model Path ──
  if (_visionModel) {
    try {
      const result = await _runVisionInference(canvas);
      if (result) {
        const { scores } = result;
        const maxScore = Math.max(scores.unripe, scores.ripe, scores.overripe);
        const ripeness: 'unripe' | 'ripe' | 'overripe' =
          maxScore === scores.unripe ? 'unripe' :
          maxScore === scores.overripe ? 'overripe' : 'ripe';

        const sorted = [scores.unripe, scores.ripe, scores.overripe].sort((a, b) => b - a);
        const confidence = Math.min(0.98, sorted[0]);

        return { ripeness, scores, confidence, available: true };
      }
    } catch (err) {
      console.warn('[VisionModel] Inference error, falling back to heuristic:', err);
    }
  }

  // ── Heuristic Fallback ──
  const features = _extractColorFeatures(canvas);
  const { scores, confidence } = _heuristicClassify(features);

  const maxScore = Math.max(scores.unripe, scores.ripe, scores.overripe);
  const ripeness: 'unripe' | 'ripe' | 'overripe' =
    maxScore === scores.unripe ? 'unripe' :
    maxScore === scores.overripe ? 'overripe' : 'ripe';

  return {
    ripeness,
    scores,
    confidence,
    available: true,
    error: '使用色彩啟發式分析（視覺 AI 模型尚未部署）',
  };
}


// ============================================================
// Acoustic Model (for browser-side inference, optional)
// ============================================================

let _acousticModel: tf.GraphModel | tf.LayersModel | null = null;
let _acousticLabels: string[] = ['unripe', 'ripe', 'overripe'];
let _acousticLoadAttempted = false;

export function getAcousticModelType(): 'ai' | 'heuristic' {
  return _acousticModel ? 'ai' : 'heuristic';
}

export function isAcousticModelReady(): boolean {
  return _acousticModel !== null;
}

export async function loadAcousticModel(): Promise<boolean> {
  if (_acousticModel) return true;
  if (_acousticLoadAttempted) return false;
  _acousticLoadAttempted = true;

  try {
    const modelUrl = `${window.location.origin}${ACOUSTIC_MODEL_JSON}`;
    const checkResp = await fetch(modelUrl, { method: 'HEAD' });
    if (!checkResp.ok) return false;

    try {
      _acousticModel = await tf.loadGraphModel(modelUrl);
    } catch {
      _acousticModel = await tf.loadLayersModel(modelUrl);
    }

    // Load labels
    try {
      const resp = await fetch(`${window.location.origin}${ACOUSTIC_LABELS_PATH}`);
      if (resp.ok) {
        const text = await resp.text();
        const parsed = text.trim().split('\n').map(l => l.trim()).filter(Boolean);
        if (parsed.length >= 2) _acousticLabels = parsed;
      }
    } catch {/* use defaults */}

    console.info('[AcousticModel] Loaded ✓ labels:', _acousticLabels);
    return true;
  } catch (err) {
    console.info('[AcousticModel] Not available, backend will handle inference:', err);
    return false;
  }
}

/**
 * Run acoustic model inference on mel-spectrogram features.
 * Input: Float32Array of flattened mel spectrogram [40 × T]
 */
export async function classifyAcousticFeatures(
  melData: Float32Array,
  melRows: number,
  melCols: number,
): Promise<ModalityResult | null> {
  if (!_acousticModel) return null;

  return tf.tidy(() => {
    const tensor = tf.tensor(melData, [1, melRows, melCols, 1]);
    const output = _acousticModel!.predict(tensor) as tf.Tensor;
    const probsArray = Array.from(output.dataSync());

    const rawScores: Record<string, number> = {};
    _acousticLabels.forEach((label, i) => {
      if (i < probsArray.length) rawScores[label.toLowerCase()] = probsArray[i];
    });

    const unripe = rawScores['unripe'] ?? 0;
    const ripe   = rawScores['ripe']   ?? 0;
    const over   = rawScores['overripe'] ?? 0;

    const total = unripe + ripe + over || 1;
    const scores = { unripe: unripe / total, ripe: ripe / total, overripe: over / total };

    const maxScore = Math.max(...Object.values(scores));
    const ripeness: 'unripe' | 'ripe' | 'overripe' =
      maxScore === scores.unripe ? 'unripe' :
      maxScore === scores.overripe ? 'overripe' : 'ripe';

    const sorted = Object.values(scores).sort((a, b) => b - a);
    const confidence = Math.min(0.98, sorted[0]);

    return { ripeness, scores, confidence, available: true };
  });
}
