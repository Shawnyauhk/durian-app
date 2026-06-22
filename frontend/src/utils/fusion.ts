import type { FusionResult, ModalityResult, RipenessLevel } from '../types';

const ACOUSTIC_WEIGHT = 0.6;
const VISION_WEIGHT = 0.4;

/**
 * Score mapping: unripe=0, ripe=0.5, overripe=1
 * Center (ripe) maps to ~0.5
 */
function ripenessToScore(scores: { unripe: number; ripe: number; overripe: number }): number {
  return scores.unripe * 0 + scores.ripe * 0.5 + scores.overripe * 1.0;
}

function scoreToRipeness(score: number): RipenessLevel {
  if (score < 0.33) return 'unripe';
  if (score < 0.67) return 'ripe';
  return 'overripe';
}

function detectContradiction(
  v: ModalityResult | null,
  a: ModalityResult | null
): boolean {
  if (!v || !a) return false;
  const vs = ripenessToScore(v.scores);
  const as = ripenessToScore(a.scores);
  // Contradiction if one says unripe (<0.25) and the other says overripe (>0.75)
  return Math.abs(vs - as) > 0.5;
}

export function computeFusion(
  vision: ModalityResult | null,
  acoustic: ModalityResult | null
): FusionResult {
  const hasVision = vision?.available && !vision.error;
  const hasAcoustic = acoustic?.available && !acoustic.error;

  let weightedScore: number;
  let confidence: number;

  if (hasVision && hasAcoustic) {
    const vs = ripenessToScore(vision!.scores);
    const as = ripenessToScore(acoustic!.scores);
    weightedScore = VISION_WEIGHT * vs + ACOUSTIC_WEIGHT * as;
    // Base confidence from weighted average of individual confidences
    const rawConfidence = VISION_WEIGHT * vision!.confidence + ACOUSTIC_WEIGHT * acoustic!.confidence;
    const contradicting = detectContradiction(vision, acoustic);
    confidence = contradicting ? rawConfidence * 0.65 : rawConfidence;
  } else if (hasAcoustic) {
    weightedScore = ripenessToScore(acoustic!.scores);
    confidence = acoustic!.confidence * 0.85; // slight penalty for single modality
  } else if (hasVision) {
    weightedScore = ripenessToScore(vision!.scores);
    confidence = vision!.confidence * 0.8; // larger penalty, vision less reliable alone
  } else {
    return {
      ripeness: 'unknown',
      confidence: 0,
      weightedScore: 0.5,
      vision: vision ?? undefined,
      acoustic: acoustic ?? undefined,
    };
  }

  const ripeness = scoreToRipeness(weightedScore);
  const contradictionWarning = detectContradiction(vision, acoustic);

  return {
    ripeness,
    confidence: Math.min(1, Math.max(0, confidence)),
    weightedScore: Math.min(1, Math.max(0, weightedScore)),
    vision: hasVision ? vision! : undefined,
    acoustic: hasAcoustic ? acoustic! : undefined,
    contradictionWarning,
  };
}
