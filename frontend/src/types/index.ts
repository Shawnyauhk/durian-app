export type RipenessLevel = 'unripe' | 'ripe' | 'overripe' | 'unknown';

export interface ModalityResult {
  ripeness: RipenessLevel;
  scores: {
    unripe: number;
    ripe: number;
    overripe: number;
  };
  confidence: number;
  available: boolean;
  error?: string;
}

export interface FusionResult {
  ripeness: RipenessLevel;
  confidence: number;
  weightedScore: number;
  vision?: ModalityResult;
  acoustic?: ModalityResult;
  defectWarning?: string;
  contradictionWarning?: boolean;
}

export type AppStep = 'intro' | 'vision' | 'acoustic' | 'result';

export interface AppState {
  step: AppStep;
  visionResult: ModalityResult | null;
  acousticResult: ModalityResult | null;
  fusionResult: FusionResult | null;
  isLoading: boolean;
}

export const RIPENESS_LABELS: Record<RipenessLevel, string> = {
  unripe: '未熟',
  ripe: '成熟',
  overripe: '過熟',
  unknown: '未知',
};

export const RIPENESS_EMOJI: Record<RipenessLevel, string> = {
  unripe: '🟡',
  ripe: '🟢',
  overripe: '🔴',
  unknown: '⚪',
};

export const RIPENESS_COLORS: Record<RipenessLevel, string> = {
  unripe: '#f59e0b',
  ripe: '#10b981',
  overripe: '#ef4444',
  unknown: '#9ca3af',
};

export const RIPENESS_BG: Record<RipenessLevel, string> = {
  unripe: 'bg-amber-50 border-amber-200',
  ripe: 'bg-emerald-50 border-emerald-200',
  overripe: 'bg-red-50 border-red-200',
  unknown: 'bg-gray-50 border-gray-200',
};
