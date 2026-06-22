import React from 'react';
import { RIPENESS_LABELS } from '../types';

interface ConfidenceBarProps {
  scores: {
    unripe: number;
    ripe: number;
    overripe: number;
  };
  label?: string;
  icon?: string;
}

const BAR_COLORS = {
  unripe: 'bg-amber-400',
  ripe: 'bg-emerald-500',
  overripe: 'bg-red-400',
};

export const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ scores, label, icon }) => {
  const entries = [
    { key: 'unripe' as const, value: scores.unripe },
    { key: 'ripe' as const, value: scores.ripe },
    { key: 'overripe' as const, value: scores.overripe },
  ];

  return (
    <div className="space-y-2">
      {label && (
        <div className="flex items-center gap-1.5 text-sm font-medium text-gray-600">
          {icon && <span>{icon}</span>}
          <span>{label}</span>
        </div>
      )}
      {entries.map(({ key, value }) => (
        <div key={key} className="flex items-center gap-2">
          <span className="text-xs text-gray-500 w-10 text-right shrink-0">
            {RIPENESS_LABELS[key]}
          </span>
          <div className="flex-1 bg-gray-100 rounded-full h-2.5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${BAR_COLORS[key]}`}
              style={{ width: `${(value * 100).toFixed(1)}%` }}
            />
          </div>
          <span className="text-xs text-gray-500 w-10 shrink-0">
            {(value * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  );
};
