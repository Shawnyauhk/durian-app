import React from 'react';
import type { AppStep } from '../types';

interface Step {
  id: AppStep;
  icon: string;
  label: string;
}

const STEPS: Step[] = [
  { id: 'vision', icon: '📸', label: 'AI眼' },
  { id: 'acoustic', icon: '🔊', label: 'AI耳' },
  { id: 'result', icon: '🧠', label: '判定' },
];

interface Props {
  currentStep: AppStep;
}

export const ProgressStepper: React.FC<Props> = ({ currentStep }) => {
  const currentIndex = STEPS.findIndex(s => s.id === currentStep);

  return (
    <div className="flex items-center justify-center gap-1 py-3 px-4">
      {STEPS.map((step, index) => {
        const isCompleted = index < currentIndex;
        const isCurrent = index === currentIndex;
        return (
          <React.Fragment key={step.id}>
            <div className="flex flex-col items-center gap-1">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-lg transition-all
                  ${isCompleted ? 'bg-emerald-500 text-white shadow-md' : ''}
                  ${isCurrent ? 'bg-durian-green text-white shadow-lg ring-2 ring-durian-green ring-offset-2' : ''}
                  ${!isCompleted && !isCurrent ? 'bg-gray-100 text-gray-400' : ''}
                `}
              >
                {isCompleted ? '✓' : step.icon}
              </div>
              <span
                className={`text-xs font-medium ${
                  isCurrent ? 'text-durian-green' : isCompleted ? 'text-emerald-500' : 'text-gray-400'
                }`}
              >
                {step.label}
              </span>
            </div>
            {index < STEPS.length - 1 && (
              <div
                className={`flex-1 h-0.5 mb-4 mx-1 transition-all ${
                  index < currentIndex ? 'bg-emerald-400' : 'bg-gray-200'
                }`}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
