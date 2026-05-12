import { RefreshCw } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';
import type { TimeRangePreset } from '../api/client';

const presets: { value: TimeRangePreset; label: string }[] = [
  { value: '1h', label: '1h' },
  { value: '6h', label: '6h' },
  { value: '24h', label: '24h' },
  { value: '7d', label: '7d' },
  { value: '30d', label: '30d' },
];

export function TimeRangeSelector() {
  const { timeRangePreset, setTimeRangePreset, refresh } = useDashboard();

  return (
    <div className="flex items-center gap-1">
      <div className="flex rounded-md border border-border overflow-hidden">
        {presets.map((p) => (
          <button
            key={p.value}
            onClick={() => setTimeRangePreset(p.value)}
            className={`px-3 py-1.5 text-xs font-medium transition-colors ${
              timeRangePreset === p.value
                ? 'bg-accent-blue text-white'
                : 'bg-surface-2 text-text-secondary hover:bg-surface-3 hover:text-text-primary'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>
      <button
        onClick={refresh}
        className="ml-1 p-1.5 rounded text-text-tertiary hover:text-text-primary hover:bg-surface-3 transition-colors"
        title="Refresh data"
      >
        <RefreshCw className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
