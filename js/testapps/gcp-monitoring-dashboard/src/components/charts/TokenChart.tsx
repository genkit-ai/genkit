import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Hash, Loader2 } from 'lucide-react';
import { useTimeSeries } from '../../hooks/useMetrics';

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function TokenChart({ featureName }: { featureName?: string }) {
  const filterStr = featureName
    ? `metric.label.featureName="${featureName}"`
    : undefined;

  const { data: inputData, isLoading: loadingInput } = useTimeSeries(
    'workload.googleapis.com/genkit/ai/generate/input/tokens',
    { filter: filterStr, reducer: 'REDUCE_SUM' }
  );

  const { data: outputData, isLoading: loadingOutput } = useTimeSeries(
    'workload.googleapis.com/genkit/ai/generate/output/tokens',
    { filter: filterStr, reducer: 'REDUCE_SUM' }
  );

  const isLoading = loadingInput || loadingOutput;

  const chartData = (() => {
    const timeMap = new Map<string, { time: string; input: number; output: number }>();

    for (const series of inputData?.timeSeries || []) {
      for (const p of series.points) {
        if (!timeMap.has(p.time)) {
          timeMap.set(p.time, { time: p.time, input: 0, output: 0 });
        }
        timeMap.get(p.time)!.input += p.value;
      }
    }

    for (const series of outputData?.timeSeries || []) {
      for (const p of series.points) {
        if (!timeMap.has(p.time)) {
          timeMap.set(p.time, { time: p.time, input: 0, output: 0 });
        }
        timeMap.get(p.time)!.output += p.value;
      }
    }

    return Array.from(timeMap.values()).sort(
      (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime()
    );
  })();

  return (
    <div className="card p-4">
      <div className="mb-1 flex items-center gap-2 text-sm text-text-secondary">
        <Hash className="h-4 w-4" />
        Token Usage
      </div>

      <div className="h-44">
        {isLoading ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="h-5 w-5 animate-spin text-text-tertiary" />
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex h-full items-center justify-center text-xs text-text-tertiary">
            No token data
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
              <XAxis
                dataKey="time"
                tickFormatter={formatTime}
                tick={{ fontSize: 10, fill: '#6e7681' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tickFormatter={formatNumber}
                tick={{ fontSize: 10, fill: '#6e7681' }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1c2128',
                  border: '1px solid #30363d',
                  borderRadius: 6,
                  fontSize: 12,
                }}
                labelFormatter={(v) => new Date(v).toLocaleString()}
                formatter={(value: number, name: string) => [
                  formatNumber(value),
                  name === 'input' ? 'Input Tokens' : 'Output Tokens',
                ]}
              />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value) =>
                  value === 'input' ? 'Input' : 'Output'
                }
              />
              <Bar dataKey="input" fill="#58a6ff" fillOpacity={0.8} radius={[2, 2, 0, 0]} />
              <Bar dataKey="output" fill="#d29922" fillOpacity={0.8} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
