import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Clock, Loader2 } from 'lucide-react';
import { useTimeSeries } from '../../hooks/useMetrics';

const METRIC_TYPE = 'workload.googleapis.com/genkit/feature/latency';

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatLatency(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

export function LatencyChart() {
  // For distribution metrics, ALIGN_DELTA converts cumulative distribution to delta,
  // then REDUCE_PERCENTILE_95 gives us p95 across series
  const { data, isLoading, error } = useTimeSeries(METRIC_TYPE, {
    aligner: 'ALIGN_DELTA',
    reducer: 'REDUCE_PERCENTILE_99',
  });

  const chartData = (() => {
    if (!data?.timeSeries?.length) return [];

    // For distribution metrics the normalized value is the mean;
    // if we get raw points, use them directly
    const allPoints: { time: string; latency: number }[] = [];

    for (const series of data.timeSeries) {
      for (const point of series.points) {
        allPoints.push({
          time: point.time,
          latency: point.value,
        });
      }
    }

    // Merge by time (sum if multiple series, though reducer should handle it)
    const timeMap = new Map<string, number>();
    for (const p of allPoints) {
      timeMap.set(p.time, (timeMap.get(p.time) || 0) + p.latency);
    }

    return Array.from(timeMap.entries())
      .map(([time, latency]) => ({ time, latency }))
      .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
  })();

  const avgLatency =
    chartData.length > 0
      ? chartData.reduce((s, d) => s + d.latency, 0) / chartData.length
      : null;

  return (
    <div className="card p-4">
      <div className="mb-1 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <Clock className="h-4 w-4" />
          Latency (p99)
        </div>
        {avgLatency !== null && (
          <span className="text-lg font-semibold text-text-primary">
            {formatLatency(avgLatency)}
          </span>
        )}
      </div>

      <div className="h-36">
        {isLoading ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="h-5 w-5 animate-spin text-text-tertiary" />
          </div>
        ) : error ? (
          <div className="flex h-full items-center justify-center text-xs text-accent-red">
            Failed to load
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex h-full items-center justify-center text-xs text-text-tertiary">
            No data in this time range
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
              <XAxis
                dataKey="time"
                tickFormatter={formatTime}
                tick={{ fontSize: 10, fill: '#6e7681' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tickFormatter={formatLatency}
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
                formatter={(value: number) => [formatLatency(value), 'p99 Latency']}
              />
              <Line
                type="monotone"
                dataKey="latency"
                stroke="#58a6ff"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3, fill: '#58a6ff' }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
