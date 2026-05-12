import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { BarChart3, Loader2 } from 'lucide-react';
import { useTimeSeries } from '../../hooks/useMetrics';

const METRIC_TYPE = 'workload.googleapis.com/genkit/feature/requests';

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function RequestsChart() {
  const { data, isLoading, error } = useTimeSeries(METRIC_TYPE, {
    groupBy: 'metric.label.status',
    reducer: 'REDUCE_SUM',
  });

  // Merge all series into a single timeline with success/failure columns
  const chartData = (() => {
    if (!data?.timeSeries?.length) return [];

    // Collect all time points and values by status
    const timeMap = new Map<string, { time: string; success: number; failure: number }>();

    for (const series of data.timeSeries) {
      const status = series.labels?.status || 'unknown';
      const key = status === 'success' ? 'success' : 'failure';

      for (const point of series.points) {
        if (!timeMap.has(point.time)) {
          timeMap.set(point.time, { time: point.time, success: 0, failure: 0 });
        }
        const entry = timeMap.get(point.time)!;
        entry[key] += point.value;
      }
    }

    return Array.from(timeMap.values()).sort(
      (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime()
    );
  })();

  const totalRequests = chartData.reduce(
    (sum, d) => sum + d.success + d.failure,
    0
  );

  return (
    <div className="card p-4">
      <div className="mb-1 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <BarChart3 className="h-4 w-4" />
          Requests
        </div>
        {!isLoading && !error && (
          <span className="text-lg font-semibold text-text-primary">
            {formatNumber(totalRequests)}
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
            <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
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
                  name === 'success' ? 'Success' : 'Failure',
                ]}
              />
              <Area
                type="monotone"
                dataKey="success"
                stackId="1"
                stroke="#3fb950"
                fill="#3fb950"
                fillOpacity={0.3}
              />
              <Area
                type="monotone"
                dataKey="failure"
                stackId="1"
                stroke="#f85149"
                fill="#f85149"
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
