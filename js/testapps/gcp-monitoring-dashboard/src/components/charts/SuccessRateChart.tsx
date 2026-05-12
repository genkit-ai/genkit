import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { TrendingUp, Loader2 } from 'lucide-react';
import { useTimeSeries } from '../../hooks/useMetrics';

const METRIC_TYPE = 'workload.googleapis.com/genkit/feature/requests';

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function SuccessRateChart() {
  const { data, isLoading, error } = useTimeSeries(METRIC_TYPE, {
    groupBy: 'metric.label.status',
    reducer: 'REDUCE_SUM',
  });

  // Compute success rate over time
  const chartData = (() => {
    if (!data?.timeSeries?.length) return [];

    const timeMap = new Map<string, { success: number; total: number }>();

    for (const series of data.timeSeries) {
      const isSuccess = series.labels?.status === 'success';

      for (const point of series.points) {
        if (!timeMap.has(point.time)) {
          timeMap.set(point.time, { success: 0, total: 0 });
        }
        const entry = timeMap.get(point.time)!;
        entry.total += point.value;
        if (isSuccess) entry.success += point.value;
      }
    }

    return Array.from(timeMap.entries())
      .map(([time, { success, total }]) => ({
        time,
        rate: total > 0 ? (success / total) * 100 : 0,
      }))
      .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
  })();

  // Overall success rate
  const overallRate = (() => {
    if (!chartData.length) return null;
    const totalSuccess = chartData.reduce((s, d) => s + d.rate, 0);
    return totalSuccess / chartData.length;
  })();

  return (
    <div className="card p-4">
      <div className="mb-1 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <TrendingUp className="h-4 w-4" />
          Success Rate
        </div>
        {overallRate !== null && (
          <span
            className={`text-lg font-semibold ${
              overallRate >= 95
                ? 'text-accent-green'
                : overallRate >= 80
                  ? 'text-accent-orange'
                  : 'text-accent-red'
            }`}
          >
            {overallRate.toFixed(1)}%
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
            <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
              <XAxis
                dataKey="time"
                tickFormatter={formatTime}
                tick={{ fontSize: 10, fill: '#6e7681' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
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
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Success Rate']}
              />
              <Line
                type="monotone"
                dataKey="rate"
                stroke="#3fb950"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 3, fill: '#3fb950' }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
