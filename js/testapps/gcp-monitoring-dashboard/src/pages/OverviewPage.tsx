import { BarChart3, Clock, TrendingUp } from 'lucide-react';

export function OverviewPage() {
  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-xl font-semibold">Monitoring</h2>
        <span className="text-sm text-text-secondary">
          Last 24 hours
        </span>
      </div>

      {/* Placeholder charts */}
      <div className="mb-8 grid grid-cols-3 gap-4">
        <div className="card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm text-text-secondary">
            <BarChart3 className="h-4 w-4" />
            Requests
          </div>
          <div className="flex h-32 items-center justify-center text-text-tertiary">
            Chart placeholder
          </div>
        </div>
        <div className="card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm text-text-secondary">
            <TrendingUp className="h-4 w-4" />
            Success rate
          </div>
          <div className="flex h-32 items-center justify-center text-text-tertiary">
            Chart placeholder
          </div>
        </div>
        <div className="card p-4">
          <div className="mb-3 flex items-center gap-2 text-sm text-text-secondary">
            <Clock className="h-4 w-4" />
            Latency (p95)
          </div>
          <div className="flex h-32 items-center justify-center text-text-tertiary">
            Chart placeholder
          </div>
        </div>
      </div>

      {/* Placeholder feature table */}
      <div className="card">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border text-left text-sm text-text-secondary">
              <th className="px-4 py-3 font-medium">Feature</th>
              <th className="px-4 py-3 font-medium">Success rate</th>
              <th className="px-4 py-3 font-medium">Requests</th>
              <th className="px-4 py-3 font-medium">Latency (p95)</th>
              <th className="px-4 py-3 font-medium">
                Input / Output / Thinking
              </th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-border text-sm text-text-tertiary">
              <td className="px-4 py-4" colSpan={5}>
                <div className="flex items-center justify-center gap-2">
                  <span>Connect to a GCP project to see data</span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
