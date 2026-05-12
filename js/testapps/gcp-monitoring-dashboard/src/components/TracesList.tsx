import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Loader2,
  XCircle,
} from 'lucide-react';
import { useTraceList } from '../hooks/useTraces';

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function formatDuration(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'success')
    return <CheckCircle2 className="h-4 w-4 text-accent-green" />;
  if (status === 'error')
    return <XCircle className="h-4 w-4 text-accent-red" />;
  return <AlertCircle className="h-4 w-4 text-text-tertiary" />;
}

export function TracesList({
  featureName,
  pageSize = 15,
}: {
  featureName: string;
  pageSize?: number;
}) {
  const [pageTokens, setPageTokens] = useState<string[]>([]);
  const [statusFilter, setStatusFilter] = useState<string>('');
  const currentPageToken = pageTokens[pageTokens.length - 1];

  const { data, isLoading, error } = useTraceList({
    featureName,
    status: statusFilter || undefined,
    pageSize,
    pageToken: currentPageToken,
  });

  const traces = data?.traces || [];
  const hasNextPage = !!data?.nextPageToken;
  const hasPrevPage = pageTokens.length > 0;
  const pageNum = pageTokens.length + 1;

  const goNext = () => {
    if (data?.nextPageToken) {
      setPageTokens((prev) => [...prev, data.nextPageToken!]);
    }
  };

  const goPrev = () => {
    setPageTokens((prev) => prev.slice(0, -1));
  };

  return (
    <div>
      {/* Filter bar */}
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-text-secondary">Traces</h3>
        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPageTokens([]);
            }}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-xs text-text-primary focus:border-accent-blue focus:outline-none"
          >
            <option value="">All statuses</option>
            <option value="success">Success</option>
            <option value="error">Error</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border text-left text-xs text-text-secondary">
              <th className="w-8 px-3 py-2.5"></th>
              <th className="px-3 py-2.5 font-medium">Time</th>
              <th className="px-3 py-2.5 font-medium">Name</th>
              <th className="px-3 py-2.5 font-medium">Type</th>
              <th className="px-3 py-2.5 font-medium text-right">Duration</th>
              <th className="w-8 px-2 py-2.5"></th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center">
                  <Loader2 className="mx-auto h-5 w-5 animate-spin text-text-tertiary" />
                </td>
              </tr>
            ) : error ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-sm text-accent-red">
                  Failed to load traces
                </td>
              </tr>
            ) : traces.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-sm text-text-tertiary">
                  No traces found
                </td>
              </tr>
            ) : (
              traces.map((trace) => (
                <tr
                  key={trace.traceId}
                  className="group border-b border-border last:border-0 hover:bg-surface-2 transition-colors"
                >
                  <td className="px-3 py-2.5">
                    <StatusIcon status={trace.rootSpan.status} />
                  </td>
                  <td className="px-3 py-2.5 text-xs text-text-secondary whitespace-nowrap">
                    {trace.rootSpan.startTime
                      ? formatTime(trace.rootSpan.startTime)
                      : '—'}
                  </td>
                  <td className="px-3 py-2.5">
                    <Link
                      to={`/trace/${trace.traceId}`}
                      className="text-sm text-accent-blue hover:underline"
                    >
                      {trace.rootSpan.name || trace.traceId.slice(0, 12)}
                    </Link>
                  </td>
                  <td className="px-3 py-2.5">
                    {trace.rootSpan.subtype && (
                      <span className={`type-badge type-badge-${trace.rootSpan.subtype}`}>
                        {trace.rootSpan.subtype}
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 text-right text-xs text-text-secondary whitespace-nowrap">
                    {formatDuration(trace.rootSpan.durationMs)}
                  </td>
                  <td className="px-2 py-2.5">
                    <Link
                      to={`/trace/${trace.traceId}`}
                      className="text-text-tertiary group-hover:text-text-secondary transition-colors"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Link>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        {/* Pagination */}
        {(hasPrevPage || hasNextPage) && (
          <div className="flex items-center justify-between border-t border-border px-4 py-2">
            <span className="text-xs text-text-tertiary">Page {pageNum}</span>
            <div className="flex gap-1">
              <button
                onClick={goPrev}
                disabled={!hasPrevPage}
                className="rounded p-1 text-text-secondary hover:bg-surface-3 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                onClick={goNext}
                disabled={!hasNextPage}
                className="rounded p-1 text-text-secondary hover:bg-surface-3 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
