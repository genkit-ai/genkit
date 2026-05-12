import { useState, useEffect } from 'react';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import { useTraceDetail } from '../hooks/useTraces';
import { SpanTree } from '../components/SpanTree';
import { SpanDetail } from '../components/SpanDetail';
import type { SpanData } from '../types';

export function TraceViewerPage() {
  const { traceId } = useParams<{ traceId: string }>();
  const { data: trace, isLoading, error } = useTraceDetail(traceId);
  const [selectedSpan, setSelectedSpan] = useState<SpanData | null>(null);

  // Auto-select root span when trace loads
  useEffect(() => {
    if (trace?.rootSpan && !selectedSpan) {
      setSelectedSpan(trace.rootSpan);
    }
  }, [trace?.rootSpan]);

  const featureName = trace?.rootSpan?.featureName;

  return (
    <div className="flex h-[calc(100vh-64px)] flex-col">
      {/* Breadcrumb */}
      <div className="flex-shrink-0 px-0 pb-4">
        <div className="mb-1 flex items-center gap-2 text-sm text-text-secondary">
          <Link to="/" className="hover:text-accent-blue">
            Genkit
          </Link>
          <span>›</span>
          {featureName ? (
            <>
              <Link
                to={`/feature/${featureName}`}
                className="hover:text-accent-blue"
              >
                {featureName}
              </Link>
              <span>›</span>
            </>
          ) : null}
          <span className="text-text-primary">Trace viewer</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => window.history.back()}
            className="text-text-secondary hover:text-text-primary"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <h2 className="text-xl font-semibold">
            {trace?.rootSpan?.name || 'Trace viewer'}
          </h2>
          {traceId && (
            <code className="rounded bg-surface-2 px-2 py-0.5 font-mono text-xs text-text-tertiary">
              {traceId.slice(0, 16)}…
            </code>
          )}
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="card flex flex-1 items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-text-tertiary" />
        </div>
      ) : error ? (
        <div className="card flex flex-1 items-center justify-center">
          <p className="text-sm text-accent-red">
            Failed to load trace. Please check the trace ID and project.
          </p>
        </div>
      ) : !trace?.rootSpan ? (
        <div className="card flex flex-1 items-center justify-center">
          <p className="text-sm text-text-tertiary">
            No spans found for this trace.
          </p>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 gap-4">
          {/* Left: Span Tree */}
          <div className="card flex w-1/2 flex-col overflow-hidden">
            <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
              <h3 className="text-sm font-medium text-text-secondary">
                Span Tree
              </h3>
              <span className="text-xs text-text-tertiary">
                {countSpans(trace.rootSpan)} spans
              </span>
            </div>
            <div className="flex-1 overflow-y-auto py-1">
              <SpanTree
                rootSpan={trace.rootSpan}
                selectedSpanId={selectedSpan?.spanId || null}
                onSelectSpan={setSelectedSpan}
              />
            </div>
          </div>

          {/* Right: Span Detail */}
          <div className="card flex w-1/2 flex-col overflow-hidden">
            {selectedSpan ? (
              <SpanDetail span={selectedSpan} traceId={traceId!} />
            ) : (
              <div className="flex flex-1 items-center justify-center">
                <p className="text-sm text-text-tertiary">
                  Select a span to view details
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/** Count total spans in a tree recursively */
function countSpans(span: SpanData): number {
  let count = 1;
  for (const child of span.children || []) {
    count += countSpans(child);
  }
  return count;
}
