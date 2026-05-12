import { ArrowLeft } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';

export function TraceViewerPage() {
  const { traceId } = useParams<{ traceId: string }>();

  return (
    <div>
      {/* Breadcrumb */}
      <div className="mb-6">
        <div className="mb-1 flex items-center gap-2 text-sm text-text-secondary">
          <Link to="/" className="hover:text-accent-blue">
            Genkit
          </Link>
          <span>›</span>
          <span>Trace viewer</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => window.history.back()}
            className="text-text-secondary hover:text-text-primary"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <h2 className="text-xl font-semibold">Trace viewer</h2>
        </div>
      </div>

      {/* Placeholder content */}
      <div className="card flex items-center justify-center p-12">
        <p className="text-text-tertiary">
          Trace viewer — will show span tree and detail for trace{' '}
          <code className="rounded bg-surface-3 px-2 py-0.5 font-mono text-xs text-text-secondary">
            {traceId}
          </code>
        </p>
      </div>
    </div>
  );
}
