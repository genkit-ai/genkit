import { useEffect, useState, type ReactNode } from 'react';
import { AlertCircle, CheckCircle2, Loader2, Terminal } from 'lucide-react';
import type { HealthResponse } from '../types';

/**
 * AuthGate checks that the backend is running and ADC is configured.
 * Shows helpful instructions if something is wrong.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<'loading' | 'ok' | 'error'>('loading');
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetch('/api/health')
      .then((r) => {
        if (!r.ok) throw new Error(`Backend returned ${r.status}`);
        return r.json() as Promise<HealthResponse>;
      })
      .then((data) => {
        if (data.authError) {
          setError(data.authError);
          setState('error');
        } else {
          setState('ok');
        }
      })
      .catch((err) => {
        setError(
          err.message?.includes('fetch')
            ? 'Cannot connect to the backend server at localhost:3000. Make sure it is running.'
            : err.message
        );
        setState('error');
      });
  }, []);

  if (state === 'loading') {
    return (
      <div className="flex min-h-screen items-center justify-center bg-surface-0">
        <Loader2 className="h-8 w-8 animate-spin text-text-tertiary" />
      </div>
    );
  }

  if (state === 'error') {
    return (
      <div className="flex min-h-screen items-center justify-center bg-surface-0 p-8">
        <div className="card max-w-lg p-8">
          <div className="mb-4 flex items-center gap-3">
            <AlertCircle className="h-8 w-8 text-accent-red" />
            <h2 className="text-lg font-semibold text-text-primary">
              Authentication Required
            </h2>
          </div>
          <p className="mb-4 text-sm text-text-secondary">{error}</p>
          <div className="rounded-md bg-surface-2 p-4">
            <p className="mb-2 text-xs font-medium text-text-secondary">
              Run these commands to set up Application Default Credentials:
            </p>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Terminal className="h-3.5 w-3.5 text-text-tertiary" />
                <code className="font-mono text-xs text-accent-blue">
                  gcloud auth application-default login
                </code>
              </div>
              <div className="flex items-center gap-2">
                <Terminal className="h-3.5 w-3.5 text-text-tertiary" />
                <code className="font-mono text-xs text-accent-blue">
                  gcloud config set project YOUR_PROJECT_ID
                </code>
              </div>
            </div>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 inline-flex items-center gap-2 rounded-md bg-accent-blue px-4 py-2 text-sm font-medium text-white hover:bg-accent-blue/80"
          >
            <CheckCircle2 className="h-4 w-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
