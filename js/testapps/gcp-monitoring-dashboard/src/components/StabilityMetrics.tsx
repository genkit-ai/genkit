import { Activity, CheckCircle2, Clock, Hash, XCircle } from 'lucide-react';
import type { FeatureOverview } from '../types';

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatLatency(ms: number | null): string {
  if (ms === null) return '—';
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

export function StabilityMetrics({ feature }: { feature: FeatureOverview }) {
  const successPct = (feature.successRate * 100).toFixed(1);
  const isHealthy = feature.successRate >= 0.95;

  return (
    <div className="card mb-6 grid grid-cols-2 gap-0 divide-x divide-border md:grid-cols-5">
      {/* Total Requests */}
      <div className="p-4">
        <div className="mb-1 flex items-center gap-1.5 text-xs text-text-tertiary">
          <Activity className="h-3 w-3" />
          Total Requests
        </div>
        <div className="text-xl font-semibold text-text-primary">
          {formatNumber(feature.totalRequests)}
        </div>
      </div>

      {/* Success Rate */}
      <div className="p-4">
        <div className="mb-1 flex items-center gap-1.5 text-xs text-text-tertiary">
          {isHealthy ? (
            <CheckCircle2 className="h-3 w-3 text-accent-green" />
          ) : (
            <XCircle className="h-3 w-3 text-accent-red" />
          )}
          Success Rate
        </div>
        <div
          className={`text-xl font-semibold ${
            isHealthy ? 'text-accent-green' : feature.successRate >= 0.8 ? 'text-accent-orange' : 'text-accent-red'
          }`}
        >
          {successPct}%
        </div>
        {feature.failureCount > 0 && (
          <div className="mt-0.5 text-xs text-accent-red">
            {formatNumber(feature.failureCount)} failures
          </div>
        )}
      </div>

      {/* Latency */}
      <div className="p-4">
        <div className="mb-1 flex items-center gap-1.5 text-xs text-text-tertiary">
          <Clock className="h-3 w-3" />
          Latency p95
        </div>
        <div className="text-xl font-semibold text-text-primary">
          {formatLatency(feature.latencyP95Ms)}
        </div>
      </div>

      {/* Input Tokens */}
      <div className="p-4">
        <div className="mb-1 flex items-center gap-1.5 text-xs text-text-tertiary">
          <Hash className="h-3 w-3" />
          Input Tokens
        </div>
        <div className="text-xl font-semibold text-text-primary">
          {formatNumber(feature.inputTokens)}
        </div>
      </div>

      {/* Output Tokens */}
      <div className="p-4">
        <div className="mb-1 flex items-center gap-1.5 text-xs text-text-tertiary">
          <Hash className="h-3 w-3" />
          Output Tokens
        </div>
        <div className="text-xl font-semibold text-text-primary">
          {formatNumber(feature.outputTokens)}
        </div>
      </div>
    </div>
  );
}
