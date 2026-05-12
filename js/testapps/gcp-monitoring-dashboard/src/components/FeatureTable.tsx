import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowUpDown,
  CheckCircle2,
  ChevronRight,
  Loader2,
  XCircle,
} from 'lucide-react';
import { useMetricsOverview } from '../hooks/useMetrics';
import type { FeatureOverview } from '../types';

type SortKey = 'name' | 'totalRequests' | 'successRate' | 'inputTokens' | 'outputTokens';
type SortDir = 'asc' | 'desc';

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function SuccessRateBadge({ rate }: { rate: number }) {
  const pct = (rate * 100).toFixed(1);
  if (rate >= 0.95) {
    return (
      <span className="inline-flex items-center gap-1 text-accent-green">
        <CheckCircle2 className="h-3.5 w-3.5" />
        {pct}%
      </span>
    );
  }
  if (rate >= 0.8) {
    return (
      <span className="inline-flex items-center gap-1 text-accent-orange">
        <CheckCircle2 className="h-3.5 w-3.5" />
        {pct}%
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-accent-red">
      <XCircle className="h-3.5 w-3.5" />
      {pct}%
    </span>
  );
}

export function FeatureTable() {
  const { data, isLoading, error } = useMetricsOverview();
  const [sortKey, setSortKey] = useState<SortKey>('totalRequests');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const features = data?.features || [];

  const sorted = [...features].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return sortDir === 'asc'
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    }
    return sortDir === 'asc'
      ? (aVal as number) - (bVal as number)
      : (bVal as number) - (aVal as number);
  });

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const SortHeader = ({
    label,
    sortId,
    className = '',
  }: {
    label: string;
    sortId: SortKey;
    className?: string;
  }) => (
    <th className={`px-4 py-3.5 font-medium ${className}`}>
      <button
        onClick={() => toggleSort(sortId)}
        className="flex items-center gap-1 hover:text-text-primary transition-colors"
      >
        {label}
        <ArrowUpDown
          className={`h-3 w-3 ${
            sortKey === sortId ? 'text-accent-blue' : 'text-text-tertiary'
          }`}
        />
      </button>
    </th>
  );

  if (isLoading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-5 w-5 animate-spin text-text-tertiary" />
          <span className="ml-2 text-sm text-text-tertiary">Loading features...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12 text-sm text-accent-red">
          Failed to load feature data
        </div>
      </div>
    );
  }

  return (
    <div className="card overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border text-left text-xs text-text-secondary">
            <SortHeader label="Feature" sortId="name" />
            <SortHeader label="Success Rate" sortId="successRate" />
            <SortHeader label="Requests" sortId="totalRequests" />
            <SortHeader label="Input Tokens" sortId="inputTokens" />
            <SortHeader label="Output Tokens" sortId="outputTokens" />
            <th className="w-8 px-2 py-3"></th>
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr>
              <td
                className="px-4 py-8 text-center text-sm text-text-tertiary"
                colSpan={6}
              >
                No features found in this time range
              </td>
            </tr>
          ) : (
            sorted.map((feature) => (
              <FeatureRow key={feature.name} feature={feature} />
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function FeatureRow({ feature }: { feature: FeatureOverview }) {
  return (
    <tr className="group border-b border-border/60 last:border-0 hover:bg-surface-2/60 transition-colors">
      <td className="px-4 py-3.5">
        <Link
          to={`/feature/${encodeURIComponent(feature.name)}`}
          className="text-sm font-medium text-accent-blue hover:underline"
        >
          {feature.name}
        </Link>
      </td>
      <td className="px-4 py-3.5 text-sm">
        <SuccessRateBadge rate={feature.successRate} />
      </td>
      <td className="px-4 py-3.5 text-sm text-text-primary">
        <span>{formatNumber(feature.totalRequests)}</span>
        {feature.failureCount > 0 && (
          <span className="ml-1 text-xs text-accent-red">
            ({formatNumber(feature.failureCount)} failed)
          </span>
        )}
      </td>
      <td className="px-4 py-3.5 text-sm text-text-secondary">
        {formatNumber(feature.inputTokens)}
      </td>
      <td className="px-4 py-3.5 text-sm text-text-secondary">
        {formatNumber(feature.outputTokens)}
      </td>
      <td className="px-2 py-3.5">
        <Link
          to={`/feature/${encodeURIComponent(feature.name)}`}
          className="text-text-tertiary group-hover:text-text-secondary transition-colors"
        >
          <ChevronRight className="h-4 w-4" />
        </Link>
      </td>
    </tr>
  );
}
