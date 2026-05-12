/**
 * Typed fetch wrapper for the backend API.
 *
 * All requests go through the Vite dev proxy (/api/* → localhost:3000).
 * In production builds, serve the frontend from the Express server directly.
 */

import type {
  AuthStatusResponse,
  HealthResponse,
  MetricsOverviewResponse,
  ProjectsResponse,
  TimeSeriesResponse,
  TracesResponse,
  TraceDetailResponse,
} from '../types/index.js';

// ── Helpers ──────────────────────────────────────────────────────────

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public body?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);

  if (!response.ok) {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = await response.text();
    }
    const msg =
      body && typeof body === 'object' && 'error' in body
        ? (body as { error: string }).error
        : `API error ${response.status}`;
    throw new ApiError(msg, response.status, body);
  }

  return response.json() as Promise<T>;
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const entries = Object.entries(params).filter(
    ([, v]) => v !== undefined && v !== null && v !== ''
  );
  if (entries.length === 0) return '';
  return '?' + new URLSearchParams(
    entries.map(([k, v]) => [k, String(v)])
  ).toString();
}

// ── Time Range Helpers ───────────────────────────────────────────────

export type TimeRangePreset = '1h' | '6h' | '24h' | '7d' | '30d';

export function presetToTimeRange(preset: TimeRangePreset): {
  startTime: string;
  endTime: string;
} {
  const endTime = new Date().toISOString();
  const ms: Record<TimeRangePreset, number> = {
    '1h': 60 * 60 * 1000,
    '6h': 6 * 60 * 60 * 1000,
    '24h': 24 * 60 * 60 * 1000,
    '7d': 7 * 24 * 60 * 60 * 1000,
    '30d': 30 * 24 * 60 * 60 * 1000,
  };
  const startTime = new Date(Date.now() - ms[preset]).toISOString();
  return { startTime, endTime };
}

// ── API Client ───────────────────────────────────────────────────────

export const api = {
  // ── Health & Auth ─────────────────────────────────────────────────

  health(): Promise<HealthResponse> {
    return fetchJson('/api/health');
  },

  authStatus(): Promise<AuthStatusResponse> {
    return fetchJson('/api/auth/status');
  },

  projects(): Promise<ProjectsResponse> {
    return fetchJson('/api/projects');
  },

  // ── Cache ─────────────────────────────────────────────────────────

  cacheStats(): Promise<{ size: number; maxSize: number }> {
    return fetchJson('/api/cache/stats');
  },

  cacheClear(): Promise<{ cleared: number; message: string }> {
    return fetchJson('/api/cache/clear', { method: 'POST' });
  },

  // ── Metrics ───────────────────────────────────────────────────────

  metricsOverview(params: {
    projectId?: string;
    startTime?: string;
    endTime?: string;
  } = {}): Promise<MetricsOverviewResponse> {
    return fetchJson(`/api/metrics/overview${qs(params)}`);
  },

  timeseries(params: {
    metricType: string;
    projectId?: string;
    startTime?: string;
    endTime?: string;
    alignmentPeriod?: string;
    aligner?: string;
    reducer?: string;
    groupBy?: string;
    filter?: string;
  }): Promise<TimeSeriesResponse> {
    return fetchJson(`/api/metrics/timeseries${qs(params)}`);
  },

  // ── Traces ────────────────────────────────────────────────────────

  traces(params: {
    projectId?: string;
    startTime?: string;
    endTime?: string;
    filter?: string;
    featureName?: string;
    status?: string;
    pageSize?: number;
    pageToken?: string;
    orderBy?: string;
  } = {}): Promise<TracesResponse> {
    return fetchJson(`/api/traces${qs(params)}`);
  },

  traceDetail(
    traceId: string,
    params: { projectId?: string } = {}
  ): Promise<TraceDetailResponse> {
    return fetchJson(`/api/traces/${encodeURIComponent(traceId)}${qs(params)}`);
  },
};

export { ApiError };
export default api;
