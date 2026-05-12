// ============================================================
// API Response Types (shared between frontend and conceptually with backend)
// ============================================================

// --- Projects ---

export interface Project {
  projectId: string;
  name: string;
}

export interface ProjectsResponse {
  projects: Project[];
}

// --- Time Range ---

export interface TimeRange {
  startTime: string; // ISO 8601
  endTime: string; // ISO 8601
}

export type TimeRangePreset = '1h' | '6h' | '24h' | '7d' | '30d';

// --- Metrics Overview ---

export interface FeatureOverview {
  name: string;
  totalRequests: number;
  successCount: number;
  failureCount: number;
  successRate: number; // 0-1
  latencyP50Ms: number | null;
  latencyP95Ms: number | null;
  inputTokens: number;
  outputTokens: number;
  thinkingTokens: number;
  inputImages: number;
  outputImages: number;
}

export interface MetricsOverviewResponse {
  features: FeatureOverview[];
}

// --- Time Series ---

export interface TimeSeriesPoint {
  time: string; // ISO 8601
  value: number;
}

export interface TimeSeries {
  labels: Record<string, string>;
  points: TimeSeriesPoint[];
}

export interface TimeSeriesResponse {
  timeSeries: TimeSeries[];
}

// --- Traces ---

export interface TraceListItem {
  traceId: string;
  rootSpan: {
    spanId: string;
    name: string;
    startTime: string;
    endTime: string;
    status: 'success' | 'error' | 'unknown';
    durationMs: number;
    featureName: string;
    type: string;
    subtype: string;
    input: string;
    output: string;
  };
  spanCount: number;
  models: string[];
}

export interface TracesResponse {
  traces: TraceListItem[];
  nextPageToken?: string;
  totalCount?: number;
}

// --- Trace Detail ---

export interface SpanData {
  spanId: string;
  parentSpanId: string | null;
  name: string;
  startTime: string;
  endTime: string;
  durationMs: number;
  status: 'success' | 'error' | 'unknown';
  type: string; // action, dotprompt, flow, flowStep, util, userEngagement
  subtype: string; // flow, model, tool, etc.
  path: string;
  input: string;
  output: string;
  isRoot: boolean;
  featureName: string;
  modelName: string;
  labels: Record<string, string>;
  children: SpanData[];
}

export interface TraceDetailResponse {
  traceId: string;
  spans: SpanData[];
  rootSpan: SpanData | null;
}

// --- Auth ---

export interface AuthStatusResponse {
  authenticated: boolean;
  projectId: string | null;
  error?: string;
  hint?: string;
}

// --- Health ---

export interface HealthResponse {
  status: string;
  timestamp: string;
  defaultProject: string | null;
  cacheSize: number;
  authError?: string;
}
