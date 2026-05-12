import { getAccessToken } from '../auth.js';

const TRACE_BASE = 'https://cloudtrace.googleapis.com/v1';

/** Raw span from Cloud Trace v1 API */
interface GcpSpan {
  spanId: string;
  name: string;
  startTime: string;
  endTime: string;
  parentSpanId?: string;
  labels: Record<string, string>;
  kind?: string;
}

/** Raw trace from Cloud Trace v1 API */
interface GcpTrace {
  projectId: string;
  traceId: string;
  spans: GcpSpan[];
}

interface GcpTraceListResponse {
  traces?: GcpTrace[];
  nextPageToken?: string;
}

/** Normalized span for our API */
export interface NormalizedSpan {
  spanId: string;
  parentSpanId: string | null;
  name: string;
  startTime: string;
  endTime: string;
  durationMs: number;
  status: 'success' | 'error' | 'unknown';
  type: string;
  subtype: string;
  path: string;
  input: string;
  output: string;
  isRoot: boolean;
  featureName: string;
  modelName: string;
  labels: Record<string, string>;
  children: NormalizedSpan[];
}

/**
 * List traces from Cloud Trace API with optional filtering.
 */
export async function listTraces(params: {
  projectId: string;
  startTime: string;
  endTime: string;
  filter?: string;
  pageSize?: number;
  pageToken?: string;
  orderBy?: string;
}): Promise<{ traces: GcpTrace[]; nextPageToken?: string }> {
  const token = await getAccessToken();
  const {
    projectId,
    startTime,
    endTime,
    filter,
    pageSize = 20,
    pageToken,
    orderBy,
  } = params;

  const queryParams = new URLSearchParams({
    startTime,
    endTime,
    pageSize: String(pageSize),
    // Request full trace data with root span info
    view: 'ROOTSPAN',
  });

  if (filter) {
    queryParams.set('filter', filter);
  }
  if (pageToken) {
    queryParams.set('pageToken', pageToken);
  }
  if (orderBy) {
    queryParams.set('orderBy', orderBy);
  }

  const url = `${TRACE_BASE}/projects/${projectId}/traces?${queryParams.toString()}`;
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Cloud Trace API error (${response.status}): ${error}`);
  }

  const data = (await response.json()) as GcpTraceListResponse;
  return {
    traces: data.traces || [],
    nextPageToken: data.nextPageToken,
  };
}

/**
 * Get a single trace with all its spans.
 */
export async function getTrace(
  projectId: string,
  traceId: string
): Promise<GcpTrace> {
  const token = await getAccessToken();
  const url = `${TRACE_BASE}/projects/${projectId}/traces/${traceId}`;
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Cloud Trace API error (${response.status}): ${error}`);
  }

  return (await response.json()) as GcpTrace;
}

/**
 * Compute duration in milliseconds between two ISO timestamps.
 */
function computeDurationMs(start: string, end: string): number {
  return new Date(end).getTime() - new Date(start).getTime();
}

/**
 * Extract genkit-specific label, handling both "genkit/xxx" format.
 */
function getGenkitLabel(
  labels: Record<string, string>,
  key: string
): string {
  return labels[`genkit/${key}`] || '';
}

/**
 * Normalize a raw GCP span into our format.
 */
function normalizeSpan(span: GcpSpan): NormalizedSpan {
  const labels = span.labels || {};
  const state = getGenkitLabel(labels, 'state');
  const type = getGenkitLabel(labels, 'type');
  const subtype = getGenkitLabel(labels, 'metadata/subtype');

  return {
    spanId: span.spanId,
    parentSpanId: span.parentSpanId || null,
    name: span.name || getGenkitLabel(labels, 'name') || '<unnamed>',
    startTime: span.startTime,
    endTime: span.endTime,
    durationMs: computeDurationMs(span.startTime, span.endTime),
    status:
      state === 'success'
        ? 'success'
        : state === 'error'
          ? 'error'
          : 'unknown',
    type: type || subtype || 'unknown',
    subtype: subtype || '',
    path: getGenkitLabel(labels, 'path'),
    input: getGenkitLabel(labels, 'input'),
    output: getGenkitLabel(labels, 'output'),
    isRoot: getGenkitLabel(labels, 'isRoot') === 'true',
    featureName: getGenkitLabel(labels, 'feature'),
    modelName: getGenkitLabel(labels, 'model'),
    labels,
    children: [],
  };
}

/**
 * Build a span tree from a flat list of spans.
 * Returns the root span with children populated recursively.
 */
export function buildSpanTree(spans: GcpSpan[]): NormalizedSpan | null {
  if (spans.length === 0) return null;

  const normalized = spans.map(normalizeSpan);
  const byId = new Map<string, NormalizedSpan>();
  for (const span of normalized) {
    byId.set(span.spanId, span);
  }

  let root: NormalizedSpan | null = null;

  for (const span of normalized) {
    if (!span.parentSpanId || !byId.has(span.parentSpanId)) {
      // This is a root span (or orphaned)
      if (!root || span.isRoot) {
        root = span;
      }
    } else {
      const parent = byId.get(span.parentSpanId);
      if (parent) {
        parent.children.push(span);
      }
    }
  }

  // Sort children by startTime
  const sortChildren = (span: NormalizedSpan) => {
    span.children.sort(
      (a, b) =>
        new Date(a.startTime).getTime() - new Date(b.startTime).getTime()
    );
    span.children.forEach(sortChildren);
  };

  if (root) {
    sortChildren(root);
  }

  return root;
}

/**
 * Normalize a trace for list view (summary with root span info).
 */
export function normalizeTraceForList(trace: GcpTrace): {
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
} {
  const spans = trace.spans || [];

  // Find root span (genkit/isRoot=true, or first span)
  const rootRaw =
    spans.find((s) => s.labels?.['genkit/isRoot'] === 'true') || spans[0];

  const root = rootRaw ? normalizeSpan(rootRaw) : null;

  // Extract model names from spans
  const models = new Set<string>();
  for (const span of spans) {
    const model = span.labels?.['genkit/model'];
    if (model) {
      models.add(model);
    }
  }

  return {
    traceId: trace.traceId,
    rootSpan: root
      ? {
          spanId: root.spanId,
          name: root.name,
          startTime: root.startTime,
          endTime: root.endTime,
          status: root.status,
          durationMs: root.durationMs,
          featureName: root.featureName,
          type: root.type,
          subtype: root.subtype,
          input: root.input,
          output: root.output,
        }
      : {
          spanId: '',
          name: '<unknown>',
          startTime: '',
          endTime: '',
          status: 'unknown' as const,
          durationMs: 0,
          featureName: '',
          type: '',
          subtype: '',
          input: '',
          output: '',
        },
    spanCount: spans.length,
    models: Array.from(models),
  };
}

/**
 * Flatten a span tree back to a flat list (for API response).
 */
export function flattenSpanTree(root: NormalizedSpan): NormalizedSpan[] {
  const result: NormalizedSpan[] = [];
  const visit = (span: NormalizedSpan) => {
    result.push(span);
    for (const child of span.children) {
      visit(child);
    }
  };
  visit(root);
  return result;
}
