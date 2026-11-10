/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import type { NestedSpanData, SpanData, TraceData } from '../types';

/** Options for {@link stackTraceSpans}. */
export interface StackTraceSpansOptions {
  /**
   * Optional predicate to exclude spans (along with their entire descendant
   * subtrees) after root internal wrapper spans have been unwrapped.
   */
  exclude?: (span: NestedSpanData) => boolean;
}

const OTEL_SPAN_KINDS = [
  'internal',
  'server',
  'client',
  'producer',
  'consumer',
];

/**
 * Sorts spans chronologically by `startTime`, breaking ties deterministically by `spanId`.
 */
export function sortSpansChronologically(spans: SpanData[]): void {
  spans.sort(
    (a, b) => a.startTime - b.startTime || a.spanId.localeCompare(b.spanId)
  );
}

/** Checks whether a span attribute value represents boolean `true` or `'true'`. */
export function isTruthyAttr(val: unknown): boolean {
  return (
    val === true || (typeof val === 'string' && val.toLowerCase() === 'true')
  );
}

function isInternalRootWrapper(span: SpanData): boolean {
  return (
    isTruthyAttr(span.attributes?.['genkit:metadata:genkit-dev-internal']) ||
    isTruthyAttr(span.attributes?.['genkit:metadata:flow:wrapperAction'])
  );
}

function unwrapInternalRoot(
  root: NestedSpanData,
  exclude?: (span: NestedSpanData) => boolean
): NestedSpanData {
  let current = root;
  while (isInternalRootWrapper(current)) {
    const next = exclude
      ? current.spans?.find((c) => isInternalRootWrapper(c) || !exclude(c))
      : current.spans?.[0];
    if (!next) break;
    current = next;
  }
  return current;
}

function pruneSpanSubtree(
  node: NestedSpanData,
  exclude: (span: NestedSpanData) => boolean
): NestedSpanData | undefined {
  if (exclude(node)) return undefined;
  if (node.spans?.length) {
    node.spans = node.spans.filter((child) =>
      Boolean(pruneSpanSubtree(child, exclude))
    );
  }
  return node;
}

/** Transforms flat spans from the trace data into a tree of spans. */
export function stackTraceSpans(
  trace: TraceData,
  options?: StackTraceSpansOptions
): NestedSpanData | undefined {
  const flatSpans = trace.spans ? Object.values(trace.spans) : [];
  if (flatSpans.length === 0) return undefined;

  let rootSpan: NestedSpanData | undefined;
  const treeSpans = new Map<string, NestedSpanData>();

  for (const span of flatSpans) {
    const node: NestedSpanData = { ...span, spans: [] };
    treeSpans.set(span.spanId, node);
    if (!span.parentSpanId) rootSpan = node;
  }

  for (const span of treeSpans.values()) {
    if (!span.parentSpanId || span.spanId === rootSpan?.spanId) continue;
    const parent = treeSpans.get(span.parentSpanId);
    if (parent) {
      parent.spans?.push(span);
    } else if (!rootSpan) {
      rootSpan = span;
    }
  }

  if (!rootSpan) return undefined;

  for (const span of treeSpans.values()) {
    if (span.spans) sortSpansChronologically(span.spans);
  }

  const bestRoot = unwrapInternalRoot(rootSpan, options?.exclude);
  return options?.exclude
    ? pruneSpanSubtree(bestRoot, options.exclude)
    : bestRoot;
}

/** Extracts the display type/subtype of a span. */
export function getSpanType(span: SpanData): string {
  const attrs = span.attributes;
  const explicit =
    (attrs?.['genkit:metadata:subtype'] as string) ||
    (attrs?.['genkit:type'] as string);
  if (explicit) return explicit;
  if (typeof span.spanKind === 'string' && span.spanKind) {
    return span.spanKind.toLowerCase();
  }
  if (typeof span.spanKind === 'number') {
    return OTEL_SPAN_KINDS[span.spanKind] || 'span';
  }
  return 'span';
}

/** Extracts the status icon ('✔', '✖', or '') of a span. */
export function getSpanStatus(span: SpanData): string {
  const state = (
    span.attributes?.['genkit:state'] as string | undefined
  )?.toLowerCase();
  if (state === 'success') return '✔';
  if (state === 'error' || state === 'failed') return '✖';
  if (span.status?.code === 1 || span.status?.code === 0) return '✔';
  if (span.status?.code === 2) return '✖';
  return '';
}

/**
 * Safely parses a JSON string attribute from a span, or returns the value as-is if already parsed.
 */
export function parseSpanJson<T = unknown>(val: unknown): T | undefined {
  if (val === undefined || val === null) return undefined;
  if (typeof val === 'string') {
    const trimmed = val.trim();
    if (!trimmed) return undefined;
    try {
      return JSON.parse(trimmed) as T;
    } catch {
      return val as unknown as T;
    }
  }
  return val as T;
}

/** Returns a span's `displayName` or `genkit:name` attribute. */
export function getSpanName(span: SpanData): string {
  return (
    span.displayName ||
    (span.attributes?.['genkit:name'] as string | undefined) ||
    ''
  );
}

function hasSpanTypeOrSubtype(span: SpanData, target: string): boolean {
  const attrs = span.attributes;
  return (
    attrs?.['genkit:metadata:subtype'] === target ||
    attrs?.['genkit:type'] === target
  );
}

/** Checks whether a span represents an agent boundary. */
export function isAgentSpan(span: SpanData): boolean {
  return (
    hasSpanTypeOrSubtype(span, 'agent') ||
    getSpanName(span).startsWith('/agent/')
  );
}

/** Checks whether a span represents a model execution. */
export function isModelSpan(span: SpanData): boolean {
  return (
    hasSpanTypeOrSubtype(span, 'model') ||
    getSpanName(span).startsWith('/model/')
  );
}

/** Checks whether a span represents a tool execution. */
export function isToolSpan(span: SpanData): boolean {
  const name = getSpanName(span);
  return (
    hasSpanTypeOrSubtype(span, 'tool') ||
    hasSpanTypeOrSubtype(span, 'tool.v2') ||
    name.startsWith('/tool/') ||
    name.startsWith('/tool.v2/')
  );
}

/**
 * Strips Genkit's leading action-type prefix (e.g. `/agent/`, `/model/`, `/tool/`) from a span/action name.
 */
export function cleanActionName(rawName?: string): string {
  if (!rawName) return '';
  return rawName.replace(
    /^\/(agent|flow|tool\.v2|tool|prompt|model|background-model|util)\//,
    ''
  );
}

/**
 * Extracts the short tool name, stripping any leading action prefix and slash-delimited namespace.
 */
export function shortToolName(rawName?: string): string {
  if (!rawName) return '';
  const cleaned = cleanActionName(rawName);
  const lastSlash = cleaned.lastIndexOf('/');
  return lastSlash >= 0 ? cleaned.substring(lastSlash + 1) : cleaned;
}

function getSpanSessionId(span: SpanData): string | undefined {
  const sid =
    span.attributes?.['genkit:metadata:agent:sessionId'] ??
    span.attributes?.['session.id'];
  return typeof sid === 'string' && sid.length > 0 ? sid : undefined;
}

function getSpanDepth(
  span: SpanData,
  spans: Record<string, SpanData>,
  maxDepth: number
): number {
  let depth = 0;
  for (
    let p = span.parentSpanId;
    p && spans[p] && depth < maxDepth;
    p = spans[p].parentSpanId
  ) {
    depth++;
  }
  return depth;
}

/**
 * Discovers the outer `sessionId` from span attributes in a trace.
 * Prefers shallower spans so root/outer agents win over nested subagents.
 */
export function extractSessionIdFromTrace(
  trace?: TraceData
): string | undefined {
  if (!trace?.spans) return undefined;
  const spans = trace.spans;
  const allSpans = Object.values(spans);
  const candidates = allSpans.flatMap((span) => {
    const sid = getSpanSessionId(span);
    return sid
      ? [{ sid, depth: getSpanDepth(span, spans, allSpans.length), span }]
      : [];
  });
  candidates.sort(
    (a, b) =>
      a.depth - b.depth ||
      (a.span.startTime ?? 0) - (b.span.startTime ?? 0) ||
      a.span.spanId.localeCompare(b.span.spanId)
  );
  return candidates[0]?.sid;
}
