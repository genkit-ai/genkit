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

import type { NestedSpanData, SpanData, TraceData } from '../types/index.js';

/** Options for {@link stackTraceSpans}. */
export interface StackTraceSpansOptions {
  /**
   * Optional predicate to exclude spans (along with their entire descendant
   * subtrees) when building the span tree.
   */
  exclude?: (span: NestedSpanData) => boolean;
}

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
  if (!trace.spans || Object.keys(trace.spans).length === 0) return undefined;
  let rootSpan: NestedSpanData | undefined = undefined;

  const treeSpans: Map<string, NestedSpanData> = new Map();
  Object.values(trace.spans).forEach((span) => {
    treeSpans.set(span.spanId, { spans: [], ...span } as NestedSpanData);
    if (!span.parentSpanId) {
      rootSpan = treeSpans.get(span.spanId);
    }
  });

  // Build the tree of spans.
  treeSpans.forEach((span) => {
    if (span.parentSpanId && span.spanId !== rootSpan?.spanId) {
      const parent = treeSpans.get(span.parentSpanId);
      if (parent) {
        parent.spans?.push(span);
      } else if (!rootSpan) {
        // This shouldn't happen, but there is a parentSpanId that we cannot
        // find. In this case, fallback to this span being the root so the UI
        // can still render properly.
        rootSpan = span;
      }
    }
  });

  // Sort children by start times for each span.
  treeSpans.forEach((span) => {
    if (span.spans) sortSpansChronologically(span.spans);
  });

  return rootSpan && options?.exclude
    ? pruneSpanSubtree(rootSpan, options.exclude)
    : rootSpan;
}

/** Extracts the display type/subtype of a span. */
export function getSpanType(span: SpanData): string {
  const attrs = span.attributes || {};
  const kind = span.spanKind;
  let kindStr = 'span';
  if (typeof kind === 'string') {
    kindStr = kind.toLowerCase();
  } else if (typeof kind === 'number') {
    const kinds = ['internal', 'server', 'client', 'producer', 'consumer'];
    kindStr = kinds[kind] || 'span';
  }
  return (
    (attrs['genkit:metadata:subtype'] as string) ||
    (attrs['genkit:type'] as string) ||
    kindStr
  );
}

/** Extracts the status icon ('✔', '✖', or '') of a span. */
export function getSpanStatus(span: SpanData): string {
  const attrs = span.attributes || {};
  const state = attrs['genkit:state'] as string;
  if (state) {
    if (state.toLowerCase() === 'success') return '✔';
    if (state.toLowerCase() === 'error' || state.toLowerCase() === 'failed')
      return '✖';
  }
  if (span.status?.code === 1 || span.status?.code === 0) return '✔';
  if (span.status?.code === 2) return '✖';
  return '';
}

/**
 * Safely parses a JSON string attribute from a span, returning the value as-is
 * if already parsed, or the raw `string` when `JSON.parse` fails (e.g. plain-text
 * tool inputs/outputs). Callers expecting an object should narrow `typeof parsed === 'object'`.
 */
export function parseSpanJson<T = unknown>(
  val: unknown
): T | string | undefined {
  if (val === undefined || val === null) return undefined;
  if (typeof val === 'string') {
    const trimmed = val.trim();
    if (!trimmed) return undefined;
    try {
      return JSON.parse(trimmed) as T;
    } catch {
      return val;
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
  const name = getSpanName(span);
  return (
    hasSpanTypeOrSubtype(span, 'model') ||
    hasSpanTypeOrSubtype(span, 'background-model') ||
    name.startsWith('/model/') ||
    name.startsWith('/background-model/')
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

function getSpanDepth(span: SpanData, spans: Record<string, SpanData>): number {
  let depth = 0;
  const visited = new Set<string>();
  for (
    let p = span.parentSpanId;
    p && spans[p] && !visited.has(p);
    p = spans[p].parentSpanId
  ) {
    visited.add(p);
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
  const candidates = Object.values(spans).flatMap((span) => {
    const sid = getSpanSessionId(span);
    return sid ? [{ sid, depth: getSpanDepth(span, spans), span }] : [];
  });
  candidates.sort(
    (a, b) =>
      a.depth - b.depth ||
      (a.span.startTime ?? 0) - (b.span.startTime ?? 0) ||
      a.span.spanId.localeCompare(b.span.spanId)
  );
  return candidates[0]?.sid;
}
