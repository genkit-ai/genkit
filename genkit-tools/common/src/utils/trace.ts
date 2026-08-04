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

import type { NestedSpanData, TraceData } from '../types';

/** Transforms flat spans from the trace data into a tree of spans. */
export function stackTraceSpans(trace: TraceData): NestedSpanData | undefined {
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
    span.spans?.sort((a, b) => a.startTime - b.startTime);
  });

  // Re-position the root node to the first non-internal span
  let bestRoot = rootSpan!;
  while (
    bestRoot.attributes['genkit:metadata:genkit-dev-internal'] ||
    bestRoot.attributes['genkit:metadata:flow:wrapperAction']
  ) {
    if (!bestRoot.spans?.length) {
      break;
    }
    bestRoot = bestRoot.spans[0];
  }

  return bestRoot;
}

/** Extracts the display type/subtype of a span. */
export function getSpanType(span: NestedSpanData): string {
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
export function getSpanStatus(span: NestedSpanData): string {
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
