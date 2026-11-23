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

import type { SpanData, TraceData } from '../../src/types/trace.js';

export const TRACE_ID = '7c273c22b219d077c6731a10d46b7d40';
export const BASE_FLOW_SPAN_ID = '22b219d077c67';

/** Helper class to manage traces in tests */
export class MockTrace {
  private BASE_FLOW_SPAN: SpanData = {
    spanId: BASE_FLOW_SPAN_ID,
    traceId: TRACE_ID,
    startTime: 1714059149481,
    endTime: 1714059149484.9653,
    attributes: {
      'genkit:type': 'flow',
      'genkit:name': 'multiSteps',
      'genkit:isRoot': true,
      'genkit:path': '/multiSteps',
      'genkit:metadata:flow:name': 'multiSteps',
      'genkit:input': '"Douglas Adams"',
      'genkit:output': '42',
      'genkit:state': 'success',
    },
    displayName: 'multiSteps',
    links: [],
    instrumentationLibrary: {
      name: 'genkit-tracer',
      version: 'v1',
    },
    spanKind: 'INTERNAL',
    sameProcessAsParentSpan: {
      value: true,
    },
    status: {
      code: 0,
    },
    timeEvents: {
      timeEvent: [],
    },
  };

  private BASE_SPAN: SpanData = {
    spanId: 'dfe3b782071ea308',
    traceId: '7c273c22b219d077c6731a10d46b7d40',
    parentSpanId: BASE_FLOW_SPAN_ID,
    startTime: 1714059149481,
    endTime: 1714059149481.1533,
    attributes: {},
    displayName: 'baseStep',
    links: [],
    instrumentationLibrary: {
      name: 'genkit-tracer',
      version: 'v1',
    },
    spanKind: 'INTERNAL',
    sameProcessAsParentSpan: {
      value: true,
    },
    status: {
      code: 0,
    },
    timeEvents: {
      timeEvent: [],
    },
  };

  private BASE_MOCK_TRACE: TraceData = {
    traceId: '7c273c22b219d077c6731a10d46b7d40',
    displayName: 'multiSteps',
    startTime: 1714059149480,
    endTime: 1714059149486.6926,
    spans: {},
  };

  private trace: TraceData;

  getTrace(): TraceData {
    return this.trace;
  }

  addSpan(params: {
    stepName: string;
    spanType: 'flowStep' | 'action';
    retrieverConfig?: {
      text: string[];
      query: string;
    };
    input?: any;
    output?: any;
  }): MockTrace {
    const { stepName, spanType, retrieverConfig, input, output } = {
      ...params,
    };
    const flowInput = input ?? 'Douglas Adams';
    const flowOutput = output ?? 42;

    if (spanType === 'flowStep') {
      const flowStep = {
        ...this.BASE_SPAN,
        spanId: `flowStep-${stepName}`,
        displayName: stepName,
        attributes: {
          'genkit:type': 'flowStep',
          'genkit:name': stepName,
          'genkit:path': `/multiSteps/${stepName}`,
          'genkit:metadata:flow:stepType': 'run',
          'genkit:metadata:flow:stepName': stepName,
          'genkit:metadata:flow:resolvedStepName': stepName,
          'genkit:metadata:flow:state': 'run',
          'genkit:input': JSON.stringify(flowInput),
          'genkit:output': JSON.stringify(flowOutput),
          'genkit:state': 'success',
        },
      };
      this.trace.spans[flowStep.spanId] = flowStep;
    } else {
      // spanType === "action"
      const actionStep = {
        ...this.BASE_SPAN,
        spanId: `action-${stepName}`,
        displayName: stepName,
        attributes: {
          'genkit:type': 'action',
          'genkit:name': `${stepName}`,
          'genkit:path': `/multiSteps/${stepName}`,
          'genkit:input': JSON.stringify(flowInput),
          'genkit:output': JSON.stringify(flowOutput),
          'genkit:state': 'success',
        } as Record<string, unknown>,
      };
      if (!!retrieverConfig) {
        const retrieverInput = {
          query: { content: [{ text: retrieverConfig.query }] },
          options: { k: 3 },
        };
        const retrieverResponse = {
          documents: retrieverConfig.text.map((t) => {
            return { content: [{ text: t }] };
          }),
        };
        actionStep.attributes['genkit:metadata:subtype'] = 'retriever';
        actionStep.attributes['genkit:input'] = JSON.stringify(retrieverInput);
        actionStep.attributes['genkit:output'] =
          JSON.stringify(retrieverResponse);
      }
      this.trace.spans[actionStep.spanId] = actionStep;
    }
    return this;
  }

  constructor(
    traceInput?: any,
    traceOutput?: any,
    baseFlowState: 'done' | 'error' = 'done'
  ) {
    const flowInput = traceInput ?? 'Douglas Adams';
    const flowOutput = traceOutput ?? 42;
    const baseFlowSpan = {
      ...this.BASE_FLOW_SPAN,
      status: {
        code: baseFlowState === 'error' ? 2 : 0,
      },
      attributes: {
        ...this.BASE_FLOW_SPAN.attributes,
        'genkit:input': JSON.stringify(flowInput),
        'genkit:output': JSON.stringify(flowOutput),
        'genkit:state': baseFlowState === 'error' ? 'error' : 'success',
      },
    };

    this.trace = {
      ...this.BASE_MOCK_TRACE,
      spans: { [baseFlowSpan.spanId]: baseFlowSpan },
    };

    return this;
  }
}

/** Builds a minimal valid SpanData for unit tests with optional field overrides. */
export function makeSpan(overrides: Partial<SpanData> = {}): SpanData {
  const spanId = overrides.spanId ?? 'span-1';
  return {
    traceId: 'trace-1',
    spanId,
    startTime: 100,
    endTime: 200,
    attributes: {},
    instrumentationLibrary: { name: 'genkit-tracer', version: 'v1' },
    spanKind: 'INTERNAL',
    status: { code: 0 },
    ...overrides,
    displayName: overrides.displayName ?? spanId,
  };
}

/**
 * Builds a TraceData record auto-keyed by each span's `spanId`.
 * Infers `startTime` and `endTime` bounds from `spans` when not explicitly overridden.
 */
export function makeTrace(
  spans: SpanData[],
  overrides: Partial<TraceData> = {}
): TraceData {
  const traceId = overrides.traceId ?? spans[0]?.traceId ?? 'trace-1';
  const keyedSpans: Record<string, SpanData> = {};
  let minStart = Number.POSITIVE_INFINITY;
  let maxEnd = 0;

  for (const span of spans) {
    keyedSpans[span.spanId] = overrides.traceId
      ? { ...span, traceId }
      : { ...span };
    if (span.startTime < minStart) minStart = span.startTime;
    if (span.endTime > maxEnd) maxEnd = span.endTime;
  }

  return {
    traceId,
    ...(spans.length > 0
      ? {
          startTime: Number.isFinite(minStart) ? minStart : 0,
          endTime: maxEnd,
        }
      : {}),
    ...overrides,
    spans: overrides.spans ? { ...keyedSpans, ...overrides.spans } : keyedSpans,
  };
}
