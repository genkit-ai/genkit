/**
 * Copyright 2026 Google LLC
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

import type { SpanData, TraceData } from '@genkit-ai/tools-common';

export const RAW_BASE64_IMAGE =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

const SAMPLE_TOOL_DEFS = [
  {
    name: 'lookupOrder',
    key: '/tool/lookupOrder',
    description: 'Verbose description that should be omitted in tree view',
    inputSchema: {
      type: 'object',
      properties: { orderId: { type: 'string' } },
      required: ['orderId'],
      $schema: 'http://json-schema.org/draft-07/schema#',
    },
  },
  {
    name: 'getShippingQuote',
    key: '/tool/getShippingQuote',
    description: 'Another verbose tool schema',
    inputSchema: {
      type: 'object',
      properties: { weight: { type: 'number' } },
    },
  },
];

export function buildSpan(
  overrides: Partial<SpanData> & { spanId: string }
): SpanData {
  return {
    traceId: 'test-id',
    displayName: overrides.spanId,
    startTime: 1000,
    endTime: 2000,
    instrumentationLibrary: { name: 'genkit-tracer' },
    spanKind: 'INTERNAL',
    status: { code: 0 },
    attributes: {},
    ...overrides,
  };
}

export function buildTrace(
  spans: Record<string, SpanData>,
  overrides: Partial<TraceData> = {}
): TraceData {
  return {
    traceId: 'test-id',
    displayName: 'testSpan',
    startTime: 1000,
    endTime: 2000,
    spans,
    ...overrides,
  };
}

export function buildOrderTrace(traceId = 'trace-compact-1'): TraceData {
  const userMsg = {
    role: 'user',
    content: [{ text: 'Quote order ORD-123' }],
  };
  const modelToolCallMsg = {
    role: 'model',
    content: [
      {
        toolRequest: {
          name: 'lookupOrder',
          ref: 'call_1',
          input: { orderId: 'ORD-123' },
        },
      },
    ],
  };
  const toolRespMsg = {
    role: 'tool',
    content: [
      {
        toolResponse: {
          name: 'lookupOrder',
          ref: 'call_1',
          output: { ok: true, weight: 2500, weightUnit: 'g' },
        },
      },
    ],
  };
  const finalModelMsg = {
    role: 'model',
    content: [{ text: 'ORD-123 | $19.99' }],
  };

  return buildTrace(
    {
      flowSpan: buildSpan({
        spanId: 'flowSpan',
        traceId,
        displayName: 'orderSupportFlow',
        startTime: 1000,
        endTime: 2500,
        attributes: {
          'genkit:type': 'action',
          'genkit:metadata:subtype': 'flow',
          'genkit:state': 'success',
          'genkit:input': JSON.stringify({
            orderId: 'ORD-123',
            img: RAW_BASE64_IMAGE,
          }),
          'genkit:output': JSON.stringify('ORD-123 | $19.99'),
        },
      }),
      gen1: buildSpan({
        spanId: 'gen1',
        parentSpanId: 'flowSpan',
        traceId,
        displayName: 'generate',
        startTime: 1010,
        endTime: 2490,
        attributes: {
          'genkit:type': 'util',
          'genkit:name': 'generate',
          'genkit:state': 'success',
          'genkit:input': JSON.stringify({
            model: 'googleai/gemini-2.5-flash',
            messages: [userMsg],
            tools: ['/tool/lookupOrder', '/tool/getShippingQuote'],
            config: { temperature: 0.2 },
            use: [],
            docs: [],
          }),
          'genkit:output': JSON.stringify({
            message: finalModelMsg,
            finishReason: 'stop',
            usage: {
              inputCharacters: 42,
              inputImages: 0,
              inputTokens: 120,
              outputTokens: 15,
              totalTokens: 145,
              thoughtsTokens: 10,
              cachedContentTokens: 0,
            },
            custom: {
              model_invocation_token_counts: [{ tokens: 999 }],
              service_tier: 'standard',
            },
            request: {
              messages: [userMsg, modelToolCallMsg, toolRespMsg],
              tools: SAMPLE_TOOL_DEFS,
              config: {},
              output: {},
            },
          }),
        },
      }),
      model1: buildSpan({
        spanId: 'model1',
        parentSpanId: 'gen1',
        traceId,
        displayName: 'googleai/gemini-2.5-flash',
        startTime: 1020,
        endTime: 1500,
        attributes: {
          'genkit:type': 'action',
          'genkit:metadata:subtype': 'model',
          'genkit:state': 'success',
          'genkit:input': JSON.stringify({
            messages: [userMsg],
            config: {},
            tools: SAMPLE_TOOL_DEFS,
            output: {},
          }),
          'genkit:output': JSON.stringify({
            finishReason: 'stop',
            message: modelToolCallMsg,
            custom: { verboseProviderPayload: true },
            raw: { duplicateVerbosePayload: true },
            usage: {
              inputTokens: 80,
              outputTokens: 12,
              totalTokens: 92,
              inputImages: 0,
              cachedContentTokens: 25,
            },
          }),
        },
      }),
      tool1: buildSpan({
        spanId: 'tool1',
        parentSpanId: 'gen1',
        traceId,
        displayName: 'lookupOrder',
        startTime: 1510,
        endTime: 1550,
        attributes: {
          'genkit:type': 'action',
          'genkit:metadata:subtype': 'tool',
          'genkit:state': 'success',
          'genkit:input': JSON.stringify({ orderId: 'ORD-123' }),
          'genkit:output': JSON.stringify({
            ok: true,
            weight: 2500,
            weightUnit: 'g',
            custom: { warehouseNote: 'keep-tool-custom-fields' },
          }),
        },
      }),
      gen2: buildSpan({
        spanId: 'gen2',
        parentSpanId: 'gen1',
        traceId,
        displayName: 'generate',
        startTime: 1560,
        endTime: 2480,
        attributes: {
          'genkit:type': 'util',
          'genkit:name': 'generate',
          'genkit:state': 'success',
          'genkit:input': JSON.stringify({
            model: 'googleai/gemini-2.5-flash',
            messages: [userMsg, modelToolCallMsg, toolRespMsg],
            tools: ['/tool/lookupOrder', '/tool/getShippingQuote'],
            config: {},
            use: [],
          }),
          'genkit:output': JSON.stringify({
            message: finalModelMsg,
            finishReason: 'stop',
            usage: { inputTokens: 120, outputTokens: 15, totalTokens: 135 },
            request: { messages: [userMsg], tools: SAMPLE_TOOL_DEFS },
          }),
        },
      }),
    },
    { traceId, displayName: 'orderSupportFlow', startTime: 1000, endTime: 2500 }
  );
}
