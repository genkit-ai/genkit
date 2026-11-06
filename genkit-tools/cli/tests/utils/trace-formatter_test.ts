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

import type { TraceData } from '@genkit-ai/tools-common';
import { describe, expect, it } from 'vitest';
import {
  cleanTraceJson,
  formatTraceTree,
} from '../../src/utils/trace-formatter';

describe('trace-formatter', () => {
  describe('cleanTraceJson', () => {
    it('should deep clone and sanitize base64 media data correctly', () => {
      const rawImage =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
      const originalTrace = {
        traceId: 'test-id',
        displayName: 'testSpan',
        spans: {
          s1: {
            spanId: 's1',
            attributes: {
              'genkit:input': JSON.stringify({ img: rawImage }),
              imgRaw: rawImage,
            },
          },
        },
      } as unknown as TraceData;

      // Without keepBase64 = false (default)
      const sanitized = cleanTraceJson(originalTrace, false);

      expect(sanitized.traceId).toBe('test-id');
      expect(sanitized).not.toBe(originalTrace); // Must be a deep clone

      const attrs = sanitized.spans!.s1.attributes!;

      // Sanitized JSON parsing of base64
      expect((attrs['genkit:input'] as any).img).toContain('base64 data');
      expect((attrs['genkit:input'] as any).img).not.toContain(rawImage);

      // Sanitized raw base64 string
      expect(attrs.imgRaw).toContain('base64 data');
      expect(attrs.imgRaw).not.toContain(rawImage);

      // Verify the type shape remains intact
      expect(typeof sanitized.traceId).toBe('string');
      expect(typeof sanitized.spans).toBe('object');
    });

    it('should deep clone and preserve base64 media data if keepBase64 is true', () => {
      const rawImage =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
      const originalTrace = {
        traceId: 'test-id',
        spans: {
          s1: {
            spanId: 's1',
            attributes: {
              'genkit:input': JSON.stringify({ img: rawImage }),
            },
          },
        },
      } as unknown as TraceData;

      const preserved = cleanTraceJson(originalTrace, true);

      expect(preserved.traceId).toBe('test-id');
      expect((preserved.spans!.s1.attributes!['genkit:input'] as any).img).toBe(
        rawImage
      );
    });
  });

  describe('formatTraceTree', () => {
    it('should print compact formatted payloads by default in tree format', () => {
      const trace = {
        traceId: 'test-id',
        displayName: 'testSpan',
        startTime: 1000,
        endTime: 2000,
        spans: {
          s1: {
            spanId: 's1',
            displayName: 'testSpan',
            startTime: 1000,
            endTime: 2000,
            attributes: {
              'genkit:input': JSON.stringify({
                query: 'Hello Genkit',
                limit: 10,
              }),
            },
          },
        },
      };
      const output = formatTraceTree(trace as any);
      expect(output).toContain('Hello Genkit');
    });

    it('should format arrays without parts nicely as bullet points', () => {
      const trace = {
        traceId: 'test-id',
        displayName: 'testSpan',
        startTime: 1000,
        endTime: 2000,
        spans: {
          s1: {
            spanId: 's1',
            displayName: 'testSpan',
            startTime: 1000,
            endTime: 2000,
            attributes: {
              'genkit:input': JSON.stringify([{ a: 1 }, { b: 2 }]),
            },
          },
        },
      };
      const output = formatTraceTree(trace as any);
      expect(output).toMatch(/- a: 1\s*- b: 2/s);
    });

    it('should cleanly fallback an empty message content to a generic Role placeholder', () => {
      const trace = {
        traceId: 'test-id',
        displayName: 'testSpan',
        startTime: 1000,
        endTime: 2000,
        spans: {
          s1: {
            spanId: 's1',
            displayName: 'testSpan',
            startTime: 1000,
            endTime: 2000,
            attributes: {
              'genkit:input': JSON.stringify({ role: 'user', content: [] }),
            },
          },
        },
      };
      const output = formatTraceTree(trace as any);
      expect(output).toContain('User: (empty)');
    });

    it('should compact generate and model spans while preserving full flow and tool payloads', () => {
      const toolDefs = [
        {
          name: 'lookupOrder',
          key: '/tool/lookupOrder',
          description:
            'Verbose description that should be omitted in tree view',
          inputSchema: {
            type: 'object',
            properties: { orderId: { type: 'string' } },
            required: ['orderId'],
            $schema: 'http://json-schema.org/draft-07/schema#',
          },
          outputSchema: {
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

      const trace: TraceData = {
        traceId: 'trace-compact-1',
        displayName: 'orderSupportFlow',
        startTime: 1000,
        endTime: 2500,
        spans: {
          flowSpan: {
            spanId: 'flowSpan',
            traceId: 'trace-compact-1',
            displayName: 'orderSupportFlow',
            startTime: 1000,
            endTime: 2500,
            instrumentationLibrary: { name: 'genkit-tracer' },
            spanKind: 'INTERNAL',
            status: { code: 0 },
            attributes: {
              'genkit:type': 'action',
              'genkit:metadata:subtype': 'flow',
              'genkit:input': JSON.stringify('ORD-123'),
              'genkit:output': JSON.stringify('ORD-123 | $19.99'),
              'genkit:state': 'success',
            },
          },
          gen1: {
            spanId: 'gen1',
            parentSpanId: 'flowSpan',
            traceId: 'trace-compact-1',
            displayName: 'generate',
            startTime: 1010,
            endTime: 2490,
            instrumentationLibrary: { name: 'genkit-tracer' },
            spanKind: 'INTERNAL',
            status: { code: 0 },
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
                  inputVideos: 0,
                  inputAudioFiles: 0,
                  outputCharacters: 16,
                  outputImages: 0,
                  outputVideos: 0,
                  outputAudioFiles: 0,
                  inputTokens: 120,
                  outputTokens: 15,
                  totalTokens: 145,
                  thoughtsTokens: 10,
                  cachedContentTokens: 0,
                  custom: { inputTextTokens: 120 },
                },
                custom: {
                  model_invocation_token_counts: [{ tokens: 999 }],
                  service_tier: 'standard',
                },
                request: {
                  messages: [userMsg, modelToolCallMsg, toolRespMsg],
                  tools: toolDefs,
                  config: {},
                  output: {},
                },
              }),
            },
          },
          model1: {
            spanId: 'model1',
            parentSpanId: 'gen1',
            traceId: 'trace-compact-1',
            displayName: 'googleai/gemini-2.5-flash',
            startTime: 1020,
            endTime: 1500,
            instrumentationLibrary: { name: 'genkit-tracer' },
            spanKind: 'INTERNAL',
            status: { code: 0 },
            attributes: {
              'genkit:type': 'action',
              'genkit:metadata:subtype': 'model',
              'genkit:state': 'success',
              'genkit:input': JSON.stringify({
                messages: [userMsg],
                config: {},
                tools: toolDefs,
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
          },
          tool1: {
            spanId: 'tool1',
            parentSpanId: 'gen1',
            traceId: 'trace-compact-1',
            displayName: 'lookupOrder',
            startTime: 1510,
            endTime: 1550,
            instrumentationLibrary: { name: 'genkit-tracer' },
            spanKind: 'INTERNAL',
            status: { code: 0 },
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
          },
          gen2: {
            spanId: 'gen2',
            parentSpanId: 'gen1',
            traceId: 'trace-compact-1',
            displayName: 'generate',
            startTime: 1560,
            endTime: 2480,
            instrumentationLibrary: { name: 'genkit-tracer' },
            spanKind: 'INTERNAL',
            status: { code: 0 },
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
                usage: {
                  inputTokens: 120,
                  outputTokens: 15,
                  totalTokens: 135,
                },
                request: { messages: [userMsg], tools: toolDefs },
              }),
            },
          },
        },
      };

      const cleaned = cleanTraceJson(trace, false);
      const output = formatTraceTree(cleaned);

      // Clean JSON must still preserve full uncompacted schemas, request, and custom fields
      expect(
        (cleaned.spans.gen1.attributes['genkit:output'] as any).request.tools
      ).toHaveLength(2);
      expect(
        (cleaned.spans.model1.attributes['genkit:input'] as any).tools[0]
          .description
      ).toContain('Verbose description');

      // Tree view summarizes tool names and omits full JSON schemas / echoed request / custom / raw
      expect(output).toContain('tools: [lookupOrder, getShippingQuote]');
      expect(output).not.toContain('Verbose description');
      expect(output).not.toContain('http://json-schema.org/draft-07/schema#');
      expect(output).not.toContain('model_invocation_token_counts');
      expect(output).not.toContain('verboseProviderPayload');
      expect(output).not.toContain('duplicateVerbosePayload');
      expect(output).not.toContain('config: {}');
      expect(output).not.toContain('output: {}');
      expect(output).not.toContain('use: []');
      expect(output).not.toContain('inputImages');

      // Non-empty config is preserved
      expect(output).toContain('temperature: 0.2');

      // Usage is condensed to non-zero token counters (including thoughtsTokens & cachedContentTokens when > 0)
      expect(output).toContain(
        'usage: inputTokens: 120, outputTokens: 15, totalTokens: 145, thoughtsTokens: 10'
      );
      expect(output).toContain(
        'usage: inputTokens: 80, outputTokens: 12, totalTokens: 92, cachedContentTokens: 25'
      );

      // Cumulative message prefixes are deduplicated while new turns remain visible
      expect(output).toContain('User: "Quote order ORD-123"');
      expect(output).toContain('messages: [same 1 message as parent]');
      expect(output).toContain('... (1 earlier message)');
      expect(output).toContain('Tool Call: lookupOrder({"orderId":"ORD-123"})');

      // Flow and tool spans retain 100% of their inputs and outputs (including custom fields on tool output)
      expect(output).toContain('orderSupportFlow (flow) [1500ms] ✔');
      expect(output).toContain('lookupOrder (tool) [40ms] ✔');
      expect(output).toContain('weight: 2500');
      expect(output).toContain('weightUnit: g');
      expect(output).toContain('warehouseNote: keep-tool-custom-fields');
    });

    it('should preserve provider custom details when model output has no message or candidates', () => {
      const trace = {
        traceId: 'blocked-trace',
        spans: {
          m1: {
            spanId: 'm1',
            displayName: 'googleai/gemini-2.5-flash',
            startTime: 100,
            endTime: 200,
            attributes: {
              'genkit:type': 'action',
              'genkit:metadata:subtype': 'model',
              'genkit:output': JSON.stringify({
                finishReason: 'blocked',
                finishMessage: 'Safety filter triggered',
                custom: { safetyCategory: 'HARM_CATEGORY_DANGEROUS' },
              }),
            },
          },
        },
      } as unknown as TraceData;

      const output = formatTraceTree(trace);
      expect(output).toContain('finishReason: blocked');
      expect(output).toContain('Safety filter triggered');
      expect(output).toContain('safetyCategory: HARM_CATEGORY_DANGEROUS');
    });

    it('should render span status error messages on failed spans', () => {
      const trace = {
        traceId: 'err-trace',
        spans: {
          t1: {
            spanId: 't1',
            displayName: 'failingTool',
            startTime: 100,
            endTime: 150,
            status: {
              code: 2,
              message: 'Invalid weight: must be in kilograms',
            },
            attributes: {
              'genkit:type': 'action',
              'genkit:metadata:subtype': 'tool',
              'genkit:state': 'error',
              'genkit:input': JSON.stringify({ weight: 2500 }),
            },
          },
        },
      } as unknown as TraceData;

      const output = formatTraceTree(trace);
      expect(output).toContain('failingTool (tool) [50ms] ✖');
      expect(output).toContain('Error: Invalid weight: must be in kilograms');
    });
  });
});
