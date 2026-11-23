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
} from '../../src/utils/trace-formatter.js';
import {
  buildOrderTrace,
  buildSpan,
  buildTrace,
} from '../testing/mock_data.js';

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

    it('preserves full uncompacted tool schemas, echoed requests, and custom fields', () => {
      const cleaned = cleanTraceJson(buildOrderTrace(), false);

      expect(cleaned).toMatchObject({
        spans: {
          gen1: {
            attributes: {
              'genkit:output': {
                request: {
                  tools: [
                    { name: 'lookupOrder' },
                    { name: 'getShippingQuote' },
                  ],
                },
                custom: { service_tier: 'standard' },
              },
            },
          },
          model1: {
            attributes: {
              'genkit:input': {
                tools: [
                  {
                    description: expect.stringContaining('Verbose description'),
                  },
                  { name: 'getShippingQuote' },
                ],
              },
            },
          },
        },
      });
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

    it('summarizes generate and model tool definitions and omits empty noise fields', () => {
      const output = formatTraceTree(cleanTraceJson(buildOrderTrace(), false));

      expect(output).toContain('tools: [lookupOrder, getShippingQuote]');
      expect(output).toContain('temperature: 0.2');
      expect(output).not.toContain('Verbose description');
      expect(output).not.toContain('http://json-schema.org/draft-07/schema#');
      expect(output).not.toContain('config: {}');
      expect(output).not.toContain('output: {}');
      expect(output).not.toContain('use: []');
    });

    it('omits echoed request, custom, and raw blobs from model and generate outputs', () => {
      const output = formatTraceTree(cleanTraceJson(buildOrderTrace(), false));

      expect(output).not.toContain('model_invocation_token_counts');
      expect(output).not.toContain('verboseProviderPayload');
      expect(output).not.toContain('duplicateVerbosePayload');
    });

    it('condenses usage to non-zero token counters', () => {
      const output = formatTraceTree(cleanTraceJson(buildOrderTrace(), false));

      expect(output).toContain(
        'usage: inputTokens: 120, outputTokens: 15, totalTokens: 145, thoughtsTokens: 10'
      );
      expect(output).toContain(
        'usage: inputTokens: 80, outputTokens: 12, totalTokens: 92, cachedContentTokens: 25'
      );
      expect(output).not.toContain('inputImages');
    });

    it('deduplicates cumulative message prefixes across nested generate and model spans', () => {
      const output = formatTraceTree(cleanTraceJson(buildOrderTrace(), false));

      expect(output).toContain('User: "Quote order ORD-123"');
      expect(output).toContain('messages: [same 1 message as parent]');
      expect(output).toContain('... (1 earlier message)');
      expect(output).toContain('Tool Call: lookupOrder({"orderId":"ORD-123"})');
    });

    it('preserves full payloads including custom fields on flow and tool spans', () => {
      const output = formatTraceTree(cleanTraceJson(buildOrderTrace(), false));

      expect(output).toContain('orderSupportFlow (flow) [1500ms] ✔');
      expect(output).toContain('lookupOrder (tool) [40ms] ✔');
      expect(output).toContain('weight: 2500');
      expect(output).toContain('weightUnit: g');
      expect(output).toContain('warehouseNote: keep-tool-custom-fields');
    });

    it('preserves parallel toolRequest and toolResponse parts in a single message', () => {
      const trace = buildTrace({
        m1: buildSpan({
          spanId: 'm1',
          displayName: 'googleai/gemini-2.5-flash',
          attributes: {
            'genkit:type': 'action',
            'genkit:metadata:subtype': 'model',
            'genkit:input': JSON.stringify({
              messages: [
                {
                  role: 'tool',
                  content: [
                    {
                      toolResponse: {
                        name: 'lookupOrder',
                        output: { ok: true },
                      },
                    },
                    {
                      toolResponse: {
                        name: 'getShippingQuote',
                        output: { price: 12.5 },
                      },
                    },
                  ],
                },
              ],
            }),
            'genkit:output': JSON.stringify({
              finishReason: 'stop',
              message: {
                role: 'model',
                content: [
                  {
                    toolRequest: {
                      name: 'lookupOrder',
                      input: { orderId: 'ORD-1' },
                    },
                  },
                  {
                    toolRequest: {
                      name: 'getShippingQuote',
                      input: { weight: 500 },
                    },
                  },
                ],
              },
            }),
          },
        }),
      });

      const output = formatTraceTree(trace);

      expect(output).toContain('Tool Output [lookupOrder]: {"ok":true}');
      expect(output).toContain(
        'Tool Output [getShippingQuote]: {"price":12.5}'
      );
      expect(output).toContain('Tool Call: lookupOrder({"orderId":"ORD-1"})');
      expect(output).toContain('Tool Call: getShippingQuote({"weight":500})');
    });

    it('preserves provider custom details when model output has no message or candidates', () => {
      const trace = buildTrace(
        {
          m1: buildSpan({
            spanId: 'm1',
            traceId: 'blocked-trace',
            displayName: 'googleai/gemini-2.5-flash',
            attributes: {
              'genkit:type': 'action',
              'genkit:metadata:subtype': 'model',
              'genkit:output': JSON.stringify({
                finishReason: 'blocked',
                finishMessage: 'Safety filter triggered',
                custom: { safetyCategory: 'HARM_CATEGORY_DANGEROUS' },
              }),
            },
          }),
        },
        { traceId: 'blocked-trace' }
      );

      const output = formatTraceTree(trace);

      expect(output).toContain('finishReason: blocked');
      expect(output).toContain('Safety filter triggered');
      expect(output).toContain('safetyCategory: HARM_CATEGORY_DANGEROUS');
    });

    it('renders span status error messages on failed spans', () => {
      const trace = buildTrace(
        {
          t1: buildSpan({
            spanId: 't1',
            traceId: 'err-trace',
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
          }),
        },
        { traceId: 'err-trace' }
      );

      const output = formatTraceTree(trace);

      expect(output).toContain('failingTool (tool) [50ms] ✖');
      expect(output).toContain('Error: Invalid weight: must be in kilograms');
    });
  });
});
