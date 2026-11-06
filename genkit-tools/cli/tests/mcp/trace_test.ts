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

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { defineTraceTools } from '../../src/mcp/trace';
import type { McpToolOptions } from '../../src/mcp/utils';

vi.mock('@genkit-ai/tools-common/utils', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@genkit-ai/tools-common/utils')>();
  return {
    ...actual,
    record: vi.fn(() => Promise.resolve()),
  };
});

describe('MCP trace tools', () => {
  let registeredTools: Record<string, (opts: any) => Promise<any>>;
  let registeredConfigs: Record<string, any>;
  let mockServer: any;
  let mockMcpManager: any;
  let mockRuntimeManager: any;
  let options: McpToolOptions;

  const rawImage =
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

  const sampleTrace = {
    traceId: 'trace-abc',
    displayName: 'orderSupportFlow',
    startTime: 1000,
    endTime: 1800,
    spans: {
      root: {
        spanId: 'root',
        traceId: 'trace-abc',
        displayName: 'orderSupportFlow',
        startTime: 1000,
        endTime: 1800,
        instrumentationLibrary: { name: 'genkit-tracer' },
        spanKind: 'INTERNAL',
        status: { code: 0 },
        attributes: {
          'genkit:type': 'action',
          'genkit:metadata:subtype': 'flow',
          'genkit:state': 'success',
          'genkit:input': JSON.stringify({ orderId: 'ORD-1', img: rawImage }),
          'genkit:output': JSON.stringify('ORD-1 | $12.50'),
        },
      },
      model1: {
        spanId: 'model1',
        parentSpanId: 'root',
        traceId: 'trace-abc',
        displayName: 'googleai/gemini-2.5-flash',
        startTime: 1050,
        endTime: 1750,
        instrumentationLibrary: { name: 'genkit-tracer' },
        spanKind: 'INTERNAL',
        status: { code: 0 },
        attributes: {
          'genkit:type': 'action',
          'genkit:metadata:subtype': 'model',
          'genkit:state': 'success',
          'genkit:input': JSON.stringify({
            messages: [{ role: 'user', content: [{ text: 'Quote ORD-1' }] }],
            tools: [
              {
                name: 'lookupOrder',
                description: 'Look up order details by ID',
                inputSchema: { type: 'object' },
              },
            ],
            config: {},
          }),
          'genkit:output': JSON.stringify({
            finishReason: 'stop',
            message: { role: 'model', content: [{ text: 'ORD-1 | $12.50' }] },
            usage: {
              inputTokens: 50,
              outputTokens: 10,
              totalTokens: 60,
              inputImages: 0,
            },
            custom: { verboseProviderTelemetry: true },
            request: {
              messages: [{ role: 'user', content: [{ text: 'Quote ORD-1' }] }],
            },
          }),
        },
      },
    },
  };

  beforeEach(() => {
    registeredTools = {};
    registeredConfigs = {};
    mockServer = {
      registerTool: vi.fn((name: string, config: any, handler: any) => {
        registeredTools[name] = handler;
        registeredConfigs[name] = config;
      }),
    };
    mockRuntimeManager = {
      getTrace: vi.fn().mockResolvedValue(sampleTrace),
    };
    mockMcpManager = {
      getManager: vi.fn().mockResolvedValue(mockRuntimeManager),
    };
    options = {
      projectRoot: '/workspace',
      explicitProjectRoot: false,
      timeout: 5000,
      manager: mockMcpManager,
    };
    defineTraceTools(mockServer, options);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders compact execution tree by default when format is omitted', async () => {
    const res = await registeredTools['get_trace']({ traceId: 'trace-abc' });

    expect(mockRuntimeManager.getTrace).toHaveBeenCalledWith({
      traceId: 'trace-abc',
    });
    expect(res.isError).toBeUndefined();
    const text = res.content[0].text;
    expect(text).toContain('Trace ID: trace-abc');
    expect(text).toContain('Execution Tree:');
    expect(text).toContain('orderSupportFlow (flow) [800ms] ✔');
    expect(text).toContain('googleai/gemini-2.5-flash (model) [700ms] ✔');
    expect(text).toContain('tools: [lookupOrder]');
    expect(text).toContain(
      'usage: inputTokens: 50, outputTokens: 10, totalTokens: 60'
    );
    expect(text).toContain('base64 data');
    expect(text).not.toContain(rawImage);
    expect(text).not.toContain('verboseProviderTelemetry');
    expect(text).not.toContain('Look up order details by ID');
  });

  it('renders compact execution tree when format is explicitly "tree"', async () => {
    const res = await registeredTools['get_trace']({
      traceId: 'trace-abc',
      format: 'tree',
    });

    expect(res.isError).toBeUndefined();
    expect(res.content[0].text).toContain('Execution Tree:');
    expect(res.content[0].text).toContain('tools: [lookupOrder]');
  });

  it('returns sanitized JSON when format is "json"', async () => {
    const res = await registeredTools['get_trace']({
      traceId: 'trace-abc',
      format: 'json',
    });

    expect(res.isError).toBeUndefined();
    const parsed = JSON.parse(res.content[0].text);
    expect(parsed.traceId).toBe('trace-abc');
    expect(parsed.spans.root.attributes['genkit:input'].orderId).toBe('ORD-1');
    expect(parsed.spans.root.attributes['genkit:input'].img).toContain(
      'base64 data'
    );
    expect(parsed.spans.root.attributes['genkit:input'].img).not.toContain(
      rawImage
    );
    // Full uncompacted JSON fields are retained in json mode
    expect(
      parsed.spans.model1.attributes['genkit:output'].custom
        .verboseProviderTelemetry
    ).toBe(true);
    expect(
      parsed.spans.model1.attributes['genkit:input'].tools[0].description
    ).toBe('Look up order details by ID');
  });

  it('returns an error when trace is not found (undefined or null)', async () => {
    mockRuntimeManager.getTrace.mockResolvedValueOnce(undefined);
    const resUndefined = await registeredTools['get_trace']({
      traceId: 'missing-id',
    });
    expect(resUndefined).toEqual({
      isError: true,
      content: [
        {
          type: 'text',
          text: "Trace with ID 'missing-id' not found.",
        },
      ],
    });

    mockRuntimeManager.getTrace.mockResolvedValueOnce(null);
    const resNull = await registeredTools['get_trace']({
      traceId: 'null-id',
    });
    expect(resNull).toEqual({
      isError: true,
      content: [
        {
          type: 'text',
          text: "Trace with ID 'null-id' not found.",
        },
      ],
    });
  });

  it('returns formatted error when getTrace throws', async () => {
    mockRuntimeManager.getTrace.mockRejectedValueOnce(
      new Error('Telemetry server unreachable')
    );

    const res = await registeredTools['get_trace']({ traceId: 'err-id' });
    expect(res).toEqual({
      isError: true,
      content: [
        {
          type: 'text',
          text: 'Error: Telemetry server unreachable',
        },
      ],
    });
  });

  it('requires projectRoot when explicitProjectRoot is enabled', async () => {
    defineTraceTools(mockServer, {
      ...options,
      explicitProjectRoot: true,
    });

    const res = await registeredTools['get_trace']({ traceId: 'trace-abc' });
    expect(res).toEqual({
      isError: true,
      content: [
        {
          type: 'text',
          text: 'Project root is required for this tool.',
        },
      ],
    });
  });
});
