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
import { defineTraceTools } from '../../src/mcp/trace.js';
import type { McpToolOptions } from '../../src/mcp/utils.js';
import { RAW_BASE64_IMAGE, buildOrderTrace } from '../testing/mock_data.js';

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
  let mockServer: any;
  let mockMcpManager: any;
  let mockRuntimeManager: any;
  let options: McpToolOptions;

  beforeEach(() => {
    registeredTools = {};
    mockServer = {
      registerTool: vi.fn((name: string, _config: any, handler: any) => {
        registeredTools[name] = handler;
      }),
    };
    mockRuntimeManager = {
      getTrace: vi.fn().mockResolvedValue(buildOrderTrace('trace-abc')),
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

  it.each([
    { name: 'format is omitted', opts: { traceId: 'trace-abc' } },
    {
      name: 'format is explicitly "tree"',
      opts: { traceId: 'trace-abc', format: 'tree' },
    },
  ])('renders compact execution tree when $name', async ({ opts }) => {
    const res = await registeredTools['get_trace'](opts);

    expect(mockRuntimeManager.getTrace).toHaveBeenCalledWith({
      traceId: 'trace-abc',
    });
    expect(res.isError).toBeUndefined();
    const text = res.content[0].text;
    expect(text).toContain('Trace ID: trace-abc');
    expect(text).toContain('Execution Tree:');
    expect(text).toContain('orderSupportFlow (flow) [1500ms] ✔');
    expect(text).toContain('googleai/gemini-2.5-flash (model) [480ms] ✔');
    expect(text).toContain('tools: [lookupOrder, getShippingQuote]');
    expect(text).toContain(
      'usage: inputTokens: 80, outputTokens: 12, totalTokens: 92, cachedContentTokens: 25'
    );
    expect(text).toContain('base64 data');
    expect(text).not.toContain(RAW_BASE64_IMAGE);
    expect(text).not.toContain('verboseProviderPayload');
    expect(text).not.toContain('Verbose description');
  });

  it('returns sanitized uncompacted JSON when format is "json"', async () => {
    const res = await registeredTools['get_trace']({
      traceId: 'trace-abc',
      format: 'json',
    });

    expect(res.isError).toBeUndefined();
    const parsed = JSON.parse(res.content[0].text);
    expect(parsed).toMatchObject({
      traceId: 'trace-abc',
      spans: {
        flowSpan: {
          attributes: {
            'genkit:input': {
              orderId: 'ORD-123',
              img: expect.stringContaining('base64 data'),
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
            'genkit:output': {
              custom: { verboseProviderPayload: true },
            },
          },
        },
      },
    });
    expect(parsed.spans.flowSpan.attributes['genkit:input'].img).not.toContain(
      RAW_BASE64_IMAGE
    );
  });

  it.each([
    { name: 'undefined', traceResult: undefined, traceId: 'missing-id' },
    { name: 'null', traceResult: null, traceId: 'null-id' },
  ])(
    'returns an error when getTrace returns $name',
    async ({ traceResult, traceId }) => {
      mockRuntimeManager.getTrace.mockResolvedValueOnce(traceResult);

      const res = await registeredTools['get_trace']({ traceId });

      expect(res).toEqual({
        isError: true,
        content: [
          {
            type: 'text',
            text: `Trace with ID '${traceId}' not found.`,
          },
        ],
      });
    }
  );

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
