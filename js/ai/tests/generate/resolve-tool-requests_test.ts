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

import { z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { describe, it } from 'node:test';
import {
  assertValidToolNames,
  resolveRestartedTools,
  resolveResumeOption,
} from '../../src/generate/resolve-tool-requests.js';
import { ToolInterruptError, defineTool } from '../../src/tool.js';

initNodeFeatures();

describe('resolveRestartedTools', () => {
  it('should handle ToolInterruptError from a restarted tool', async () => {
    const registry = new Registry();
    const interruptTool = defineTool(
      registry,
      {
        name: 'interruptTool',
        description: 'interrupt tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        throw new ToolInterruptError({ reason: 'testing' });
      }
    );

    const rawRequest = {
      tools: [interruptTool],
      messages: [
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'interruptTool',
                input: {},
              },
              metadata: { resumed: true },
            },
          ],
        },
      ],
    } as any;

    const result = await resolveRestartedTools(registry, rawRequest);

    assert.strictEqual(result.length, 1);
    assert.deepStrictEqual(result[0].metadata?.interrupt, {
      reason: 'testing',
    });
  });
});

describe('resolveResumeOption', () => {
  it('should resolve provided tool response', async () => {
    const registry = new Registry();
    const tool = defineTool(
      registry,
      {
        name: 'testTool',
        description: 'test tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => 'test'
    );

    const rawRequest = {
      messages: [
        { role: 'user', content: [{ text: 'hi' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'testTool', ref: '123' } }],
        },
      ],
      resume: {
        respond: [
          {
            toolResponse: {
              name: 'testTool',
              ref: '123',
              output: 'manual answer',
            },
          },
        ],
      },
    } as any;

    const result = await resolveResumeOption(registry, rawRequest, [tool]);

    assert.ok(result.revisedRequest);
    assert.strictEqual(result.revisedRequest.messages.length, 3);
    assert.strictEqual(result.revisedRequest.messages[2].role, 'tool');
    assert.deepStrictEqual(
      (result.toolMessage?.content[0] as any).toolResponse.output,
      'manual answer'
    );
  });

  it('should handle ToolInterruptError from a restarted tool during resume', async () => {
    const registry = new Registry();
    const interruptTool = defineTool(
      registry,
      {
        name: 'interruptTool',
        description: 'interrupt tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        throw new ToolInterruptError({ reason: 'testing-resume' });
      }
    );

    const rawRequest = {
      messages: [
        { role: 'user', content: [{ text: 'hi' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: { name: 'interruptTool', ref: '123', input: {} },
            },
          ],
        },
      ],
      resume: {
        restart: [
          { toolRequest: { name: 'interruptTool', ref: '123', input: {} } },
        ],
      },
    } as any;

    const result = await resolveResumeOption(registry, rawRequest, [
      interruptTool,
    ]);

    assert.ok(result.interruptedResponse);
    assert.strictEqual(result.interruptedResponse.finishReason, 'interrupted');
    assert.deepStrictEqual(
      (result.interruptedResponse.message?.content[0] as any).metadata
        .interrupt,
      { reason: 'testing-resume' }
    );
  });
});

describe('assertValidToolNames', () => {
  it('should throw GenkitError on duplicate tool names', () => {
    const registry = new Registry();
    const tool1 = defineTool(
      registry,
      {
        name: 'test/tool',
        description: 'desc',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => ''
    );
    const tool2 = defineTool(
      registry,
      {
        name: 'other/tool',
        description: 'desc',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => ''
    );

    assert.throws(() => assertValidToolNames([tool1, tool2]), {
      name: 'GenkitError',
      status: 'INVALID_ARGUMENT',
    });
  });

  it('should pass on unique tool names', () => {
    const registry = new Registry();
    const tool1 = defineTool(
      registry,
      {
        name: 'test/tool1',
        description: 'desc',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => ''
    );
    const tool2 = defineTool(
      registry,
      {
        name: 'other/tool2',
        description: 'desc',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => ''
    );

    assert.doesNotThrow(() => assertValidToolNames([tool1, tool2]));
  });
});
