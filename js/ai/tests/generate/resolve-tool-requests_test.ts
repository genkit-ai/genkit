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
  resolveToolRequests,
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

  it('replays a pending output with its content and metadata, stripping the pending keys', async () => {
    const registry = new Registry();
    const rawRequest = {
      messages: [
        { role: 'user', content: [{ text: 'hi' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: { name: 'multiTool', ref: 'm1', input: {} },
              metadata: {
                pendingOutput: 'main output',
                pendingContent: [{ text: 'part 1' }],
                pendingMetadata: { custom: 'data' },
                other: 'kept',
              },
            },
          ],
        },
      ],
      resume: {},
    } as any;

    const result = await resolveResumeOption(registry, rawRequest, []);

    assert.deepStrictEqual(result.toolMessage?.content[0], {
      toolResponse: {
        name: 'multiTool',
        ref: 'm1',
        output: 'main output',
        content: [{ text: 'part 1' }],
      },
      metadata: { other: 'kept', source: 'pending', custom: 'data' },
    });
    assert.deepStrictEqual(
      result.revisedRequest?.messages[1].content[0].metadata,
      { other: 'kept' }
    );
  });

  it('replays a pending output that is falsy', async () => {
    const registry = new Registry();
    const rawRequest = {
      messages: [
        { role: 'user', content: [{ text: 'hi' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: { name: 'countTool', ref: 'c1', input: {} },
              metadata: { pendingOutput: 0 },
            },
          ],
        },
      ],
      resume: {},
    } as any;

    const result = await resolveResumeOption(registry, rawRequest, []);

    assert.strictEqual(
      (result.toolMessage?.content[0] as any).toolResponse.output,
      0
    );
  });

  it('replays a void output after a JSON round trip', async () => {
    const registry = new Registry();
    const voidTool = defineTool(
      registry,
      { name: 'voidTool', description: 'returns nothing' },
      async () => {}
    );
    const ask = defineTool(
      registry,
      {
        name: 'ask',
        description: 'asks',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        throw new ToolInterruptError({ q: 1 });
      }
    );

    const { revisedModelMessage } = await resolveToolRequests(
      { messages: [] } as any,
      {
        role: 'model',
        content: [
          { toolRequest: { name: 'voidTool', ref: 'v1', input: {} } },
          { toolRequest: { name: 'ask', ref: 'a1', input: {} } },
        ],
      },
      [voidTool, ask]
    );
    // Stashed as null: a session store's JSON drops a key holding undefined.
    assert.strictEqual(
      revisedModelMessage?.content[0].metadata?.pendingOutput,
      null
    );

    const persisted = JSON.parse(JSON.stringify(revisedModelMessage));
    const result = await resolveResumeOption(
      registry,
      {
        messages: [{ role: 'user', content: [{ text: 'hi' }] }, persisted],
        resume: {
          respond: [
            { toolResponse: { name: 'ask', ref: 'a1', output: 'yes' } },
          ],
        },
      } as any,
      [voidTool, ask]
    );
    assert.deepStrictEqual(
      result.toolMessage?.content.map((p: any) => [
        p.toolResponse.ref,
        p.toolResponse.output,
      ]),
      [
        ['v1', null],
        ['a1', 'yes'],
      ]
    );
  });

  it('emits resumed tool responses in request order', async () => {
    const registry = new Registry();
    const slowTool = defineTool(
      registry,
      {
        name: 'slowTool',
        description: 'slow tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        await new Promise((r) => setTimeout(r, 20));
        return 'slow done';
      }
    );

    const rawRequest = {
      messages: [
        { role: 'user', content: [{ text: 'hi' }] },
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'slowTool', ref: 's1', input: {} } },
            { toolRequest: { name: 'fastTool', ref: 'f1', input: {} } },
          ],
        },
      ],
      resume: {
        restart: [{ toolRequest: { name: 'slowTool', ref: 's1', input: {} } }],
        respond: [
          { toolResponse: { name: 'fastTool', ref: 'f1', output: 'fast' } },
        ],
      },
    } as any;

    const result = await resolveResumeOption(registry, rawRequest, [slowTool]);

    assert.deepStrictEqual(
      result.toolMessage?.content.map((p: any) => p.toolResponse.ref),
      ['s1', 'f1']
    );
  });

  it('keeps resolved siblings as pending outputs when a restarted tool interrupts again', async () => {
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
        throw new ToolInterruptError({ reason: 'again' });
      }
    );

    const rawRequest = {
      messages: [
        { role: 'user', content: [{ text: 'hi' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: { name: 'interruptTool', ref: 'i1', input: {} },
              metadata: { interrupt: true },
            },
            {
              toolRequest: { name: 'otherTool', ref: 'o1', input: {} },
              metadata: { interrupt: true },
            },
          ],
        },
      ],
      resume: {
        restart: [
          { toolRequest: { name: 'interruptTool', ref: 'i1', input: {} } },
        ],
        respond: [
          {
            toolResponse: { name: 'otherTool', ref: 'o1', output: 'answered' },
          },
        ],
      },
    } as any;

    const result = await resolveResumeOption(registry, rawRequest, [
      interruptTool,
    ]);

    assert.ok(result.interruptedResponse);
    const content = result.interruptedResponse.message!.content as any[];
    assert.deepStrictEqual(content[0].metadata.interrupt, { reason: 'again' });
    // The answered sibling is replayed on the next resume rather than demanded
    // again: its response rides on its request as a pending output.
    assert.strictEqual(content[1].metadata.pendingOutput, 'answered');
    assert.strictEqual(content[1].metadata.interrupt, undefined);
    assert.strictEqual(content[1].metadata.resolvedInterrupt, true);
  });
});

describe('resolveToolRequests', () => {
  it('emits tool responses in request order', async () => {
    const registry = new Registry();
    const slowTool = defineTool(
      registry,
      {
        name: 'slowTool',
        description: 'slow tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        await new Promise((r) => setTimeout(r, 20));
        return 'slow done';
      }
    );
    const fastTool = defineTool(
      registry,
      {
        name: 'fastTool',
        description: 'fast tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => 'fast done'
    );

    const { toolMessage, revisedModelMessage } = await resolveToolRequests(
      { messages: [] } as any,
      {
        role: 'model',
        content: [
          { toolRequest: { name: 'slowTool', ref: 's1', input: {} } },
          { toolRequest: { name: 'fastTool', ref: 'f1', input: {} } },
        ],
      },
      [slowTool, fastTool]
    );

    assert.strictEqual(revisedModelMessage, undefined);
    assert.deepStrictEqual(
      toolMessage?.content.map((p: any) => p.toolResponse.ref),
      ['s1', 'f1']
    );
  });

  it('stashes a multipart response beside pendingOutput on an interrupt', async () => {
    const registry = new Registry();
    const multiTool = defineTool(
      registry,
      {
        name: 'multiTool',
        description: 'multipart tool',
        multipart: true,
      },
      async () => ({
        output: 'main output',
        content: [{ text: 'part 1' }],
        metadata: { custom: 'data' },
      })
    );
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

    const { toolMessage, revisedModelMessage } = await resolveToolRequests(
      { messages: [] } as any,
      {
        role: 'model',
        content: [
          { toolRequest: { name: 'multiTool', ref: 'm1', input: {} } },
          { toolRequest: { name: 'interruptTool', ref: 'i1', input: {} } },
        ],
      },
      [multiTool, interruptTool]
    );

    assert.strictEqual(toolMessage, undefined);
    assert.deepStrictEqual(revisedModelMessage?.content[0].metadata, {
      pendingOutput: 'main output',
      pendingContent: [{ text: 'part 1' }],
      pendingMetadata: { custom: 'data' },
    });
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
