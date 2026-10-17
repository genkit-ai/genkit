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
import { generate } from '../../src/generate.js';
import { defineGenerateAction } from '../../src/generate/action.js';
import { generateMiddleware } from '../../src/generate/middleware.js';
import {
  resolveRestartedTools,
  resolveToolRequest,
  toToolMap,
} from '../../src/generate/resolve-tool-requests.js';
import { defineModel, type ToolResponsePart } from '../../src/model.js';
import { defineTool, tool } from '../../src/tool.js';

initNodeFeatures();

describe('multipart middleware tools', () => {
  for (const dynamic of [true, false]) {
    for (const actionRoute of [true, false]) {
      it(`preserves output, content and metadata (${dynamic ? 'dynamic' : 'registered'}, ${actionRoute ? 'action' : 'generate'})`, async () => {
        const registry = new Registry();
        const config = {
          name: 'multipartTool',
          description: 'Returns a result with additional context',
          multipart: true as const,
          inputSchema: z.object({ value: z.string() }),
          outputSchema: z.string(),
        };
        const implementation = async (input: { value: string }) => ({
          output: input.value,
          content: [{ text: 'Additional context' }],
          metadata: { source: 'middleware' },
        });
        const injectedTool = dynamic
          ? tool(config, implementation)
          : defineTool(registry, config, implementation);
        const middleware = generateMiddleware({ name: 'multipart' }, () => ({
          tools: [injectedTool],
        }));
        const expectedPart: ToolResponsePart = {
          toolResponse: {
            name: 'multipartTool',
            ref: 'call-1',
            output: 'hello',
            content: [{ text: 'Additional context' }],
          },
          metadata: { source: 'middleware' },
        };
        let calls = 0;
        const model = defineModel(
          registry,
          { name: 'testModel' },
          async (req) => {
            assert.strictEqual(req.tools?.[0].key, '/tool.v2/multipartTool');
            if (++calls === 1) {
              return {
                message: {
                  role: 'model',
                  content: [
                    {
                      toolRequest: {
                        name: 'multipartTool',
                        ref: 'call-1',
                        input: { value: 'hello' },
                      },
                    },
                  ],
                },
              };
            }
            assert.deepStrictEqual(req.messages.at(-1)?.content, [
              expectedPart,
            ]);
            return { message: { role: 'model', content: [{ text: 'done' }] } };
          }
        );

        if (actionRoute) {
          const action = defineGenerateAction(registry);
          const response = await action({
            model: model.__action.name,
            messages: [{ role: 'user', content: [{ text: 'hello' }] }],
            use: [middleware()],
          });
          assert.strictEqual(response.message?.content[0].text, 'done');
        } else {
          const response = await generate(registry, {
            model,
            prompt: 'hello',
            use: [middleware()],
          });
          assert.strictEqual(response.text, 'done');
          assert.deepStrictEqual(
            response.messages.find((message) => message.role === 'tool')
              ?.content,
            [expectedPart]
          );
        }
        assert.strictEqual(calls, 2);
      });
    }
  }

  it('preserves multipart responses when restarting an interrupted middleware tool', async () => {
    const registry = new Registry();
    registry.apiStability = 'beta';
    const injectedTool = tool(
      {
        name: 'approvalTool',
        description: 'Returns context after approval',
        multipart: true,
        outputSchema: z.string(),
      },
      async (_, ctx) => {
        if (!ctx.resumed) ctx.interrupt({ needsApproval: true });
        return {
          output: 'approved',
          content: [{ text: 'Approved context' }],
          metadata: { approved: true },
        };
      }
    );
    const middleware = generateMiddleware({ name: 'approval' }, () => ({
      tools: [injectedTool],
    }));
    let calls = 0;
    const model = defineModel(
      registry,
      { name: 'approvalModel' },
      async (req) => {
        if (++calls === 1) {
          return {
            message: {
              role: 'model',
              content: [{ toolRequest: { name: 'approvalTool', input: {} } }],
            },
          };
        }
        assert.deepStrictEqual(req.messages.at(-1)?.content, [
          {
            toolResponse: {
              name: 'approvalTool',
              output: 'approved',
              content: [{ text: 'Approved context' }],
            },
            metadata: { approved: true },
          },
        ]);
        return { message: { role: 'model', content: [{ text: 'done' }] } };
      }
    );
    const interrupted = await generate(registry, {
      model,
      prompt: 'hello',
      use: [middleware()],
    });
    assert.strictEqual(interrupted.finishReason, 'interrupted');
    const response = await generate(registry, {
      model,
      messages: interrupted.messages,
      use: [middleware()],
      resume: {
        restart: [injectedTool.restart(interrupted.interrupts[0], true)],
      },
    });
    assert.strictEqual(response.text, 'done');
    assert.strictEqual(calls, 2);
  });

  it('resolves multipart middleware tools during resumed request lookup', async () => {
    const registry = new Registry();
    const injectedTool = tool(
      {
        name: 'resumedTool',
        description: 'Returns resumed output',
        multipart: true,
        outputSchema: z.string(),
      },
      async () => ({ output: 'resumed output' })
    );
    const restarted = await resolveRestartedTools(
      registry,
      {
        messages: [
          {
            role: 'model',
            content: [
              {
                toolRequest: { name: 'resumedTool', input: {} },
                metadata: { resumed: true },
              },
            ],
          },
        ],
      },
      [{ tools: [injectedTool] }]
    );
    assert.strictEqual(restarted[0].metadata?.pendingOutput, 'resumed output');
  });

  it('rejects unknown and duplicate names in a mixed tool map', async () => {
    const multipartTool = tool({
      name: 'multipart',
      description: 'Multipart tool',
      multipart: true,
    });
    const basicTool = tool({ name: 'basic', description: 'Basic tool' });
    await assert.rejects(
      resolveToolRequest(
        { messages: [] },
        { toolRequest: { name: 'unknown', input: {} } },
        toToolMap([basicTool, multipartTool])
      ),
      /Tool unknown not found/
    );
    assert.throws(
      () =>
        toToolMap([
          multipartTool,
          tool({ name: 'multipart', description: 'Duplicate name' }),
        ]),
      /Cannot provide two tools with the same name/
    );
  });
});
