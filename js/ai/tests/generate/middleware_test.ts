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
import { beforeEach, describe, it } from 'node:test';
import {
  GenerateResponseChunk,
  generate,
  generateStream,
} from '../../src/generate.js';
import {
  GenerateMiddlewareDef,
  generateMiddleware,
} from '../../src/generate/middleware.js';
import { resolveRestartedTools } from '../../src/generate/resolve-tool-requests.js';
import { defineModel } from '../../src/model.js';
import { ToolInterruptError, defineTool, tool } from '../../src/tool.js';

initNodeFeatures();

describe('generateMiddleware', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
  });

  it('runs generate and model middleware in the correct order', async () => {
    const executionOrder: string[] = [];

    const mockModel = defineModel(
      registry,
      { name: 'mockModel' },
      async (req) => {
        executionOrder.push('modelExecution');
        return {
          message: {
            role: 'model',
            content: [{ text: 'response' }],
          },
        };
      }
    );

    const testMiddleware = generateMiddleware(
      { name: 'testMiddleware' },
      () => ({
        generate: async (req, ctx, next) => {
          executionOrder.push('generateBefore');
          const res = await next(req, ctx);
          executionOrder.push('generateAfter');
          return res;
        },
        model: async (req, ctx, next) => {
          executionOrder.push('modelBefore');
          const res = await next(req, ctx);
          executionOrder.push('modelAfter');
          return res;
        },
      })
    );

    await generate(registry, {
      model: mockModel,
      prompt: 'hi',
      use: [testMiddleware()],
    });

    assert.deepStrictEqual(executionOrder, [
      'generateBefore',
      'modelBefore',
      'modelExecution',
      'modelAfter',
      'generateAfter',
    ]);
  });

  it('runs tool middleware correctly', async () => {
    const executionOrder: string[] = [];

    const mockTool = defineTool(
      registry,
      {
        name: 'mockTool',
        description: 'A mock tool',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        executionOrder.push('toolExecution');
        return 'tool output';
      }
    );

    let turns = 0;
    const mockModel = defineModel(
      registry,
      { name: 'mockModelWithTool' },
      async (req) => {
        executionOrder.push('modelExecution');
        turns++;
        if (turns === 1) {
          return {
            message: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    name: mockTool.__action.name,
                    ref: '123',
                    input: {},
                  },
                },
              ],
            },
          };
        } else {
          return {
            message: {
              role: 'model',
              content: [{ text: 'final response' }],
            },
          };
        }
      }
    );

    const testMiddleware = generateMiddleware(
      { name: 'testMiddleware' },
      () => {
        let turnCount = 0;
        return {
          generate: async (req, ctx, next) => {
            const t = ++turnCount;
            executionOrder.push('generateBefore-' + t);
            const res = await next(req, ctx);
            executionOrder.push('generateAfter-' + t);
            return res;
          },
          model: async (req, ctx, next) => {
            executionOrder.push(`modelBefore-${turnCount}`);
            const res = await next(req, ctx);
            executionOrder.push(`modelAfter-${turnCount}`);
            return res;
          },
          tool: async (req, ctx, next) => {
            executionOrder.push(`toolBefore-${turnCount}`);
            const res = await next(req, ctx);
            executionOrder.push(`toolAfter-${turnCount}`);
            return res;
          },
        };
      }
    );

    await generate(registry, {
      model: mockModel,
      tools: [mockTool],
      prompt: 'hi',
      use: [testMiddleware()],
    });

    assert.deepStrictEqual(executionOrder, [
      'generateBefore-1',
      'modelBefore-1', // Turn 1
      'modelExecution',
      'modelAfter-1',
      'toolBefore-1', // Tool execution
      'toolExecution',
      'toolAfter-1',
      'generateBefore-2',
      'modelBefore-2', // Turn 2
      'modelExecution',
      'modelAfter-2',
      'generateAfter-2',
      'generateAfter-1',
    ]);
  });

  it('supports configuration and old-style function middleware', async () => {
    let configValue = '';

    const mockModel = defineModel(
      registry,
      { name: 'mockModel' },
      async (req) => {
        return {
          message: {
            role: 'model',
            content: [{ text: 'response' }],
          },
        };
      }
    );

    const testMiddleware = generateMiddleware(
      { name: 'configMw', configSchema: z.object({ val: z.string() }) },
      (options) => ({
        model: async (req, ctx, next) => {
          configValue = options.config?.val || '';
          return next(req, ctx);
        },
      })
    );

    let oldStyleExecuted = false;
    const oldStyleMiddleware = async (req: any, next: any) => {
      oldStyleExecuted = true;
      return next(req);
    };

    await generate(registry, {
      model: mockModel,
      prompt: 'test',
      use: [testMiddleware({ val: 'test_config' }), oldStyleMiddleware],
    });

    assert.strictEqual(configValue, 'test_config');
    assert.strictEqual(oldStyleExecuted, true);
  });

  it('supports pre-registered middleware (e.g. installed via plugin)', async () => {
    let executed = false;
    let configValue = '';

    const mockModel = defineModel(
      registry,
      { name: 'mockModel' },
      async (req) => {
        return {
          message: {
            role: 'model',
            content: [{ text: 'response' }],
          },
        };
      }
    );

    const preRegisteredMw = generateMiddleware<{ pluginOption: string }>(
      { name: 'preRegisteredMw', configSchema: z.object({ val: z.string() }) },
      (middlewareOpts) => ({
        model: async (req, ctx, next) => {
          executed = true;
          configValue = middlewareOpts.pluginConfig?.pluginOption || '';
          return next(req, ctx);
        },
      })
    );

    // Act as a plugin registering the middleware
    const myPlugin = preRegisteredMw.plugin({ pluginOption: 'plugin_config' });
    assert.ok(myPlugin.middleware);
    assert.deepStrictEqual((myPlugin.middleware()[0] as any).pluginOptions, {
      pluginOption: 'plugin_config',
    });

    const middlewares = myPlugin.middleware();
    assert.strictEqual(middlewares.length, 1);
    const mw = middlewares[0];

    // Verify name and properties are correctly copied/preserved
    assert.strictEqual(mw.name, 'preRegisteredMw');
    assert.strictEqual(mw.configSchema, preRegisteredMw.configSchema);
    assert.strictEqual(mw.toJson, preRegisteredMw.toJson);

    middlewares.forEach((mw: any) => {
      registry.registerValue('middleware', mw.name, mw);
    });

    await generate(registry, {
      model: mockModel,
      prompt: 'test',
      use: [{ name: 'preRegisteredMw' }],
    });

    assert.strictEqual(executed, true);
    assert.strictEqual(configValue, 'plugin_config');
  });

  it('should resolve tools injected by middleware during restarts', async () => {
    const middlewareTool = defineTool(
      registry,
      {
        name: 'middlewareTool',
        description: 'injected by middleware',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => 'success'
    );

    const middleware: GenerateMiddlewareDef = {
      tools: [middlewareTool],
    };

    const rawRequest = {
      tools: [],
      messages: [
        {
          role: 'model',
          content: [
            {
              toolRequest: { name: 'middlewareTool', input: {} },
              metadata: { resumed: true },
            },
          ],
        },
      ],
    } as any;

    const result = await resolveRestartedTools(registry, rawRequest, [
      middleware,
    ]);

    assert.strictEqual(result.length, 1);
    assert.deepStrictEqual(result[0].metadata?.pendingOutput, 'success');
  });

  it('throws an error if a middleware factory is passed without being called', async () => {
    const mockModel = defineModel(
      registry,
      { name: 'mockModel' },
      async () => ({
        message: { role: 'model', content: [{ text: 'done' }] },
        finishReason: 'stop',
      })
    );
    const streamModifyingMw = generateMiddleware({ name: 'dummy' }, () => ({}));

    await assert.rejects(
      async () => {
        await generate(registry, {
          model: mockModel,
          prompt: 'test',
          use: [streamModifyingMw as any],
        });
      },
      (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.match(err.message, /must be called with \(\)/);
        return true;
      }
    );
  });

  it('can intercept and modify the stream from model and generate interceptors', async () => {
    const chunkIntercepts: string[] = [];

    const mockStreamingModel = defineModel(
      registry,
      { name: 'mockStreamingModel' },
      async (req, streamingCallback) => {
        if (streamingCallback) {
          streamingCallback({ content: [{ text: 'chunk1' }] });
          streamingCallback({ content: [{ text: 'chunk2' }] });
        }
        return {
          message: {
            role: 'model',
            content: [{ text: 'chunk1chunk2' }],
          },
          finishReason: 'stop',
        };
      }
    );

    const streamModifyingMw = generateMiddleware(
      { name: 'streamModifier' },
      () => ({
        model: async (req, ctx, next) => {
          const originalOnChunk = ctx.onChunk;
          let interceptedCtx = ctx;
          if (originalOnChunk) {
            interceptedCtx = {
              ...ctx,
              onChunk: (chunk) => {
                chunkIntercepts.push(`model_mw: ${chunk.content[0].text}`);
                chunk.content[0].text = chunk.content[0].text?.toUpperCase();
                originalOnChunk(chunk);
              },
            };
          }
          return next(req, interceptedCtx);
        },
        generate: async (req, ctx, next) => {
          const originalOnChunk = ctx.onChunk;
          let interceptedCtx = ctx;
          if (originalOnChunk) {
            interceptedCtx = {
              ...ctx,
              onChunk: (chunk) => {
                chunkIntercepts.push(`gen_mw: ${chunk.content[0].text}`);
                chunk.content[0].text = `[${chunk.content[0].text}]`;
                originalOnChunk(chunk);
              },
            };
          }
          const res = await next(req, interceptedCtx);
          if (res.message) {
            return {
              ...res,
              message: {
                ...res.message,
                content: [
                  { text: `modified_result: ${res.message.content[0].text}` },
                ],
              },
            };
          }
          return res;
        },
      })
    );

    let finalChunks: string[] = [];

    const { response, stream } = generateStream(registry, {
      model: mockStreamingModel,
      prompt: 'test streaming mw',
      use: [streamModifyingMw()],
    });

    for await (const chunk of stream) {
      finalChunks.push(chunk.text);
    }

    const res = await response;

    assert.deepStrictEqual(chunkIntercepts, [
      'model_mw: chunk1',
      'gen_mw: CHUNK1',
      'model_mw: chunk2',
      'gen_mw: CHUNK2',
    ]);

    assert.deepStrictEqual(finalChunks, ['[CHUNK1]', '[CHUNK2]']);
    assert.strictEqual(res.text, 'modified_result: chunk1chunk2');
  });

  it('executes multiple middleware in the correct order', async () => {
    const executionOrder: string[] = [];

    const mw1 = generateMiddleware({ name: 'mw1' }, () => ({
      async generate(opts, ctx, next) {
        executionOrder.push('mw1:gen:start');
        const res = await next(opts, ctx);
        executionOrder.push('mw1:gen:end');
        return res;
      },
      async model(req, ctx, next) {
        executionOrder.push('mw1:model:start');
        const res = await next(req, ctx);
        executionOrder.push('mw1:model:end');
        return res;
      },
    }));

    const mw2 = generateMiddleware({ name: 'mw2' }, () => ({
      async generate(opts, ctx, next) {
        executionOrder.push('mw2:gen:start');
        const res = await next(opts, ctx);
        executionOrder.push('mw2:gen:end');
        return res;
      },
      async model(req, ctx, next) {
        executionOrder.push('mw2:model:start');
        const res = await next(req, ctx);
        executionOrder.push('mw2:model:end');
        return res;
      },
    }));

    const mockModel = defineModel(
      registry,
      { name: 'mockModel' },
      async () => ({
        message: { role: 'model', content: [{ text: 'done' }] },
        finishReason: 'stop',
      })
    );

    await generate(registry, {
      model: mockModel,
      prompt: 'test multiple',
      use: [mw1(), mw2()],
    });

    // The entire 'generate' layer runs before we ever descend to the 'model' level
    assert.deepStrictEqual(executionOrder, [
      'mw1:gen:start',
      'mw2:gen:start',
      'mw1:model:start',
      'mw2:model:start',
      'mw2:model:end',
      'mw1:model:end',
      'mw2:gen:end',
      'mw1:gen:end',
    ]);
  });

  it('supports a combination of new middleware and old-style functional middleware', async () => {
    const executionOrder: string[] = [];

    const newMw = generateMiddleware({ name: 'newMw' }, () => ({
      async generate(opts, ctx, next) {
        executionOrder.push('newMw:gen:start');
        const res = await next(opts, ctx);
        executionOrder.push('newMw:gen:end');
        return res;
      },
      async model(req, ctx, next) {
        executionOrder.push('newMw:model:start');
        const res = await next(req, ctx);
        executionOrder.push('newMw:model:end');
        return res;
      },
    }));

    const oldMw1 = async (req: any, next: any) => {
      executionOrder.push('oldMw1:model:start');
      const res = await next(); // Validating 0-argument backwards-compatibility
      executionOrder.push('oldMw1:model:end');
      return res;
    };

    const oldMw2 = async (req: any, ctx: any, next: any) => {
      executionOrder.push('oldMw2:model:start');
      const res = await next(req, ctx);
      executionOrder.push('oldMw2:model:end');
      return res;
    };

    const mockModel = defineModel(
      registry,
      { name: 'mockModel' },
      async () => ({
        message: { role: 'model', content: [{ text: 'done' }] },
        finishReason: 'stop',
      })
    );

    await generate(registry, {
      model: mockModel,
      prompt: 'test mixed',
      use: [oldMw1, newMw(), oldMw2],
    });

    assert.deepStrictEqual(executionOrder, [
      'newMw:gen:start', // Generate level ALWAYS runs first across full array
      'oldMw1:model:start',
      'newMw:model:start',
      'oldMw2:model:start',
      'oldMw2:model:end',
      'newMw:model:end',
      'oldMw1:model:end',
      'newMw:gen:end',
    ]);
  });

  it('injects tools from new-style generateMiddleware and executes tool requests', async () => {
    let toolExecutionCount = 0;

    const injectedTool = tool(
      {
        name: 'injectedTool',
        description: 'injected tool description',
        inputSchema: z.object({ arg: z.string() }),
        outputSchema: z.string(),
      },
      async (input) => {
        toolExecutionCount++;
        return `Result: ${input.arg}`;
      }
    );

    const toolMiddleware = generateMiddleware({ name: 'toolMw' }, () => ({
      tools: [injectedTool],
    }));

    let callCount = 0;
    const mockToolModel = defineModel(
      registry,
      { name: 'mockToolModel' },
      async (req) => {
        callCount++;
        // Assert that the tools sent to the model include the injected tool
        assert.ok(req.tools?.find((t) => t.name === 'injectedTool'));

        if (callCount === 1) {
          return {
            message: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    name: 'injectedTool',
                    ref: 'call_1',
                    input: { arg: 'hello' },
                  },
                },
              ],
            },
            finishReason: 'stop',
          };
        } else {
          assert.strictEqual(req.messages[2].role, 'tool');
          const toolData = req.messages[2].content[0].toolResponse;
          assert.strictEqual(toolData?.name, 'injectedTool');
          assert.strictEqual(toolData?.output, 'Result: hello');

          return {
            message: { role: 'model', content: [{ text: 'final response' }] },
            finishReason: 'stop',
          };
        }
      }
    );

    const result = await generate(registry, {
      model: mockToolModel,
      prompt: 'test tools',
      use: [toolMiddleware()],
    });

    assert.strictEqual(result.text, 'final response');
    assert.strictEqual(toolExecutionCount, 1);
  });

  it('handles ToolInterruptError from middleware', async () => {
    const mockTool = defineTool(
      registry,
      {
        name: 'interruptTool',
        description: 'interrupts',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        return 'foo';
      }
    );

    const interruptMiddleware = generateMiddleware(
      { name: 'interruptMw' },
      () => ({
        tool: async (req, ctx, next) => {
          throw new ToolInterruptError({ some: 'metadata' });
        },
      })
    );

    const mockModel = defineModel(
      registry,
      { name: 'mockModelWithTool' },
      async (req) => {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: mockTool.__action.name,
                  ref: '123',
                  input: {},
                },
              },
            ],
          },
        };
      }
    );

    const result = await generate(registry, {
      model: mockModel,
      prompt: 'hi',
      tools: ['interruptTool'],
      use: [interruptMiddleware()],
    });

    assert.strictEqual(result.finishReason, 'interrupted');
    const interruptPart = result.message?.content.find(
      (p) => p.metadata?.interrupt
    );
    assert.ok(interruptPart);
    assert.deepStrictEqual(interruptPart.metadata?.interrupt, {
      some: 'metadata',
    });
  });

  it('resumes tool execution with modified metadata after interrupt', async () => {
    const mockTool = defineTool(
      registry,
      {
        name: 'interruptTool',
        description: 'interrupts',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        return 'tool output';
      }
    );

    let middlewareRunCount = 0;
    const interruptMiddleware = generateMiddleware(
      { name: 'interruptMw' },
      () => ({
        tool: async (req, ctx, next) => {
          middlewareRunCount++;
          if (req.metadata?.['approved'] === true) {
            return next(req, ctx);
          }
          throw new ToolInterruptError({ some: 'metadata' });
        },
      })
    );

    let callCount = 0;
    const mockModel = defineModel(
      registry,
      { name: 'mockModelWithTool' },
      async (req) => {
        callCount++;
        if (callCount === 1) {
          return {
            message: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    name: mockTool.__action.name,
                    ref: '123',
                    input: {},
                  },
                },
              ],
            },
          };
        } else {
          return {
            message: {
              role: 'model',
              content: [{ text: 'final response' }],
            },
          };
        }
      }
    );

    const result = await generate(registry, {
      model: mockModel,
      prompt: 'hi',
      tools: ['interruptTool'],
      use: [interruptMiddleware()],
    });

    assert.strictEqual(result.finishReason, 'interrupted');
    const interruptPart = result.interrupts[0];
    assert.ok(interruptPart);
    assert.strictEqual(middlewareRunCount, 1);

    // Modify metadata
    if (interruptPart.metadata) {
      interruptPart.metadata = { ...interruptPart.metadata, approved: true };
    }

    const result2 = await generate(registry, {
      model: mockModel,
      messages: result.messages,
      tools: ['interruptTool'],
      use: [interruptMiddleware()],
      resume: {
        restart: [interruptPart],
      },
    });

    assert.strictEqual(result2.text, 'final response');
    // Middleware should have run again
    assert.strictEqual(middlewareRunCount, 2);
  });

  it('re-runs generate middleware after resuming tool execution', async () => {
    const mockTool = defineTool(
      registry,
      {
        name: 'interruptTool',
        description: 'interrupts',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => {
        return 'tool output';
      }
    );

    let generateMiddlewareCallCount = 0;
    let seenToolResponseInGenerate = false;

    const testMiddleware = generateMiddleware({ name: 'testMw' }, () => ({
      generate: async (req, ctx, next) => {
        generateMiddlewareCallCount++;
        const lastMsg = req.request.messages[req.request.messages.length - 1];
        if (lastMsg?.role === 'tool') {
          seenToolResponseInGenerate = true;
        }
        return next(req, ctx);
      },
      tool: async (req, ctx, next) => {
        if (req.metadata?.['approved'] === true) {
          return next(req, ctx);
        }
        throw new ToolInterruptError({ some: 'metadata' });
      },
    }));

    let callCount = 0;
    const mockModel = defineModel(
      registry,
      { name: 'mockModelWithTool2' },
      async (req) => {
        callCount++;
        if (callCount === 1) {
          return {
            message: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    name: mockTool.__action.name,
                    ref: '123',
                    input: {},
                  },
                },
              ],
            },
          };
        } else {
          return {
            message: {
              role: 'model',
              content: [{ text: 'final response' }],
            },
          };
        }
      }
    );

    const result = await generate(registry, {
      model: mockModel,
      prompt: 'hi',
      tools: ['interruptTool'],
      use: [testMiddleware()],
    });

    assert.strictEqual(result.finishReason, 'interrupted');
    const interruptPart = result.interrupts[0];
    assert.ok(interruptPart);

    // Modify metadata
    if (interruptPart.metadata) {
      interruptPart.metadata = { ...interruptPart.metadata, approved: true };
    }

    generateMiddlewareCallCount = 0; // Reset
    seenToolResponseInGenerate = false;

    await generate(registry, {
      model: mockModel,
      messages: result.messages,
      tools: ['interruptTool'],
      use: [testMiddleware()],
      resume: {
        restart: [interruptPart],
      },
    });

    assert.ok(
      seenToolResponseInGenerate,
      'Generate middleware should see the tool response'
    );
    assert.strictEqual(generateMiddlewareCallCount, 2);
  });

  it('should handle tool middleware returning undefined', async () => {
    const mockTool = tool(
      {
        name: 'mockTool',
        description: 'a mock tool',
        inputSchema: z.object({}),
      },
      async () => 'tool response'
    );

    const mockModel = defineModel(
      registry,
      { name: 'mockModelWithTool3' },
      async (req) => {
        if (req.messages.length === 1) {
          return {
            message: {
              role: 'model',
              content: [
                {
                  toolRequest: {
                    name: mockTool.__action.name,
                    ref: '123',
                    input: {},
                  },
                },
              ],
            },
          };
        }
        return { message: { role: 'model', content: [{ text: 'done' }] } };
      }
    );

    const testMiddleware = generateMiddleware(
      { name: 'swallowToolMw' },
      () => ({
        tool: async (req, ctx, next) => {
          return undefined; // Swallowing the tool call
        },
      })
    );

    const result = await generate(registry, {
      model: mockModel,
      prompt: 'hi',
      tools: [mockTool],
      use: [testMiddleware()],
    });

    // Verify it doesn't crash and completes.
    assert.strictEqual(result.text, 'done');

    // We expect 3 messages:
    // 1. User: "hi" (the prompt)
    // 2. Model: toolRequest (from Turn 1)
    // 3. Model: "done" (from Turn 2)
    // There should be NO 'tool' role message in between because the middleware swallowed it!
    assert.strictEqual(result.messages.length, 3);
    assert.strictEqual(result.messages[0].role, 'user');
    assert.strictEqual(result.messages[1].role, 'model');
    assert.strictEqual(result.messages[2].role, 'model'); // Consecutive model message!

    // Ensure no tool response parts exist
    const hasToolResponse = result.messages.some((m) =>
      m.content.some((c) => c.toolResponse)
    );
    assert.ok(!hasToolResponse, 'Should not contain any tool response');
  });

  it('passes and respects envelope updates in generate middleware', async () => {
    let receivedIndex = -1;
    let receivedTurn = -1;

    const mockModel = defineModel(
      registry,
      { name: 'mockModel' },
      async () => ({
        message: { role: 'model', content: [{ text: 'done' }] },
      })
    );

    const testMiddleware = generateMiddleware({ name: 'test' }, () => ({
      generate: async (envelope, ctx, next) => {
        receivedIndex = envelope.messageIndex;
        receivedTurn = envelope.currentTurn;
        // Increment messageIndex by 5 and currentTurn by 2
        return next(
          {
            ...envelope,
            messageIndex: envelope.messageIndex + 5,
            currentTurn: envelope.currentTurn + 2,
          },
          ctx
        );
      },
    }));

    let checkIndex = -1;
    let checkTurn = -1;
    const checkerMiddleware = generateMiddleware({ name: 'checker' }, () => ({
      generate: async (envelope, ctx, next) => {
        checkIndex = envelope.messageIndex;
        checkTurn = envelope.currentTurn;
        return next(envelope, ctx);
      },
    }));

    await generate(registry, {
      model: mockModel,
      prompt: 'hi',
      use: [testMiddleware(), checkerMiddleware()],
    });

    assert.strictEqual(receivedIndex, 0, 'Initial messageIndex should be 0');
    assert.strictEqual(receivedTurn, 0, 'Initial currentTurn should be 0');
    assert.strictEqual(
      checkIndex,
      5,
      'Checker should see incremented messageIndex'
    );
    assert.strictEqual(
      checkTurn,
      2,
      'Checker should see incremented currentTurn'
    );
  });

  it('wraps raw chunks from middleware in GenerateResponseChunk', async () => {
    const mockModel = defineModel(
      registry,
      { name: 'mockModel' },
      async () => ({
        message: { role: 'model', content: [{ text: 'done' }] },
      })
    );

    const rawChunkMiddleware = generateMiddleware({ name: 'rawChunk' }, () => ({
      generate: async (envelope, ctx, next) => {
        if (ctx.onChunk) {
          // Send a raw object instead of GenerateResponseChunk
          ctx.onChunk({ content: [{ text: 'raw content' }] } as any);
        }
        return next(envelope, ctx);
      },
    }));

    const chunks: any[] = [];
    const { stream, response } = generateStream(registry, {
      model: mockModel,
      prompt: 'test',
      use: [rawChunkMiddleware()],
    });

    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    await response;

    const rawChunk = chunks.find((c) => c.text === 'raw content');
    assert.ok(rawChunk, 'Should find the raw content chunk');
    assert.strictEqual(rawChunk.index, 0, 'Should have index 0');
    assert.deepStrictEqual(
      rawChunk.previousChunks,
      [],
      'Should have empty previousChunks'
    );
  });

  it('accumulates middleware chunks into sharedPreviousChunks for subsequent model chunks', async () => {
    const mockStreamingModel = defineModel(
      registry,
      { name: 'mockStreamingModel' },
      async (req, streamingCallback) => {
        if (streamingCallback) {
          // The model emits a chunk
          streamingCallback({ content: [{ text: 'model chunk' }] });
        }
        return {
          message: { role: 'model', content: [{ text: 'done' }] },
        };
      }
    );

    const rawChunkMiddleware = generateMiddleware(
      { name: 'rawChunk2' },
      () => ({
        generate: async (envelope, ctx, next) => {
          if (ctx.onChunk) {
            // Middleware emits a raw chunk BEFORE the model runs
            ctx.onChunk({ content: [{ text: 'middleware chunk ' }] } as any);
          }
          return next(envelope, ctx);
        },
      })
    );

    const chunks: GenerateResponseChunk[] = [];
    const { stream, response } = generateStream(registry, {
      model: mockStreamingModel,
      prompt: 'test',
      use: [rawChunkMiddleware()],
    });

    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    await response;

    // We expect 2 chunks in the stream
    assert.strictEqual(chunks.length, 2);
    assert.strictEqual(chunks[0].text, 'middleware chunk ');
    assert.strictEqual(chunks[1].text, 'model chunk');

    // CRITICAL ASSERTION: The model chunk should have the middleware chunk in its previousChunks!
    assert.strictEqual(chunks[1].previousChunks!.length, 1);
    assert.strictEqual(
      chunks[1].previousChunks![0].content[0].text,
      'middleware chunk '
    );
    assert.strictEqual(
      chunks[1].accumulatedText,
      'middleware chunk model chunk'
    );
  });
});
