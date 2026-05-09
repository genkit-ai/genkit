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

import { z, type PluginProvider } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import {
  generate,
  generateStream,
  normalizeMiddleware,
  toGenerateActionOptions,
  toGenerateRequest,
  type GenerateOptions,
} from '../../src/generate.js';
import { generateMiddleware } from '../../src/generate/middleware.js';
import {
  defineModel,
  type ModelAction,
  type ModelMiddleware,
  type ModelMiddlewareWithOptions,
} from '../../src/model.js';
import { defineResource } from '../../src/resource.js';
import { defineTool } from '../../src/tool.js';

initNodeFeatures();

describe('toGenerateRequest', () => {
  const registry = new Registry();
  // register tools
  const tellAFunnyJoke = defineTool(
    registry,
    {
      name: 'tellAFunnyJoke',
      description:
        'Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.',
      inputSchema: z.object({ topic: z.string() }),
      outputSchema: z.string(),
    },
    async (input) => {
      return `Why did the ${input.topic} cross the road?`;
    }
  );

  const namespacedPlugin: PluginProvider = {
    name: 'namespaced',
    initializer: async () => {},
  };
  registry.registerPluginProvider('namespaced', namespacedPlugin);

  defineTool(
    registry,
    {
      name: 'namespaced/add',
      description: 'add two numbers together',
      inputSchema: z.object({ a: z.number(), b: z.number() }),
      outputSchema: z.number(),
    },
    async ({ a, b }) => a + b
  );

  const testCases = [
    {
      should: 'translate a string prompt correctly',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        prompt: 'Tell a joke about dogs.',
      },
      expectedOutput: {
        messages: [
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [],
        output: {},
      },
    },
    {
      should:
        'translate a string prompt correctly with tools referenced by their name',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        tools: ['tellAFunnyJoke'],
        prompt: 'Tell a joke about dogs.',
      },
      expectedOutput: {
        messages: [
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [
          {
            name: 'tellAFunnyJoke',
            description:
              'Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.',
            outputSchema: {
              type: 'string',
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            inputSchema: {
              type: 'object',
              properties: { topic: { type: 'string' } },
              required: ['topic'],
              additionalProperties: true,
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
        ],
        output: {},
      },
    },
    {
      should: 'strip namespaces from tools when passing to the model',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        tools: ['namespaced/add'],
        prompt: 'Add 10 and 5.',
      },
      expectedOutput: {
        messages: [{ role: 'user', content: [{ text: 'Add 10 and 5.' }] }],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [
          {
            description: 'add two numbers together',
            inputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              additionalProperties: true,
              properties: { a: { type: 'number' }, b: { type: 'number' } },
              required: ['a', 'b'],
              type: 'object',
            },
            name: 'add',
            outputSchema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              type: 'number',
            },
            metadata: { originalName: 'namespaced/add' },
          },
        ],
        output: {},
      },
    },
    {
      should:
        'translate a string prompt correctly with tools referenced by their action',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        tools: [tellAFunnyJoke],
        prompt: 'Tell a joke about dogs.',
      },
      expectedOutput: {
        messages: [
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [
          {
            name: 'tellAFunnyJoke',
            description:
              'Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.',
            outputSchema: {
              type: 'string',
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
            inputSchema: {
              type: 'object',
              properties: { topic: { type: 'string' } },
              required: ['topic'],
              additionalProperties: true,
              $schema: 'http://json-schema.org/draft-07/schema#',
            },
          },
        ],
        output: {},
      },
    },
    {
      should: 'translate a media prompt correctly',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        prompt: [
          { text: 'describe the following image:' },
          {
            media: {
              url: 'https://picsum.photos/200',
              contentType: 'image/jpeg',
            },
          },
        ],
      },
      expectedOutput: {
        messages: [
          {
            role: 'user',
            content: [
              { text: 'describe the following image:' },
              {
                media: {
                  url: 'https://picsum.photos/200',
                  contentType: 'image/jpeg',
                },
              },
            ],
          },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [],
        output: {},
      },
    },
    {
      should: 'translate a prompt with history correctly',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        messages: [
          { content: [{ text: 'hi' }], role: 'user' },
          { content: [{ text: 'how can I help you' }], role: 'model' },
        ],
        prompt: 'Tell a joke about dogs.',
      },
      expectedOutput: {
        messages: [
          { content: [{ text: 'hi' }], role: 'user' },
          { content: [{ text: 'how can I help you' }], role: 'model' },
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [],
        output: {},
      },
    },
    {
      should: 'pass context through to the model',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        prompt: 'Tell a joke with context.',
        docs: [{ content: [{ text: 'context here' }] }],
      },
      expectedOutput: {
        messages: [
          { content: [{ text: 'Tell a joke with context.' }], role: 'user' },
        ],
        config: undefined,
        docs: [{ content: [{ text: 'context here' }] }],
        resources: [],
        tools: [],
        output: {},
      },
    },
    {
      should:
        'throw a FAILED_PRECONDITION error if trying to resume without a model message',
      prompt: {
        messages: [{ role: 'system', content: [{ text: 'sys' }] }],
        resume: {
          respond: { toolResponse: { name: 'test', output: { foo: 'bar' } } },
        },
      },
      throws: 'FAILED_PRECONDITION',
    },
    {
      should:
        'throw a FAILED_PRECONDITION error if trying to resume a model message without toolRequests',
      prompt: {
        messages: [
          { role: 'user', content: [{ text: 'hi' }] },
          { role: 'model', content: [{ text: 'there' }] },
        ],
        resume: {
          respond: { toolResponse: { name: 'test', output: { foo: 'bar' } } },
        },
      },
      throws: 'FAILED_PRECONDITION',
    },
    {
      should: 'passes through output options',
      prompt: {
        model: 'vertexai/gemini-1.0-pro',
        prompt: 'Tell a joke about dogs.',
        output: {
          constrained: true,
          format: 'banana',
        },
      },
      expectedOutput: {
        messages: [
          { role: 'user', content: [{ text: 'Tell a joke about dogs.' }] },
        ],
        config: undefined,
        docs: undefined,
        resources: [],
        tools: [],
        output: {
          constrained: true,
          format: 'banana',
        },
      },
    },
  ];
  for (const test of testCases) {
    it(test.should, async () => {
      if (test.throws) {
        await assert.rejects(
          async () => {
            await toGenerateRequest(registry, test.prompt as GenerateOptions);
          },
          { name: 'GenkitError', status: test.throws }
        );
      } else {
        const actualOutput = await toGenerateRequest(
          registry,
          test.prompt as GenerateOptions
        );
        assert.deepStrictEqual(actualOutput, test.expectedOutput);
      }
    });
  }
});

describe('toGenerateActionOptions', () => {
  const registry = new Registry();

  it('should return action options with undefined model', async () => {
    const options: GenerateOptions = {
      prompt: 'hello',
    };
    const actionOptions = await toGenerateActionOptions(registry, options);
    assert.strictEqual(actionOptions.model, undefined);
    assert.deepStrictEqual(actionOptions.messages, [
      { role: 'user', content: [{ text: 'hello' }] },
    ]);
  });
});

describe('generate', () => {
  let registry: Registry;
  var echoModel: ModelAction;

  beforeEach(() => {
    registry = new Registry();
    echoModel = defineModel(
      registry,
      {
        name: 'echoModel',
      },
      async (request) => {
        return {
          message: {
            role: 'model',
            content: [
              {
                text:
                  'Echo: ' +
                  request.messages
                    .map((m) => m.content.map((c) => c.text).join())
                    .join(),
              },
            ],
          },
          finishReason: 'stop',
        };
      }
    );
  });

  it('applies middleware', async () => {
    const wrapRequest: ModelMiddleware = async (req, next) => {
      return next({
        ...req,
        messages: [
          {
            role: 'user',
            content: [
              {
                text:
                  '(' +
                  req.messages
                    .map((m) => m.content.map((c) => c.text).join())
                    .join() +
                  ')',
              },
            ],
          },
        ],
      });
    };
    const wrapResponse: ModelMiddleware = async (req, next) => {
      const res = await next(req);
      return {
        message: {
          role: 'model',
          content: [
            {
              text: '[' + res.message!.content.map((c) => c.text).join() + ']',
            },
          ],
        },
        finishReason: res.finishReason,
      };
    };

    const response = await generate(registry, {
      prompt: 'banana',
      model: echoModel,
      use: [wrapRequest, wrapResponse],
    });
    const want = '[Echo: (banana)]';
    assert.deepStrictEqual(response.text, want);
  });
});

describe('generate', () => {
  let registry: Registry;
  beforeEach(() => {
    registry = new Registry();

    defineModel(
      registry,
      { name: 'echo', supports: { tools: true } },
      async (input) => ({
        message: input.messages[0],
        finishReason: 'stop',
      })
    );
  });

  it('should preserve the request in the returned response, enabling .messages', async () => {
    const response = await generate(registry, {
      model: 'echo',
      prompt: 'Testing messages',
    });
    assert.deepEqual(
      response.messages.map((m) => m.content[0].text),
      ['Testing messages', 'Testing messages']
    );
  });

  it('applies resources in the registry', async () => {
    defineResource(
      registry,
      { name: 'testResource', template: 'test://resource/{param}' },
      (input) => ({
        content: [{ text: 'resource' }, { text: input.uri }],
      })
    );

    const response = await generate(registry, {
      model: 'echo',
      prompt: [
        { text: 'some text' },
        { resource: { uri: 'test://resource/value' } },
      ],
    });
    assert.deepEqual(response.messages[0].content, [
      { text: 'some text' },
      {
        metadata: {
          resource: {
            template: 'test://resource/{param}',
            uri: 'test://resource/value',
          },
        },
        text: 'resource',
      },
      {
        metadata: {
          resource: {
            template: 'test://resource/{param}',
            uri: 'test://resource/value',
          },
        },
        text: 'test://resource/value',
      },
    ]);
  });

  it('throws when resource not found', async () => {
    const response = generate(registry, {
      model: 'echo',
      prompt: [{ text: 'some text' }, { resource: { uri: 'test://resource' } }],
    });
    await assert.rejects(response, {
      message:
        'NOT_FOUND: failed to find matching resource for test://resource',
    });
  });

  describe('generateStream', () => {
    it('should stream out chunks', async () => {
      const registry = new Registry();

      defineModel(
        registry,
        { name: 'echo-streaming', supports: { tools: true } },
        async (input, streamingCallback) => {
          streamingCallback!({ content: [{ text: 'hello, ' }] });
          streamingCallback!({ content: [{ text: 'world!' }] });
          return {
            message: input.messages[0],
            finishReason: 'stop',
          };
        }
      );

      const { response, stream } = generateStream(registry, {
        model: 'echo-streaming',
        prompt: 'Testing streaming',
      });

      const streamed: any[] = [];
      for await (const chunk of stream) {
        streamed.push(chunk.toJSON());
      }
      assert.deepStrictEqual(streamed, [
        {
          index: 0,
          role: 'model',
          content: [{ text: 'hello, ' }],
        },
        {
          index: 0,
          role: 'model',
          content: [{ text: 'world!' }],
        },
      ]);
      assert.deepEqual(
        (await response).messages.map((m) => m.content[0].text),
        ['Testing streaming', 'Testing streaming']
      );
    });

    it('should stream out chunks (v2 model)', async () => {
      const registry = new Registry();

      defineModel(
        registry,
        { apiVersion: 'v2', name: 'echo-streaming', supports: { tools: true } },
        async (input, { sendChunk }) => {
          sendChunk({ content: [{ text: 'hello, ' }] });
          sendChunk({ content: [{ text: 'world!' }] });
          return {
            message: input.messages[0],
            finishReason: 'stop',
          };
        }
      );

      const { response, stream } = generateStream(registry, {
        model: 'echo-streaming',
        prompt: 'Testing streaming',
      });

      const streamed: any[] = [];
      for await (const chunk of stream) {
        streamed.push(chunk.toJSON());
      }
      assert.deepStrictEqual(streamed, [
        {
          index: 0,
          role: 'model',
          content: [{ text: 'hello, ' }],
        },
        {
          index: 0,
          role: 'model',
          content: [{ text: 'world!' }],
        },
      ]);
      assert.deepEqual(
        (await response).messages.map((m) => m.content[0].text),
        ['Testing streaming', 'Testing streaming']
      );
    });
  });

  it('should use custom stepName parameter in tracing', async () => {
    const response = await generate(registry, {
      model: 'echo',
      prompt: 'Testing custom step name',
      stepName: 'test-generate-custom',
    });
    assert.deepEqual(
      response.messages.map((m) => m.content[0].text),
      ['Testing custom step name', 'Testing custom step name']
    );
  });

  it('should default to "generate" name when no stepName is provided', async () => {
    const response = await generate(registry, {
      model: 'echo',
      prompt: 'Testing default step name',
    });
    assert.deepEqual(
      response.messages.map((m) => m.content[0].text),
      ['Testing default step name', 'Testing default step name']
    );
  });

  it('handles multipart tool responses', async () => {
    defineTool(
      registry,
      {
        name: 'multiTool',
        description: 'a tool with multiple parts',
        multipart: true,
      },
      async () => {
        return {
          output: 'main output',
          content: [{ text: 'part 1' }],
          metadata: { custom: 'data' },
        };
      }
    );

    let requestCount = 0;
    defineModel(
      registry,
      { name: 'multi-tool-model', supports: { tools: true } },
      async (input) => {
        requestCount++;
        return {
          message: {
            role: 'model',
            content: [
              requestCount == 1
                ? {
                    toolRequest: {
                      name: 'multiTool',
                      input: {},
                    },
                  }
                : { text: 'done' },
            ],
          },
          finishReason: 'stop',
        };
      }
    );

    const response = await generate(registry, {
      model: 'multi-tool-model',
      prompt: 'go',
      tools: ['multiTool'],
    });
    assert.deepStrictEqual(response.messages, [
      {
        role: 'user',
        content: [
          {
            text: 'go',
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            toolRequest: {
              name: 'multiTool',
              input: {},
            },
          },
        ],
      },
      {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'multiTool',
              output: 'main output',
              content: [
                {
                  text: 'part 1',
                },
              ],
            },
            metadata: { custom: 'data' },
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            text: 'done',
          },
        ],
      },
    ]);
  });

  it('handles fallback tool responses', async () => {
    defineTool(
      registry,
      {
        name: 'fallbackTool',
        description: 'a tool with fallback output',
        multipart: true,
      },
      async () => {
        return {
          output: 'fallback output',
          content: [{ text: 'part 1' }],
        };
      }
    );

    let requestCount = 0;
    defineModel(
      registry,
      { name: 'fallback-tool-model', supports: { tools: true } },
      async (input) => {
        requestCount++;
        return {
          message: {
            role: 'model',
            content: [
              requestCount == 1
                ? {
                    toolRequest: {
                      name: 'fallbackTool',
                      input: {},
                    },
                  }
                : { text: 'done' },
            ],
          },
          finishReason: 'stop',
        };
      }
    );

    const response = await generate(registry, {
      model: 'fallback-tool-model',
      prompt: 'go',
      tools: ['fallbackTool'],
    });
    assert.deepStrictEqual(response.messages, [
      {
        role: 'user',
        content: [
          {
            text: 'go',
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            toolRequest: {
              name: 'fallbackTool',
              input: {},
            },
          },
        ],
      },
      {
        role: 'tool',
        content: [
          {
            toolResponse: {
              name: 'fallbackTool',
              output: 'fallback output',
              content: [
                {
                  text: 'part 1',
                },
              ],
            },
          },
        ],
      },
      {
        role: 'model',
        content: [
          {
            text: 'done',
          },
        ],
      },
    ]);
  });

  it('middleware can intercept streaming callback', async () => {
    const registry = new Registry();
    const echoModel = defineModel(
      registry,
      {
        apiVersion: 'v2',
        name: 'echoModel',
        supports: { tools: true },
      },
      async (_, { sendChunk }) => {
        if (sendChunk) {
          sendChunk({ content: [{ text: 'chunk1' }] });
          sendChunk({ content: [{ text: 'chunk2' }] });
        }
        return {
          message: {
            role: 'model',
            content: [{ text: 'done' }],
          },
          finishReason: 'stop',
        };
      }
    );

    const interceptMiddleware: ModelMiddlewareWithOptions = async (
      req,
      opts,
      next
    ) => {
      const originalOnChunk = opts!.onChunk;
      return next(req, {
        ...opts,
        onChunk: (chunk) => {
          if (originalOnChunk) {
            const text = chunk.content?.[0]?.text;
            originalOnChunk({
              ...chunk,
              content: [{ text: `intercepted: ${text}` }],
            });
          }
        },
      });
    };

    const { response, stream } = generateStream(registry, {
      model: echoModel,
      prompt: 'test',
      use: [interceptMiddleware],
    });

    const streamed: any[] = [];
    for await (const chunk of stream) {
      streamed.push(chunk.content[0].text);
    }

    assert.deepStrictEqual(streamed, [
      'intercepted: chunk1',
      'intercepted: chunk2',
    ]);
    await response;
  });

  it('middleware can modify context', async () => {
    const registry = new Registry();
    const checkContextModel = defineModel(
      registry,
      {
        apiVersion: 'v2',
        name: 'checkContextModel',
        supports: { context: true },
      },
      async (request, { context }) => {
        return {
          message: {
            role: 'model',
            content: [{ text: `Context: ${context?.myValue}` }],
          },
          finishReason: 'stop',
        };
      }
    );

    const contextMiddleware: ModelMiddlewareWithOptions = async (
      req,
      opts,
      next
    ) => {
      return next(req, {
        ...opts,
        context: {
          ...opts?.context,
          myValue: 'foo',
        },
      });
    };

    const response = await generate(registry, {
      model: checkContextModel,
      prompt: 'test',
      use: [contextMiddleware],
    });

    assert.strictEqual(response.text, 'Context: foo');
  });

  it('middleware can chain option modifications', async () => {
    const registry = new Registry();
    const checkContextModel = defineModel(
      registry,
      {
        apiVersion: 'v2',
        name: 'checkContextModel',
        supports: { context: true },
      },
      async (request, { context }) => {
        return {
          message: {
            role: 'model',
            content: [{ text: `Context: ${JSON.stringify(context)}` }],
          },
          finishReason: 'stop',
        };
      }
    );

    const middleware1: ModelMiddlewareWithOptions = async (req, opts, next) => {
      return next(req, {
        ...opts,
        context: {
          ...opts?.context,
          val: [...(opts?.context?.val ?? []), 'A'],
        },
      });
    };

    const middleware2: ModelMiddlewareWithOptions = async (req, opts, next) => {
      return next(req, {
        ...opts,
        context: {
          ...opts?.context,
          val: [...(opts?.context?.val ?? []), 'B'],
        },
      });
    };

    const response = await generate(registry, {
      model: checkContextModel,
      prompt: 'test',
      use: [middleware1, middleware2],
    });

    const context = JSON.parse(response.text.substring('Context: '.length));
    assert.deepStrictEqual(context.val, ['A', 'B']);
  });
});

describe('normalizeMiddleware', () => {
  it('handles legacy functional middleware by wrapping it', async () => {
    const registry = new Registry();
    const legacyMw = async (req: any, next: any) => {
      return next(req);
    };

    const refs = await normalizeMiddleware(registry, [legacyMw]);

    assert.strictEqual(refs.length, 1);
    assert.match(refs[0].name, /^dynamic-middleware-\d+-/);

    const registered = await registry.lookupValue<any>(
      'middleware',
      refs[0].name
    );
    assert.ok(registered);
  });

  it('handles MiddlewareRef objects created by calling middleware', async () => {
    const registry = new Registry();
    const myMw = generateMiddleware({ name: 'myMw' }, () => ({}));

    // Call it to get a MiddlewareRef
    const refs = await normalizeMiddleware(registry, [myMw()]);

    assert.strictEqual(refs.length, 1);
    assert.strictEqual(refs[0].name, 'myMw');

    const registered = await registry.lookupValue<any>('middleware', 'myMw');
    assert.ok(registered);
  });

  it('handles MiddlewareRef objects', async () => {
    const registry = new Registry();
    const myMw = generateMiddleware({ name: 'myMw' }, () => ({}));
    registry.registerValue('middleware', 'myMw', myMw);

    const refs = await normalizeMiddleware(registry, [{ name: 'myMw' }]);

    assert.strictEqual(refs.length, 1);
    assert.strictEqual(refs[0].name, 'myMw');
  });

  it('throws when uncalled middleware definition is passed as a function', async () => {
    const registry = new Registry();
    const myMw = generateMiddleware({ name: 'myMw' }, () => ({}));

    // Pass the definition function itself, which has .instantiate and .plugin
    await assert.rejects(
      async () => {
        await normalizeMiddleware(registry, [myMw as any]);
      },
      {
        name: 'GenkitError',
        status: 'INVALID_ARGUMENT',
      }
    );
  });
});
