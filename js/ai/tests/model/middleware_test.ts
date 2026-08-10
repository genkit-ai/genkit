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

import { GenkitError, z } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import type { DocumentData } from '../../src/document.js';
import { configureFormats } from '../../src/formats/index.js';
import { generate } from '../../src/generate.js';
import {
  defineModel,
  type GenerateRequest,
  type GenerateResponseData,
  type MessageData,
} from '../../src/model.js';
import {
  CONTEXT_PREFACE,
  TEST_ONLY,
  augmentWithContext,
  fallback,
  retry,
  simulateConstrainedGeneration,
  simulateSystemPrompt,
  validateSupport,
  type AugmentWithContextOptions,
} from '../../src/model/middleware.js';
import { defineProgrammableModel } from '../helpers.js';

const { setRetryTimeout } = TEST_ONLY;

initNodeFeatures();

describe('validateSupport', () => {
  const examples: Record<string, GenerateRequest> = {
    multiturn: {
      messages: [
        { role: 'user', content: [{ text: 'hello' }] },
        { role: 'model', content: [{ text: 'hi' }] },
        { role: 'user', content: [{ text: 'how are you' }] },
      ],
    },
    media: {
      messages: [
        {
          role: 'user',
          content: [{ media: { url: 'https://example.com/image.png' } }],
        },
      ],
    },
    tools: {
      messages: [
        {
          role: 'user',
          content: [{ media: { url: 'https://example.com/image.png' } }],
        },
      ],
      tools: [
        {
          name: 'someTool',
          description: 'hello world',
          inputSchema: { type: 'object' },
        },
      ],
    },
    json: {
      messages: [
        {
          role: 'user',
          content: [{ text: 'hello world' }],
        },
      ],
      output: { format: 'json' },
    },
  };

  let nextCalled = false;
  const noopNext: (
    req?: GenerateRequest
  ) => Promise<GenerateResponseData> = async () => {
    nextCalled = true;
    return {};
  };
  beforeEach(() => (nextCalled = false));

  it('accepts anything when no supports is present', () => {
    const runner = validateSupport({ name: 'test-model' });
    for (const example of Object.values(examples)) {
      runner(example, noopNext);
    }
    assert.ok(nextCalled, "next() wasn't called");
  });

  it('throws when media is supplied but not supported', async () => {
    const runner = validateSupport({
      name: 'test-model',
      supports: {
        media: false,
      },
    });
    await assert.rejects(
      runner(examples.media, noopNext),
      /does not support media/
    );
  });

  it('throws when tools are supplied but not supported', async () => {
    const runner = validateSupport({
      name: 'test-model',
      supports: {
        tools: false,
      },
    });
    await assert.rejects(
      runner(examples.tools, noopNext),
      /does not support tool use/
    );
  });

  it('throws when multiturn messages are supplied but not supported', async () => {
    const runner = validateSupport({
      name: 'test-model',
      supports: {
        multiturn: false,
      },
    });
    await assert.rejects(
      runner(examples.multiturn, noopNext),
      /does not support multiple messages/
    );
  });
});

describe('retry', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
    configureFormats(registry);
  });

  it('should not retry on success', async () => {
    const pm = defineProgrammableModel(registry);
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [{ text: 'success' }],
        },
      };
    };

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [retry()],
    });

    assert.strictEqual(pm.requestCount, 1);
    assert.strictEqual(result.text, 'success');
  });

  it('should retry on a retryable GenkitError', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      if (requestCount < 3) {
        throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
      }
      return {
        message: {
          role: 'model',
          content: [{ text: 'success' }],
        },
      };
    };

    setRetryTimeout((callback, ms) => {
      callback();
      return 0 as any;
    });

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [retry({ maxRetries: 3 })],
    });

    assert.strictEqual(requestCount, 3);
    assert.strictEqual(result.text, 'success');
  });

  it('should retry on a non-GenkitError', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      if (requestCount < 2) {
        throw new Error('generic error');
      }
      return {
        message: {
          role: 'model',
          content: [{ text: 'success' }],
        },
      };
    };

    setRetryTimeout((callback, ms) => {
      callback();
      return 0 as any;
    });

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [retry({ maxRetries: 2 })],
    });

    assert.strictEqual(requestCount, 2);
    assert.strictEqual(result.text, 'success');
  });

  it('should throw after exhausting retries', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
    };

    setRetryTimeout((callback, ms) => {
      callback();
      return 0 as any;
    });

    await assert.rejects(
      generate(registry, {
        model: 'programmableModel',
        prompt: 'test',
        use: [retry({ maxRetries: 2 })],
      }),
      /UNAVAILABLE: test/
    );

    assert.strictEqual(requestCount, 3);
  });

  it('should call onError callback', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      throw new Error('test error');
    };

    setRetryTimeout((callback, ms) => {
      callback();
      return 0 as any;
    });

    let errorCount = 0;
    let lastError: Error | undefined;
    await assert.rejects(
      generate(registry, {
        model: 'programmableModel',
        prompt: 'test',
        use: [
          retry({
            maxRetries: 2,
            onError: (err, attempt) => {
              errorCount++;
              lastError = err;
              assert.strictEqual(attempt, errorCount);
            },
          }),
        ],
      }),
      /test error/
    );

    assert.strictEqual(requestCount, 3);
    assert.strictEqual(errorCount, 2);
    assert.ok(lastError);
    assert.strictEqual(lastError!.message, 'test error');
  });

  it('should not retry on non-retryable status', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      throw new GenkitError({ status: 'INVALID_ARGUMENT', message: 'test' });
    };

    await assert.rejects(
      generate(registry, {
        model: 'programmableModel',
        prompt: 'test',
        use: [retry({ maxRetries: 2 })],
      }),
      /INVALID_ARGUMENT: test/
    );

    assert.strictEqual(requestCount, 1);
  });

  it('should respect initial delay', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      if (requestCount < 2) {
        throw new Error('generic error');
      }
      return {
        message: {
          role: 'model',
          content: [{ text: 'success' }],
        },
      };
    };

    let totalDelay = 0;
    setRetryTimeout((callback, ms) => {
      totalDelay += ms!;
      callback();
      return 0 as any;
    });

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [retry({ maxRetries: 2, initialDelayMs: 50, noJitter: true })],
    });

    assert.strictEqual(requestCount, 2);
    assert.strictEqual(result.text, 'success');
    assert.strictEqual(totalDelay, 50);
  });

  it('should respect backoff factor', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      if (requestCount < 3) {
        throw new Error('generic error');
      }
      return {
        message: {
          role: 'model',
          content: [{ text: 'success' }],
        },
      };
    };

    let totalDelay = 0;
    setRetryTimeout((callback, ms) => {
      totalDelay += ms!;
      callback();
      return 0 as any;
    });

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [
        retry({
          maxRetries: 3,
          initialDelayMs: 20,
          backoffFactor: 2,
          noJitter: true,
        }),
      ],
    });

    assert.strictEqual(requestCount, 3);
    assert.strictEqual(result.text, 'success');
    assert.strictEqual(totalDelay, 20 + 40);
  });

  it('should apply jitter', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      if (requestCount < 2) {
        throw new Error('generic error');
      }
      return {
        message: {
          role: 'model',
          content: [{ text: 'success' }],
        },
      };
    };

    let totalDelay = 0;
    setRetryTimeout((callback, ms) => {
      totalDelay += ms!;
      callback();
      return 0 as any;
    });

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [
        retry({
          maxRetries: 2,
          initialDelayMs: 50,
          noJitter: false, // do jitter
        }),
      ],
    });

    assert.strictEqual(requestCount, 2);
    assert.strictEqual(result.text, 'success');
    assert.ok(totalDelay >= 50);
    assert.ok(totalDelay <= 1050);
  });

  it('should respect max delay', async () => {
    const pm = defineProgrammableModel(registry);
    let requestCount = 0;
    pm.handleResponse = async (req, sc) => {
      requestCount++;
      if (requestCount < 3) {
        throw new Error('generic error');
      }
      return {
        message: {
          role: 'model',
          content: [{ text: 'success' }],
        },
      };
    };

    let totalDelay = 0;
    setRetryTimeout((callback, ms) => {
      totalDelay += ms!;
      callback();
      return 0 as any;
    });

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [
        retry({
          maxRetries: 3,
          initialDelayMs: 20,
          backoffFactor: 2,
          maxDelayMs: 30,
          noJitter: true,
        }),
      ],
    });

    assert.strictEqual(requestCount, 3);
    assert.strictEqual(result.text, 'success');
    assert.strictEqual(totalDelay, 20 + 30);
  });
});

describe('fallback', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
    configureFormats(registry);
  });

  it('should not fallback on success', async () => {
    const pm = defineProgrammableModel(registry, {}, 'programmableModel');
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [{ text: 'success' }],
        },
      };
    };

    const fallbackPm = defineProgrammableModel(registry, {}, 'fallbackModel');

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [fallback({ registry }, { models: ['fallbackModel'] })],
    });

    assert.strictEqual(pm.requestCount, 1);
    assert.strictEqual(fallbackPm.requestCount, 0);
  });

  it('should call onError callback', async () => {
    const pm = defineProgrammableModel(registry, {}, 'programmableModel');
    pm.handleResponse = async () => {
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
    };

    const fallbackPm = defineProgrammableModel(registry, {}, 'fallbackModel');
    fallbackPm.handleResponse = async () => {
      throw new GenkitError({ status: 'INTERNAL', message: 'fallback fail' });
    };

    let errorCount = 0;
    let lastError: Error | undefined;
    await assert.rejects(
      generate(registry, {
        model: 'programmableModel',
        prompt: 'test',
        use: [
          fallback({ registry } as any, {
            models: ['fallbackModel'],
            onError: (err) => {
              errorCount++;
              lastError = err;
            },
          }),
        ],
      }),
      /INTERNAL: fallback fail/
    );

    assert.strictEqual(pm.requestCount, 1);
    assert.strictEqual(fallbackPm.requestCount, 1);
    assert.strictEqual(errorCount, 2);
    assert.ok(lastError);
    assert.strictEqual(lastError!.message, 'INTERNAL: fallback fail');
  });

  it('should fallback on a fallbackable error', async () => {
    const pm = defineProgrammableModel(registry, {}, 'programmableModel');
    pm.handleResponse = async () => {
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
    };

    const fallbackPm = defineProgrammableModel(registry, {}, 'fallbackModel');
    fallbackPm.handleResponse = async () => {
      return {
        message: {
          role: 'model',
          content: [{ text: 'fallback success' }],
        },
      };
    };

    const result = await generate(registry, {
      model: 'programmableModel',
      prompt: 'test',
      use: [fallback({ registry }, { models: ['fallbackModel'] })],
    });

    assert.strictEqual(pm.requestCount, 1);
    assert.strictEqual(fallbackPm.requestCount, 1);
    assert.strictEqual(result.text, 'fallback success');
  });

  it('should throw after all fallbacks fail', async () => {
    const pm = defineProgrammableModel(registry, {}, 'programmableModel');
    pm.handleResponse = async (req, sc) => {
      throw new GenkitError({ status: 'UNAVAILABLE', message: 'test' });
    };

    const fallbackPm = defineProgrammableModel(registry, {}, 'fallbackModel');
    fallbackPm.handleResponse = async (req, sc) => {
      throw new GenkitError({ status: 'INTERNAL', message: 'fallback fail' });
    };

    await assert.rejects(
      generate(registry, {
        model: 'programmableModel',
        prompt: 'test',
        use: [fallback({ registry }, { models: ['fallbackModel'] })],
      }),
      /INTERNAL: fallback fail/
    );

    assert.strictEqual(pm.requestCount, 1);
    assert.strictEqual(fallbackPm.requestCount, 1);
  });

  it('should not fallback on non-fallbackable error', async () => {
    const pm = defineProgrammableModel(registry, {}, 'programmableModel');
    pm.handleResponse = async (req, sc) => {
      throw new GenkitError({ status: 'INVALID_ARGUMENT', message: 'test' });
    };

    const fallbackPm = defineProgrammableModel(registry, {}, 'fallbackModel');

    await assert.rejects(
      generate(registry, {
        model: 'programmableModel',
        prompt: 'test',
        use: [fallback({ registry }, { models: ['fallbackModel'] })],
      }),
      /INVALID_ARGUMENT: test/
    );

    assert.strictEqual(pm.requestCount, 1);
    assert.strictEqual(fallbackPm.requestCount, 0);
  });
});

const registry = new Registry();
configureFormats(registry);

const echoModel = defineModel(registry, { name: 'echo' }, async (req) => {
  return {
    finishReason: 'stop',
    message: {
      role: 'model',
      content: [{ data: req }],
    },
  };
});

describe('simulateSystemPrompt', () => {
  function testRequest(
    req: GenerateRequest,
    options?: Parameters<typeof simulateSystemPrompt>[0]
  ) {
    return new Promise((resolve, reject) => {
      simulateSystemPrompt(options)(req, resolve as any);
    });
  }

  it('does not modify a request with no system prompt', async () => {
    const req: GenerateRequest = {
      messages: [{ role: 'user', content: [{ text: 'hello' }] }],
    };
    assert.deepEqual(await testRequest(req), req);
  });

  it('keeps other messages in place', async () => {
    const req: GenerateRequest = {
      messages: [
        { role: 'system', content: [{ text: 'I am a system message' }] },
        { role: 'user', content: [{ text: 'hello' }] },
      ],
    };
    assert.deepEqual(await testRequest(req), {
      messages: [
        {
          role: 'user',
          content: [
            { text: 'SYSTEM INSTRUCTIONS:\n' },
            { text: 'I am a system message' },
          ],
        },
        {
          role: 'model',
          content: [{ text: 'Understood.' }],
        },
        {
          role: 'user',
          content: [{ text: 'hello' }],
        },
      ],
    });
  });
});

describe('augmentWithContext', () => {
  async function testRequest(
    messages: MessageData[],
    context?: DocumentData[],
    options?: AugmentWithContextOptions
  ) {
    const changedRequest = await new Promise<GenerateRequest>(
      (resolve, reject) => {
        augmentWithContext(options)(
          {
            messages,
            docs: context,
          },
          resolve as any
        );
      }
    );
    return changedRequest.messages;
  }

  it('should not change a message with empty context', async () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'first part' }] },
    ];
    assert.deepEqual(await testRequest(messages, undefined), messages);
    assert.deepEqual(await testRequest(messages, []), messages);
  });

  it('should not change a message that already has a context part with content', async () => {
    const messages: MessageData[] = [
      {
        role: 'user',
        content: [{ text: 'first part', metadata: { purpose: 'context' } }],
      },
    ];
    assert.deepEqual(
      await testRequest(messages, [{ content: [{ text: 'i am context' }] }]),
      messages
    );
  });

  it('should augment a message that has a pending context part', async () => {
    const messages: MessageData[] = [
      {
        role: 'user',
        content: [{ metadata: { purpose: 'context', pending: true } }],
      },
    ];
    assert.deepEqual(
      await testRequest(messages, [{ content: [{ text: 'i am context' }] }]),
      [
        {
          content: [
            {
              metadata: {
                purpose: 'context',
              },
              text: `${CONTEXT_PREFACE}- [0]: i am context\n\n`,
            },
          ],
          role: 'user',
        },
      ]
    );
  });

  it('should append a new text part', async () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'first part' }] },
    ];
    const result = await testRequest(messages, [
      { content: [{ text: 'i am context' }] },
      { content: [{ text: 'i am more context' }] },
    ]);
    assert.deepEqual(result[0].content.at(-1), {
      text: `${CONTEXT_PREFACE}- [0]: i am context\n- [1]: i am more context\n\n`,
      metadata: { purpose: 'context' },
    });
  });

  it('should append to the last user message', async () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'first part' }] },
      {
        role: 'tool',
        content: [{ toolResponse: { name: 'testTool', output: { abc: 123 } } }],
      },
    ];
    const result = await testRequest(messages, [
      { content: [{ text: 'i am context' }] },
      { content: [{ text: 'i am more context' }] },
    ]);
    assert.deepEqual(result[0].content.at(-1), {
      text: `${CONTEXT_PREFACE}- [0]: i am context\n- [1]: i am more context\n\n`,
      metadata: { purpose: 'context' },
    });
  });

  it('should use a custom preface', async () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'first part' }] },
    ];
    const result = await testRequest(
      messages,
      [
        { content: [{ text: 'i am context' }] },
        { content: [{ text: 'i am more context' }] },
      ],
      { preface: '\n\nCheck this out:\n\n' }
    );
    assert.deepEqual(result[0].content.at(-1), {
      text: '\n\nCheck this out:\n\n- [0]: i am context\n- [1]: i am more context\n\n',
      metadata: { purpose: 'context' },
    });
  });

  it('should elide a null preface', async () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'first part' }] },
    ];
    const result = await testRequest(
      messages,
      [
        { content: [{ text: 'i am context' }] },
        { content: [{ text: 'i am more context' }] },
      ],
      { preface: null }
    );
    assert.deepEqual(result[0].content.at(-1), {
      text: '- [0]: i am context\n- [1]: i am more context\n\n',
      metadata: { purpose: 'context' },
    });
  });

  it('should use a citationKey', async () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'first part' }] },
    ];
    const result = await testRequest(
      messages,
      [
        { content: [{ text: 'i am context' }], metadata: { uid: 'first' } },
        {
          content: [{ text: 'i am more context' }],
          metadata: { uid: 'second' },
        },
      ],
      { citationKey: 'uid' }
    );
    assert.deepEqual(result[0].content.at(-1), {
      text: `${CONTEXT_PREFACE}- [first]: i am context\n- [second]: i am more context\n\n`,
      metadata: { purpose: 'context' },
    });
  });

  it('should use "ref", "id", and index, in that order, if citationKey is unspecified', async () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'first part' }] },
    ];
    const result = await testRequest(messages, [
      {
        content: [{ text: 'i am context' }],
        metadata: { ref: 'first', id: 'wrong' },
      },
      {
        content: [{ text: 'i am more context' }],
        metadata: { id: 'second' },
      },
      {
        content: [{ text: 'i am even more context' }],
      },
    ]);
    assert.deepEqual(result[0].content.at(-1), {
      text: `${CONTEXT_PREFACE}- [first]: i am context\n- [second]: i am more context\n- [2]: i am even more context\n\n`,
      metadata: { purpose: 'context' },
    });
  });

  it('should use a custom itemTemplate', async () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'first part' }] },
    ];
    const result = await testRequest(
      messages,
      [
        { content: [{ text: 'i am context' }], metadata: { uid: 'first' } },
        {
          content: [{ text: 'i am more context' }],
          metadata: { uid: 'second' },
        },
      ],
      { itemTemplate: (d) => `* (${d.metadata!.uid}) -- ${d.text}\n` }
    );
    assert.deepEqual(result[0].content.at(-1), {
      text: `${CONTEXT_PREFACE}* (first) -- i am context\n* (second) -- i am more context\n\n`,
      metadata: { purpose: 'context' },
    });
  });
});

describe.only('simulateConstrainedGeneration', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
    configureFormats(registry);
  });

  it('injects the instructions into the request', async () => {
    const pm = defineProgrammableModel(registry);
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [{ text: '```\n{"foo": "bar"}\n```' }],
        },
      };
    };

    const { output } = await generate(registry, {
      model: 'programmableModel',
      prompt: 'generate json',
      output: {
        schema: z.object({
          foo: z.string(),
        }),
        format: 'json',
      },
    });
    assert.deepEqual(output, { foo: 'bar' });
    assert.deepStrictEqual(pm.lastRequest, {
      config: {},
      messages: [
        {
          role: 'user',
          content: [
            { text: 'generate json' },
            {
              metadata: {
                purpose: 'output',
              },
              text:
                'Output should be in JSON format and conform to the following schema:\n' +
                '\n' +
                '```\n' +
                '{"type":"object","properties":{"foo":{"type":"string"}},"required":["foo"],"additionalProperties":true,"$schema":"http://json-schema.org/draft-07/schema#"}\n' +
                '```\n',
            },
          ],
        },
      ],
      output: {
        constrained: false,
      },
      tools: [],
    });
  });

  it('injects the instructions into the request idempotently', async () => {
    const pm = defineProgrammableModel(registry);
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [{ text: '```\n{"foo": "bar"}\n```' }],
        },
      };
    };

    const { output } = await generate(registry, {
      model: 'programmableModel',
      prompt: 'generate json',
      use: [
        simulateConstrainedGeneration({
          instructionsRenderer: (schema) =>
            `must be json: ${JSON.stringify(schema)}`,
        }),
      ],
      output: {
        schema: z.object({
          foo: z.string(),
        }),
        format: 'json',
      },
    });
    assert.deepEqual(output, { foo: 'bar' });
    assert.deepStrictEqual(pm.lastRequest, {
      config: {},
      messages: [
        {
          role: 'user',
          content: [
            { text: 'generate json' },
            {
              metadata: {
                purpose: 'output',
              },
              text: 'must be json: {"type":"object","properties":{"foo":{"type":"string"}},"required":["foo"],"additionalProperties":true,"$schema":"http://json-schema.org/draft-07/schema#"}',
            },
          ],
        },
      ],
      output: {
        constrained: false,
      },
      tools: [],
    });
  });

  it('relies on native support -- no instructions', async () => {
    const pm = defineProgrammableModel(registry, {
      supports: { constrained: 'all' },
    });
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [{ text: '```\n{"foo": "bar"}\n```' }],
        },
      };
    };

    const { output } = await generate(registry, {
      model: 'programmableModel',
      prompt: 'generate json',
      output: {
        schema: z.object({
          foo: z.string(),
        }),
      },
    });
    assert.deepEqual(output, { foo: 'bar' });
    assert.deepStrictEqual(pm.lastRequest, {
      config: {},
      messages: [
        {
          role: 'user',
          content: [{ text: 'generate json' }],
        },
      ],
      output: {
        constrained: true,
        contentType: 'application/json',
        format: 'json',
        schema: {
          $schema: 'http://json-schema.org/draft-07/schema#',
          additionalProperties: true,
          properties: {
            foo: {
              type: 'string',
            },
          },
          required: ['foo'],
          type: 'object',
        },
      },
      tools: [],
    });
  });

  it('uses format instructions when instructions is explicitly set to true', async () => {
    const pm = defineProgrammableModel(registry, {
      supports: { constrained: 'all' },
    });
    pm.handleResponse = async (req, sc) => {
      return {
        message: {
          role: 'model',
          content: [{ text: '```\n{"foo": "bar"}\n```' }],
        },
      };
    };

    const { output } = await generate(registry, {
      model: 'programmableModel',
      prompt: 'generate json',
      output: {
        instructions: true,
        constrained: false,
        schema: z.object({
          foo: z.string(),
        }),
      },
    });
    assert.deepEqual(output, { foo: 'bar' });
    assert.deepStrictEqual(pm.lastRequest, {
      config: {},
      messages: [
        {
          role: 'user',
          content: [
            { text: 'generate json' },
            {
              metadata: {
                purpose: 'output',
              },
              text:
                'Output should be in JSON format and conform to the following schema:\n' +
                '\n' +
                '```\n' +
                '{"type":"object","properties":{"foo":{"type":"string"}},"required":["foo"],"additionalProperties":true,"$schema":"http://json-schema.org/draft-07/schema#"}\n' +
                '```\n',
            },
          ],
        },
      ],
      output: {
        constrained: false,
        contentType: 'application/json',
        format: 'json',
        schema: {
          $schema: 'http://json-schema.org/draft-07/schema#',
          additionalProperties: true,
          properties: {
            foo: {
              type: 'string',
            },
          },
          required: ['foo'],
          type: 'object',
        },
      },
      tools: [],
    });
  });
});
