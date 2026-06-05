/**
 * Copyright 2025 Google LLC
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

import * as assert from 'assert';
import Fastify, { type FastifyInstance } from 'fastify';
import {
  UserFacingError,
  genkit,
  z,
  type GenerateResponseData,
  type Genkit,
} from 'genkit';
import { InMemoryStreamManager } from 'genkit/beta';
import { runFlow, streamFlow } from 'genkit/beta/client';
import type { ContextProvider, RequestData } from 'genkit/context';
import type { GenerateResponseChunkData, ModelAction } from 'genkit/model';
import getPort from 'get-port';
import { afterEach, beforeEach, describe, it } from 'node:test';
import {
  fastifyHandler,
  genkitFastify,
  withFlowOptions,
} from '../src/index.js';

interface Context {
  auth: {
    user: string;
  };
}

const contextProvider: ContextProvider<Context> = (req: RequestData) => {
  assert.ok(req.method, 'method must be set');
  assert.ok(req.headers, 'headers must be set');
  assert.ok(req.input, 'input must be set');

  if (req.headers['authorization'] !== 'open sesame') {
    throw new UserFacingError('PERMISSION_DENIED', 'not authorized');
  }
  return {
    auth: {
      user: 'Ali Baba',
    },
  };
};

describe('fastifyHandler', async () => {
  let app: FastifyInstance;
  let port: number;

  beforeEach(async () => {
    const ai = genkit({});
    const echoModel = defineEchoModel(ai);

    const voidInput = ai.defineFlow('voidInput', async () => {
      return 'banana';
    });

    const stringInput = ai.defineFlow('stringInput', async (input) => {
      const { text } = await ai.generate({
        model: 'echoModel',
        prompt: input,
      });
      return text;
    });

    const objectInput = ai.defineFlow(
      { name: 'objectInput', inputSchema: z.object({ question: z.string() }) },
      async (input) => {
        const { text } = await ai.generate({
          model: 'echoModel',
          prompt: input.question,
        });
        return text;
      }
    );

    const streamingFlow = ai.defineFlow(
      {
        name: 'streamingFlow',
        inputSchema: z.object({ question: z.string() }),
      },
      async (input, sendChunk) => {
        const { text } = await ai.generate({
          model: 'echoModel',
          prompt: input.question,
          onChunk: sendChunk,
        });
        return text;
      }
    );

    const flowWithAuth = ai.defineFlow(
      {
        name: 'flowWithAuth',
        inputSchema: z.object({ question: z.string() }),
      },
      async (input, { context }) => {
        return `${input.question} - ${JSON.stringify(context!.auth)}`;
      }
    );

    app = Fastify();
    port = await getPort();

    app.post('/voidInput', fastifyHandler(voidInput));
    app.post('/stringInput', fastifyHandler(stringInput));
    app.post('/objectInput', fastifyHandler(objectInput));
    app.post('/streamingFlow', fastifyHandler(streamingFlow));
    app.post(
      '/streamingFlowDurable',
      fastifyHandler(streamingFlow, {
        streamManager: new InMemoryStreamManager(),
      })
    );
    app.post('/flowWithAuth', fastifyHandler(flowWithAuth, { contextProvider }));
    // Can also expose any action.
    app.post('/echoModel', fastifyHandler(echoModel));
    app.post(
      '/echoModelWithAuth',
      fastifyHandler(echoModel, { contextProvider })
    );

    await app.listen({ port });
  });

  afterEach(async () => {
    await app.close();
  });

  describe('runFlow', () => {
    it('should call a void input flow', async () => {
      const result = await runFlow({
        url: `http://localhost:${port}/voidInput`,
      });
      assert.strictEqual(result, 'banana');
    });

    it('should run a flow with string input', async () => {
      const result = await runFlow({
        url: `http://localhost:${port}/stringInput`,
        input: 'hello',
      });
      assert.strictEqual(result, 'Echo: hello');
    });

    it('should run a flow with object input', async () => {
      const result = await runFlow({
        url: `http://localhost:${port}/objectInput`,
        input: {
          question: 'olleh',
        },
      });
      assert.strictEqual(result, 'Echo: olleh');
    });

    it('should fail a bad input', async () => {
      const result = runFlow({
        url: `http://localhost:${port}/objectInput`,
        input: {
          badField: 'hello',
        },
      });
      await assert.rejects(result, (err: Error) => {
        return err.message.includes('INVALID_ARGUMENT');
      });
    });

    it('should call a flow with auth', async () => {
      const result = await runFlow<string>({
        url: `http://localhost:${port}/flowWithAuth`,
        input: {
          question: 'hello',
        },
        headers: {
          Authorization: 'open sesame',
        },
      });
      assert.strictEqual(result, 'hello - {"user":"Ali Baba"}');
    });

    it('should fail a flow with bad auth', async () => {
      const result = runFlow({
        url: `http://localhost:${port}/flowWithAuth`,
        input: {
          question: 'hello',
        },
        headers: {
          Authorization: 'thief #24',
        },
      });
      await assert.rejects(result, (err) => {
        return (err as Error).message.includes('not authorized');
      });
    });

    it('should call a model', async () => {
      const result = await runFlow({
        url: `http://localhost:${port}/echoModel`,
        input: {
          messages: [{ role: 'user', content: [{ text: 'hello' }] }],
        },
      });
      assert.strictEqual(result.finishReason, 'stop');
      assert.deepStrictEqual(result.message, {
        role: 'model',
        content: [{ text: 'Echo: hello' }],
      });
    });

    it('should call a model with auth', async () => {
      const result = await runFlow<GenerateResponseData>({
        url: `http://localhost:${port}/echoModelWithAuth`,
        input: {
          messages: [{ role: 'user', content: [{ text: 'hello' }] }],
        },
        headers: {
          Authorization: 'open sesame',
        },
      });
      assert.strictEqual(result.finishReason, 'stop');
      assert.deepStrictEqual(result.message, {
        role: 'model',
        content: [{ text: 'Echo: hello' }],
      });
    });

    it('should fail a model with bad auth', async () => {
      const result = runFlow({
        url: `http://localhost:${port}/echoModelWithAuth`,
        input: {
          messages: [{ role: 'user', content: [{ text: 'hello' }] }],
        },
        headers: {
          Authorization: 'thief #24',
        },
      });
      await assert.rejects(result, (err) => {
        return (err as Error).message.includes('not authorized');
      });
    });
  });

  describe('streamFlow', () => {
    it('stream a flow', async () => {
      const result = streamFlow<string, GenerateResponseChunkData>({
        url: `http://localhost:${port}/streamingFlow`,
        input: {
          question: 'olleh',
        },
      });

      const gotChunks: GenerateResponseChunkData[] = [];
      for await (const chunk of result.stream) {
        gotChunks.push(chunk);
      }

      assert.deepStrictEqual(gotChunks, [
        { index: 0, role: 'model', content: [{ text: '3' }] },
        { index: 0, role: 'model', content: [{ text: '2' }] },
        { index: 0, role: 'model', content: [{ text: '1' }] },
      ]);

      assert.strictEqual(await result.output, 'Echo: olleh');
    });

    it('should create and subscribe to a durable stream', async () => {
      const result = streamFlow({
        url: `http://localhost:${port}/streamingFlowDurable`,
        input: {
          question: 'durable',
        },
      });

      const streamId = await result.streamId;
      assert.ok(streamId);

      const subscription = streamFlow({
        url: `http://localhost:${port}/streamingFlowDurable`,
        input: {
          question: 'durable',
        },
        streamId: streamId!,
      });

      const gotChunks: GenerateResponseChunkData[] = [];
      for await (const chunk of subscription.stream) {
        gotChunks.push(chunk);
      }

      const originalChunks: GenerateResponseChunkData[] = [];
      for await (const chunk of result.stream) {
        originalChunks.push(chunk);
      }

      assert.deepStrictEqual(gotChunks, originalChunks);
      assert.strictEqual(await subscription.output, 'Echo: durable');
      assert.strictEqual(await result.output, 'Echo: durable');
    });

    it('should return 204 for a non-existent stream', async () => {
      try {
        const result = streamFlow({
          url: `http://localhost:${port}/streamingFlowDurable`,
          input: {
            question: 'durable',
          },
          streamId: 'non-existent-stream-id',
        });
        for await (const _ of result.stream) {
        }
        assert.fail('should have thrown');
      } catch (err: any) {
        assert.strictEqual(err.message, 'NOT_FOUND: Stream not found.');
      }
    });

    it('stream a model', async () => {
      const result = streamFlow({
        url: `http://localhost:${port}/echoModel`,
        input: {
          messages: [{ role: 'user', content: [{ text: 'olleh' }] }],
        },
      });

      const gotChunks: any[] = [];
      for await (const chunk of result.stream) {
        gotChunks.push(chunk);
      }

      const output = await result.output;
      assert.strictEqual(output.finishReason, 'stop');
      assert.deepStrictEqual(output.message, {
        role: 'model',
        content: [{ text: 'Echo: olleh' }],
      });

      assert.deepStrictEqual(gotChunks, [
        { content: [{ text: '3' }] },
        { content: [{ text: '2' }] },
        { content: [{ text: '1' }] },
      ]);
    });
  });
});

describe('genkitFastify', async () => {
  let app: FastifyInstance;
  let port: number;

  beforeEach(async () => {
    const ai = genkit({});
    defineEchoModel(ai);

    const voidInput = ai.defineFlow('voidInput', async () => {
      return 'banana';
    });

    const stringInput = ai.defineFlow('stringInput', async (input) => {
      const { text } = await ai.generate({
        model: 'echoModel',
        prompt: input,
      });
      return text;
    });

    const streamingFlow = ai.defineFlow(
      {
        name: 'streamingFlow',
        inputSchema: z.object({ question: z.string() }),
      },
      async (input, sendChunk) => {
        const { text } = await ai.generate({
          model: 'echoModel',
          prompt: input.question,
          onChunk: sendChunk,
        });
        return text;
      }
    );

    const flowWithAuth = ai.defineFlow(
      {
        name: 'flowWithAuth',
        inputSchema: z.object({ question: z.string() }),
      },
      async (input, { context }) => {
        return `${input.question} - ${JSON.stringify(context!.auth)}`;
      }
    );

    app = Fastify();
    port = await getPort();

    await app.register(genkitFastify, {
      pathPrefix: '/api',
      flows: [
        voidInput,
        stringInput,
        streamingFlow,
        withFlowOptions(flowWithAuth, { contextProvider }),
      ],
    });

    await app.listen({ port });
  });

  afterEach(async () => {
    await app.close();
  });

  it('should call a void input flow at the prefixed path', async () => {
    const result = await runFlow({
      url: `http://localhost:${port}/api/voidInput`,
    });
    assert.strictEqual(result, 'banana');
  });

  it('should run a flow with string input', async () => {
    const result = await runFlow({
      url: `http://localhost:${port}/api/stringInput`,
      input: 'hello',
    });
    assert.strictEqual(result, 'Echo: hello');
  });

  it('should call a flow with auth', async () => {
    const result = await runFlow<string>({
      url: `http://localhost:${port}/api/flowWithAuth`,
      input: { question: 'hello' },
      headers: { Authorization: 'open sesame' },
    });
    assert.strictEqual(result, 'hello - {"user":"Ali Baba"}');
  });

  it('should fail a flow with bad auth', async () => {
    const result = runFlow({
      url: `http://localhost:${port}/api/flowWithAuth`,
      input: { question: 'hello' },
      headers: { Authorization: 'thief #24' },
    });
    await assert.rejects(result, (err) => {
      return (err as Error).message.includes('not authorized');
    });
  });

  it('stream a flow', async () => {
    const result = streamFlow<string, GenerateResponseChunkData>({
      url: `http://localhost:${port}/api/streamingFlow`,
      input: { question: 'olleh' },
    });

    const gotChunks: GenerateResponseChunkData[] = [];
    for await (const chunk of result.stream) {
      gotChunks.push(chunk);
    }

    assert.deepStrictEqual(gotChunks, [
      { index: 0, role: 'model', content: [{ text: '3' }] },
      { index: 0, role: 'model', content: [{ text: '2' }] },
      { index: 0, role: 'model', content: [{ text: '1' }] },
    ]);

    assert.strictEqual(await result.output, 'Echo: olleh');
  });
});

export function defineEchoModel(ai: Genkit): ModelAction {
  return ai.defineModel(
    {
      name: 'echoModel',
    },
    async (request, streamingCallback) => {
      streamingCallback?.({ content: [{ text: '3' }] });
      streamingCallback?.({ content: [{ text: '2' }] });
      streamingCallback?.({ content: [{ text: '1' }] });
      return {
        message: {
          role: 'model',
          content: [
            {
              text:
                'Echo: ' +
                request.messages
                  .map(
                    (m) =>
                      (m.role === 'user' || m.role === 'model'
                        ? ''
                        : `${m.role}: `) + m.content.map((c) => c.text).join()
                  )
                  .join(),
            },
          ],
        },
        finishReason: 'stop',
      };
    }
  );
}
