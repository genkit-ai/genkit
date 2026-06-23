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
  let abortableFlowResult: [string, boolean] | undefined;

  beforeEach(async () => {
    abortableFlowResult = undefined;
    const ai = genkit({});
    const echoModel = defineEchoModel(ai);
    defineEchoModelV2(ai);

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

    const streamingFlowV2 = ai.defineFlow(
      {
        name: 'streamingFlowV2',
        inputSchema: z.object({ question: z.string() }),
      },
      async (input, sendChunk) => {
        const { text } = await ai.generate({
          model: 'echoModelV2',
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

    // A flow that echoes back the init data to verify it was received.
    const flowWithInit = ai.defineFlow(
      {
        name: 'flowWithInit',
        inputSchema: z.string(),
      },
      async (input) => {
        return `input: ${input}`;
      }
    );
    // Monkey-patch the run method to capture and return init data.
    const originalRun = flowWithInit.run.bind(flowWithInit);
    flowWithInit.run = async (input: any, options: any) => {
      const result = await originalRun(input, options);
      // Embed init in the result so we can verify it was passed through.
      result.result = `input: ${input}, init: ${JSON.stringify(options?.init)}`;
      return result;
    };

    // A flow with an initSchema to exercise real init validation.
    const flowWithInitSchema = ai.defineFlow(
      {
        name: 'flowWithInitSchema',
        inputSchema: z.string(),
        initSchema: z.object({ sessionId: z.string() }),
      },
      async (input, { init }) =>
        `input: ${input}, sessionId: ${(init as { sessionId: string }).sessionId}`
    );

    const abortableFlow = ai.defineFlow(
      {
        name: 'abortableFlow',
      },
      async (_, { abortSignal, sendChunk }) => {
        let itersLeft = 20;
        while (itersLeft > 0 && !abortSignal.aborted) {
          await new Promise((r) => {
            setTimeout(r, 100);
          });
          sendChunk(itersLeft);
          itersLeft--;
        }
        abortableFlowResult = [
          itersLeft > 0 ? 'success' : 'failure',
          abortSignal.aborted,
        ];
        return itersLeft > 0 ? 'success' : 'failure';
      }
    );

    app = Fastify();
    port = await getPort();

    // Simulates a plugin/hook (e.g. @fastify/cors) that sets response headers
    // before the handler runs. These must survive reply.hijack() when streaming.
    app.addHook('onRequest', async (_req, reply) => {
      reply.header('x-pre-hijack', 'preserved');
    });

    app.post('/voidInput', fastifyHandler(voidInput));
    app.post('/stringInput', fastifyHandler(stringInput));
    app.post('/objectInput', fastifyHandler(objectInput));
    app.post('/streamingFlow', fastifyHandler(streamingFlow));
    app.post('/streamingFlowV2', fastifyHandler(streamingFlowV2));
    app.post(
      '/streamingFlowDurable',
      fastifyHandler(streamingFlow, {
        streamManager: new InMemoryStreamManager(),
      })
    );
    app.post(
      '/flowWithAuth',
      fastifyHandler(flowWithAuth, { contextProvider })
    );
    // Can also expose any action.
    app.post('/echoModel', fastifyHandler(echoModel));
    app.post(
      '/echoModelWithAuth',
      fastifyHandler(echoModel, { contextProvider })
    );
    app.post('/flowWithInit', fastifyHandler(flowWithInit));
    app.post('/flowWithInitSchema', fastifyHandler(flowWithInitSchema));
    app.post('/abortableFlow', fastifyHandler(abortableFlow));
    // A stream manager whose open() rejects, to exercise the error path after
    // reply.hijack() (the response must still be closed, not leaked).
    const throwingStreamManager = {
      open: async () => {
        throw new Error('open failed');
      },
      subscribe: async () => {},
    } as unknown as InMemoryStreamManager;
    app.post(
      '/streamSetupThrows',
      fastifyHandler(streamingFlow, { streamManager: throwingStreamManager })
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

    it('should set x-genkit-trace-id and x-genkit-span-id headers', async () => {
      const response = await fetch(`http://localhost:${port}/voidInput`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: null }),
      });
      assert.strictEqual(response.status, 200);
      assert.ok(response.headers.get('x-genkit-trace-id'));
      assert.ok(response.headers.get('x-genkit-span-id'));
    });

    it('should return 400 when the body is missing', async () => {
      const response = await fetch(`http://localhost:${port}/voidInput`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      assert.strictEqual(response.status, 400);
    });

    it('should pass init data to the action', async () => {
      const result = await runFlow<string>({
        url: `http://localhost:${port}/flowWithInit`,
        input: 'hello',
        init: { sessionId: 'abc123', temperature: 0.7 },
      });
      assert.strictEqual(
        result,
        'input: hello, init: {"sessionId":"abc123","temperature":0.7}'
      );
    });

    it('should pass undefined init when not provided', async () => {
      const result = await runFlow<string>({
        url: `http://localhost:${port}/flowWithInit`,
        input: 'hello',
      });
      assert.strictEqual(result, 'input: hello, init: undefined');
    });

    it('should validate init against initSchema and pass it to the action', async () => {
      const result = await runFlow<string>({
        url: `http://localhost:${port}/flowWithInitSchema`,
        input: 'hello',
        init: { sessionId: 'abc123' },
      });
      assert.strictEqual(result, 'input: hello, sessionId: abc123');
    });

    it('should reject init that does not conform to initSchema', async () => {
      const result = runFlow<string>({
        url: `http://localhost:${port}/flowWithInitSchema`,
        input: 'hello',
        // sessionId should be a string, not a number.
        init: { sessionId: 123 },
      });
      await assert.rejects(result, (err: Error) => {
        return err.message.includes('INVALID_ARGUMENT');
      });
    });

    // TODO: This test is flaky, skipping until fixed (mirrors the express plugin).
    it.skip('should abort a flow', async () => {
      const controller = new AbortController();
      const response = fetch(`http://localhost:${port}/abortableFlow`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ data: null }),
        signal: controller.signal,
      });

      setTimeout(() => controller.abort(), 10);

      await assert.rejects(response); // abort error

      await new Promise((r) => {
        setTimeout(r, 200);
      });

      assert.deepStrictEqual(abortableFlowResult, ['success', true]);
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

    it('should subscribe to a stream in progress', async () => {
      const result = streamFlow({
        url: `http://localhost:${port}/streamingFlowDurable`,
        input: {
          question: 'durable',
        },
      });

      const streamId = await result.streamId;
      assert.ok(streamId);

      // Don't wait for the original stream to finish.
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

      assert.deepStrictEqual(gotChunks.length, 3);
      assert.strictEqual(await subscription.output, 'Echo: durable');
    });

    it('sends streaming headers when subscribing to a durable stream', async () => {
      // The subscribe path must still set Content-Type/Transfer-Encoding so
      // clients and proxies treat the response as a stream.
      const result = streamFlow({
        url: `http://localhost:${port}/streamingFlowDurable`,
        input: { question: 'durable' },
      });
      const streamId = await result.streamId;
      assert.ok(streamId);

      const response = await fetch(
        `http://localhost:${port}/streamingFlowDurable`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            'x-genkit-stream-id': streamId!,
          },
          body: JSON.stringify({ data: { question: 'durable' } }),
        }
      );
      assert.match(response.headers.get('content-type') ?? '', /text\/plain/);
      await response.text(); // drain
      await result.output;
    });

    it('stream a flow (v2 model)', async () => {
      const result = streamFlow<string, GenerateResponseChunkData>({
        url: `http://localhost:${port}/streamingFlowV2`,
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

      assert.strictEqual(await result.output, 'Echo v2: olleh');
    });

    it('preserves headers set by earlier hooks on a streamed response', async () => {
      // A streamed response goes through reply.hijack(); headers set by hooks
      // (like CORS) must still reach the client. Without this, the browser
      // rejects the streamed response even when the preflight passed.
      const response = await fetch(`http://localhost:${port}/streamingFlow`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ data: { question: 'hi' } }),
      });
      assert.strictEqual(response.headers.get('x-pre-hijack'), 'preserved');
      await response.text(); // drain the stream
    });

    it('detects streaming for a multi-value, mixed-case Accept header', async () => {
      // Clients/proxies can send a media-type list and mixed casing, e.g.
      // "Text/Event-Stream, */*"; the handler should still stream rather than
      // fall back to a single JSON response.
      const response = await fetch(`http://localhost:${port}/streamingFlow`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'Text/Event-Stream, */*',
        },
        body: JSON.stringify({ data: { question: 'hi' } }),
      });
      const text = await response.text();
      assert.match(text, /^data: /m); // SSE frames, not a single JSON body
    });

    it('closes the hijacked response if stream setup throws', async () => {
      // streamManager.open() rejects after reply.hijack(); the handler must
      // close the response with an error rather than leak the open socket.
      const response = await fetch(
        `http://localhost:${port}/streamSetupThrows`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
          },
          body: JSON.stringify({ data: { question: 'hi' } }),
        }
      );
      const text = await response.text(); // resolves (no hang)
      assert.match(text, /error:/);
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
        withFlowOptions(stringInput, { path: 'customPath' }),
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

  it('should route to a flow with a custom path option', async () => {
    const result = await runFlow({
      url: `http://localhost:${port}/api/customPath`,
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

describe('genkitFastify path normalization', async () => {
  let app: FastifyInstance;
  let port: number;

  afterEach(async () => {
    await app.close();
  });

  // A prefix without a leading slash or with stray slashes would otherwise
  // throw when Fastify registers the route.
  for (const prefix of ['api', '/api/', 'api//']) {
    it(`normalizes pathPrefix "${prefix}" to a valid route`, async () => {
      const ai = genkit({});
      const voidInput = ai.defineFlow('voidInput', async () => 'banana');

      app = Fastify();
      port = await getPort();
      await app.register(genkitFastify, {
        pathPrefix: prefix,
        flows: [voidInput],
      });
      await app.listen({ port });

      const result = await runFlow({
        url: `http://localhost:${port}/api/voidInput`,
      });
      assert.strictEqual(result, 'banana');
    });
  }
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

export function defineEchoModelV2(ai: Genkit): ModelAction {
  return ai.defineModel(
    {
      apiVersion: 'v2',
      name: 'echoModelV2',
    },
    async (request, { sendChunk }) => {
      sendChunk({ content: [{ text: '3' }] });
      sendChunk({ content: [{ text: '2' }] });
      sendChunk({ content: [{ text: '1' }] });
      return {
        message: {
          role: 'model',
          content: [
            {
              text:
                'Echo v2: ' +
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
