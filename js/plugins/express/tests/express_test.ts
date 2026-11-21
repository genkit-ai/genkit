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

import * as assert from 'assert';
import express from 'express';
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
import type * as http from 'http';
import { afterEach, beforeEach, describe, it } from 'node:test';
import {
  expressHandler,
  startFlowServer,
  withFlowOptions,
  type FlowServer,
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

// Takes long enough for a test client to disconnect while the handler is still
// waiting for the context.
const slowContextProvider: ContextProvider<Context> = async () => {
  await new Promise((resolve) => setTimeout(resolve, 300));
  return { auth: { user: 'Ali Baba' } };
};

describe('expressHandler', async () => {
  let server: http.Server;
  let port;
  let flowStarted: Promise<void>;
  let resolveFlowStarted: () => void;
  let flowFinished: Promise<{ aborted: boolean; signal: AbortSignal }>;
  let resolveFlowFinished: (v: {
    aborted: boolean;
    signal: AbortSignal;
  }) => void;

  beforeEach(async () => {
    flowStarted = new Promise<void>((r) => {
      resolveFlowStarted = r;
    });
    flowFinished = new Promise((r) => {
      resolveFlowFinished = r;
    });
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

    // Reports when it starts and what its abortSignal looked like when it
    // finished. With the input 'wait' it holds the request open until the client
    // disconnects (the timer is only a safety net so that a regression fails the
    // test instead of hanging it).
    const abortableFlow = ai.defineFlow(
      {
        name: 'abortableFlow',
        inputSchema: z.string(),
      },
      async (input, { abortSignal }) => {
        resolveFlowStarted();
        if (input === 'wait') {
          await new Promise<void>((resolve) => {
            if (abortSignal.aborted) {
              resolve();
              return;
            }
            const timer = setTimeout(resolve, 3000);
            abortSignal.addEventListener(
              'abort',
              () => {
                clearTimeout(timer);
                resolve();
              },
              { once: true }
            );
          });
        }
        resolveFlowFinished({
          aborted: abortSignal.aborted,
          signal: abortSignal,
        });
        return 'done';
      }
    );

    const app = express();
    app.use(express.json());
    port = await getPort();

    app.post('/voidInput', expressHandler(voidInput));
    app.post('/stringInput', expressHandler(stringInput));
    app.post('/objectInput', expressHandler(objectInput));
    app.post('/streamingFlow', expressHandler(streamingFlow));
    app.post(
      '/streamingFlowDurable',
      expressHandler(streamingFlow, {
        streamManager: new InMemoryStreamManager(),
      })
    );
    app.post(
      '/flowWithAuth',
      expressHandler(flowWithAuth, { contextProvider })
    );
    // Can also expose any action.
    app.post('/echoModel', expressHandler(echoModel));
    app.post(
      '/echoModelWithAuth',
      expressHandler(echoModel, { contextProvider })
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

    app.post('/flowWithInit', expressHandler(flowWithInit));
    app.post('/flowWithInitSchema', expressHandler(flowWithInitSchema));
    app.post('/abortableFlow', expressHandler(abortableFlow));
    app.post(
      '/abortableFlowSlowContext',
      expressHandler(abortableFlow, { contextProvider: slowContextProvider })
    );

    server = app.listen(port, () => {
      console.log(`Example app listening on port ${port}`);
    });
  });

  afterEach(() => {
    server.close();
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

    it('should fail a flow with auth', async () => {
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

    it('should fail a flow with auth', async () => {
      const result = runFlow({
        url: `http://localhost:${port}/echoModelWithAuth`,
        input: {
          messages: [
            {
              role: 'user',
              content: [{ text: 'hello' }],
            },
          ],
        },
        headers: {
          Authorization: 'thief #24',
        },
      });
      await assert.rejects(result, (err) => {
        return (err as Error).message.includes('not authorized');
      });
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

    it('aborts the flow when the client disconnects', async () => {
      const controller = new AbortController();
      const response = fetch(`http://localhost:${port}/abortableFlow`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: 'wait' }),
        signal: controller.signal,
      });
      await flowStarted;
      controller.abort();
      await assert.rejects(response); // abort error

      const { aborted } = await flowFinished;
      assert.strictEqual(aborted, true);
    });

    it('aborts the flow when the client disconnects while the context is being resolved', async () => {
      const controller = new AbortController();
      const response = fetch(
        `http://localhost:${port}/abortableFlowSlowContext`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: 'wait' }),
          signal: controller.signal,
        }
      );
      // Disconnect while the slow context provider is still running, so the
      // flow has not started yet.
      await new Promise((resolve) => setTimeout(resolve, 100));
      controller.abort();
      await assert.rejects(response); // abort error

      const { aborted } = await flowFinished;
      assert.strictEqual(aborted, true);
    });

    it('does not abort the flow for a request that completes', async () => {
      const result = await runFlow({
        url: `http://localhost:${port}/abortableFlow`,
        input: 'quick',
      });
      assert.strictEqual(result, 'done');

      const { aborted, signal } = await flowFinished;
      assert.strictEqual(aborted, false);
      // Give the connection a chance to close; that must not abort the flow.
      await new Promise((r) => setTimeout(r, 50));
      assert.strictEqual(signal.aborted, false);
    });

    it('should abort a flow when the client hangs up', async () => {
      const controller = new AbortController();
      // A JSON body matters: the request stream closes once body parsing has
      // read it, so only the response can report the client going away.
      const response = runFlow({
        url: `http://localhost:${port}/abortableFlow`,
        input: 'wait',
        abortSignal: controller.signal,
      });

      await flowStarted;
      controller.abort();

      await assert.rejects(response); // abort error

      const { aborted } = await flowFinished;
      assert.strictEqual(aborted, true);
    });
  });

  describe('streamFlow', () => {
    it('aborts the flow when a streaming client disconnects', async () => {
      const controller = new AbortController();
      const response = fetch(`http://localhost:${port}/abortableFlow`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ data: 'wait' }),
        signal: controller.signal,
      });
      await flowStarted;
      controller.abort();
      await assert.rejects(response); // abort error

      const { aborted } = await flowFinished;
      assert.strictEqual(aborted, true);
    });

    it('does not abort the flow for a streaming request that completes', async () => {
      const { output } = streamFlow<string, string>({
        url: `http://localhost:${port}/abortableFlow`,
        input: 'quick',
      });
      assert.strictEqual(await output, 'done');

      const { aborted, signal } = await flowFinished;
      assert.strictEqual(aborted, false);
      await new Promise((r) => setTimeout(r, 50));
      assert.strictEqual(signal.aborted, false);
    });

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

      // of note here is that we're consuming the original stream after re-subscription
      // which should still work fine.
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

    it('stream a model', async () => {
      const result = streamFlow({
        url: `http://localhost:${port}/echoModel`,
        input: {
          messages: [
            {
              role: 'user',
              content: [{ text: 'olleh' }],
            },
          ],
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

describe('startFlowServer', async () => {
  let server: FlowServer;
  let port;

  beforeEach(async () => {
    const ai = genkit({});
    defineEchoModel(ai);
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

    port = await getPort();

    server = startFlowServer({
      flows: [
        voidInput,
        stringInput,
        objectInput,
        streamingFlow,
        withFlowOptions(streamingFlow, {
          streamManager: new InMemoryStreamManager(),
          path: 'streamingFlowDurable',
        }),
        streamingFlowV2,
        withFlowOptions(flowWithAuth, { contextProvider }),
      ],
      port,
    });
  });

  afterEach(() => {
    server.stop();
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

    it('should fail a flow with auth', async () => {
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

      // of note here is that we're consuming the original stream after re-subscription
      // which should still work fine.
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
});

export function defineEchoModel(ai: Genkit): ModelAction {
  return ai.defineModel(
    {
      name: 'echoModel',
    },
    async (request, streamingCallback) => {
      streamingCallback?.({
        content: [
          {
            text: '3',
          },
        ],
      });
      streamingCallback?.({
        content: [
          {
            text: '2',
          },
        ],
      });
      streamingCallback?.({
        content: [
          {
            text: '1',
          },
        ],
      });
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
      sendChunk({
        content: [
          {
            text: '3',
          },
        ],
      });
      sendChunk({
        content: [
          {
            text: '2',
          },
        ],
      });
      sendChunk({
        content: [
          {
            text: '1',
          },
        ],
      });
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
