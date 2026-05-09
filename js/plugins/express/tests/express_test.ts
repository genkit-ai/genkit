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

describe('expressHandler', async () => {
  let server: http.Server;
  let port;
  let abortableFlowResult;

  beforeEach(async () => {
    abortableFlowResult = undefined;
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
    app.post('/abortableFlow', expressHandler(abortableFlow));

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

    // TODO: This test is flaky, skipping until fixed.
    it.skip('should abort a flow with auth', async () => {
      const controller = new AbortController();
      const response = fetch(`http://localhost:${port}/abortableFlow`, {
        method: 'POST',
        signal: controller.signal,
      });
      // TODO: make this work instead of direct fetch. For some reason doesn't work in a test,
      // even though appears to be doing exactly the same thing.
      /*
      const response = runFlow({
        url: `http://localhost:${port}/abortableFlow`,
        abortSignal: controller.signal,
      });
      */

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

    it.only('should return 204 for a non-existent stream', async () => {
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
