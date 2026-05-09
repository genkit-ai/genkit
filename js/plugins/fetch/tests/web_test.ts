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
import * as http from 'http';
import { afterEach, beforeEach, describe, it } from 'node:test';
import {
  fetchHandler,
  fetchHandlers,
  type ActionWithOptions,
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

/**
 * Collects the request body from a Node.js IncomingMessage.
 */
function getRequestBody(req: http.IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

/**
 * Creates a Web API Request from a Node.js IncomingMessage and body.
 */
function nodeRequestToWebRequest(
  req: http.IncomingMessage,
  baseUrl: string,
  body: Buffer
): Request {
  const url = new URL(req.url || '/', baseUrl);
  return new Request(url.toString(), {
    method: req.method || 'GET',
    headers: req.headers as HeadersInit,
    body: body.length > 0 ? new Uint8Array(body) : undefined,
  });
}

/**
 * Writes a Web API Response to a Node.js ServerResponse.
 */
async function writeWebResponseToNode(
  response: Response,
  res: http.ServerResponse
): Promise<void> {
  const headers: Record<string, string> = {};
  response.headers.forEach((value, key) => {
    headers[key] = value;
  });
  res.writeHead(response.status, headers);
  if (response.body) {
    const reader = response.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
    }
  }
  res.end();
}

describe('fetchHandler', () => {
  it('returns a handler that runs the action', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow('testFlow', async () => 'ok');
    const handler = fetchHandler(flow);
    const request = new Request('http://localhost/testFlow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: null }),
    });
    const response = await handler(request);
    assert.strictEqual(response.status, 200);
    const json = (await response.json()) as { result: string };
    assert.strictEqual(json.result, 'ok');
  });

  it('accepts options (contextProvider)', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow(
      {
        name: 'authFlow',
        inputSchema: z.object({ question: z.string() }),
      },
      async () => 'ok'
    );
    const handler = fetchHandler(flow, { contextProvider });
    const request = new Request('http://localhost/authFlow', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'open sesame',
      },
      body: JSON.stringify({ data: { question: 'hello' } }),
    });
    const response = await handler(request);
    assert.strictEqual(response.status, 200);
  });
});

describe('fetchHandler (single action)', () => {
  describe('direct Request/Response (no server)', () => {
    it('should return 400 when body is not JSON', async () => {
      const ai = genkit({});
      const flow = ai.defineFlow('test', async () => 'ok');
      const request = new Request('http://localhost/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: 'not json',
      });
      const response = await fetchHandler(flow)(request);
      assert.strictEqual(response.status, 400);
      const json = (await response.json()) as { status: string };
      assert.strictEqual(json.status, 'INVALID_ARGUMENT');
    });

    it('should return 400 when body has no data field', async () => {
      const ai = genkit({});
      const flow = ai.defineFlow('test', async () => 'ok');
      const request = new Request('http://localhost/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ foo: 'bar' }),
      });
      const response = await fetchHandler(flow)(request);
      assert.strictEqual(response.status, 400);
      const json = (await response.json()) as { status: string };
      assert.strictEqual(json.status, 'INVALID_ARGUMENT');
    });

    it('should run a void input flow', async () => {
      const ai = genkit({});
      const flow = ai.defineFlow('voidInput', async () => 'banana');
      const request = new Request('http://localhost/voidInput', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: null }),
      });
      const response = await fetchHandler(flow)(request);
      assert.strictEqual(response.status, 200);
      const json = (await response.json()) as { result: string };
      assert.strictEqual(json.result, 'banana');
    });

    it('should run a flow with string input', async () => {
      const ai = genkit({});
      defineEchoModel(ai);
      const flow = ai.defineFlow('stringInput', async (input: string) => {
        const { text } = await ai.generate({
          model: 'echoModel',
          prompt: input,
        });
        return text;
      });
      const request = new Request('http://localhost/stringInput', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: 'hello' }),
      });
      const response = await fetchHandler(flow)(request);
      assert.strictEqual(response.status, 200);
      const json = (await response.json()) as { result: string };
      assert.strictEqual(json.result, 'Echo: hello');
    });

    it('should run a flow with object input', async () => {
      const ai = genkit({});
      defineEchoModel(ai);
      const flow = ai.defineFlow(
        {
          name: 'objectInput',
          inputSchema: z.object({ question: z.string() }),
        },
        async (input) => {
          const { text } = await ai.generate({
            model: 'echoModel',
            prompt: input.question,
          });
          return text;
        }
      );
      const request = new Request('http://localhost/objectInput', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: { question: 'olleh' } }),
      });
      const response = await fetchHandler(flow)(request);
      assert.strictEqual(response.status, 200);
      const json = (await response.json()) as { result: string };
      assert.strictEqual(json.result, 'Echo: olleh');
    });

    it('should return error for invalid input', async () => {
      const ai = genkit({});
      defineEchoModel(ai);
      const flow = ai.defineFlow(
        {
          name: 'objectInput',
          inputSchema: z.object({ question: z.string() }),
        },
        async (input) => input.question
      );
      const request = new Request('http://localhost/objectInput', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: { badField: 'hello' } }),
      });
      const response = await fetchHandler(flow)(request);
      assert.strictEqual(response.status, 400);
      const json = (await response.json()) as { status?: string };
      assert.ok(
        json.status === 'INVALID_ARGUMENT' ||
          (json as { message?: string }).message?.includes('INVALID')
      );
    });

    it('should call a flow with auth', async () => {
      const ai = genkit({});
      const flow = ai.defineFlow(
        {
          name: 'flowWithAuth',
          inputSchema: z.object({ question: z.string() }),
        },
        async (input, { context }) => {
          return `${input.question} - ${JSON.stringify(context!.auth)}`;
        }
      );
      const request = new Request('http://localhost/flowWithAuth', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'open sesame',
        },
        body: JSON.stringify({ data: { question: 'hello' } }),
      });
      const response = await fetchHandler(flow, { contextProvider })(request);
      assert.strictEqual(response.status, 200);
      const json = (await response.json()) as { result: string };
      assert.strictEqual(json.result, 'hello - {"user":"Ali Baba"}');
    });

    it('should fail a flow with auth when unauthorized', async () => {
      const ai = genkit({});
      const flow = ai.defineFlow(
        {
          name: 'flowWithAuth',
          inputSchema: z.object({ question: z.string() }),
        },
        async (input, { context }) => String(context?.auth)
      );
      const request = new Request('http://localhost/flowWithAuth', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'thief #24',
        },
        body: JSON.stringify({ data: { question: 'hello' } }),
      });
      const response = await fetchHandler(flow, { contextProvider })(request);
      assert.strictEqual(response.status, 403);
      const json = (await response.json()) as { message?: string };
      assert.ok(json.message?.includes('not authorized'));
    });

    it('should set x-genkit-trace-id and x-genkit-span-id headers', async () => {
      const ai = genkit({});
      const flow = ai.defineFlow('traceFlow', async () => 'ok');
      const request = new Request('http://localhost/traceFlow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: null }),
      });
      const response = await fetchHandler(flow)(request);
      assert.strictEqual(response.status, 200);
      assert.ok(response.headers.get('x-genkit-trace-id'));
      assert.ok(response.headers.get('x-genkit-span-id'));
    });
  });
});

describe('fetchHandlers with HTTP server (runFlow/streamFlow)', () => {
  let server: http.Server;
  let port: number;

  beforeEach(async () => {
    const ai = genkit({});
    const echoModel = defineEchoModel(ai);
    const voidInput = ai.defineFlow('voidInput', async () => 'banana');
    const stringInput = ai.defineFlow('stringInput', async (input: string) => {
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

    port = await getPort();
    server = await createServerWithFlows(
      port,
      voidInput,
      stringInput,
      objectInput,
      streamingFlow,
      flowWithAuth,
      echoModel
    );
  });

  afterEach(() => {
    server.close();
  });

  describe('runFlow', () => {
    it('should call a void input flow', async () => {
      const result = await runFlow({
        url: `http://localhost:${port}/voidInput`,
        input: null,
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
        input: { question: 'olleh' },
      });
      assert.strictEqual(result, 'Echo: olleh');
    });

    it('should fail a bad input', async () => {
      const result = runFlow({
        url: `http://localhost:${port}/objectInput`,
        input: { badField: 'hello' },
      });
      await assert.rejects(result, (err: Error) =>
        err.message.includes('INVALID_ARGUMENT')
      );
    });

    it('should call a flow with auth', async () => {
      const result = await runFlow<string>({
        url: `http://localhost:${port}/flowWithAuth`,
        input: { question: 'hello' },
        headers: { Authorization: 'open sesame' },
      });
      assert.strictEqual(result, 'hello - {"user":"Ali Baba"}');
    });

    it('should fail a flow with auth', async () => {
      const result = runFlow({
        url: `http://localhost:${port}/flowWithAuth`,
        input: { question: 'hello' },
        headers: { Authorization: 'thief #24' },
      });
      await assert.rejects(result, (err: Error) =>
        (err as Error).message.includes('not authorized')
      );
    });

    it('should call a model', async () => {
      const result = await runFlow({
        url: `http://localhost:${port}/echoModel`,
        input: {
          messages: [{ role: 'user', content: [{ text: 'hello' }] }],
        },
      });
      assert.strictEqual((result as GenerateResponseData).finishReason, 'stop');
      assert.deepStrictEqual((result as GenerateResponseData).message, {
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
        headers: { Authorization: 'open sesame' },
      });
      assert.strictEqual(result.finishReason, 'stop');
      assert.deepStrictEqual(result.message, {
        role: 'model',
        content: [{ text: 'Echo: hello' }],
      });
    });

    it('should fail a model with auth when unauthorized', async () => {
      const result = runFlow({
        url: `http://localhost:${port}/echoModelWithAuth`,
        input: {
          messages: [{ role: 'user', content: [{ text: 'hello' }] }],
        },
        headers: { Authorization: 'thief #24' },
      });
      await assert.rejects(result, (err: Error) =>
        (err as Error).message.includes('not authorized')
      );
    });
  });

  describe('streamFlow', () => {
    it('stream a flow', async () => {
      const result = streamFlow<string, GenerateResponseChunkData>({
        url: `http://localhost:${port}/streamingFlow`,
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

    it('should create and subscribe to a durable stream', async () => {
      const result = streamFlow({
        url: `http://localhost:${port}/streamingFlowDurable`,
        input: { question: 'durable' },
      });

      const streamId = await result.streamId;
      assert.ok(streamId);

      const subscription = streamFlow({
        url: `http://localhost:${port}/streamingFlowDurable`,
        input: { question: 'durable' },
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
        input: { question: 'durable' },
      });

      const streamId = await result.streamId;
      assert.ok(streamId);

      const subscription = streamFlow({
        url: `http://localhost:${port}/streamingFlowDurable`,
        input: { question: 'durable' },
        streamId: streamId!,
      });

      const gotChunks: GenerateResponseChunkData[] = [];
      for await (const chunk of subscription.stream) {
        gotChunks.push(chunk);
      }

      assert.strictEqual(gotChunks.length, 3);
      assert.strictEqual(await subscription.output, 'Echo: durable');
    });

    it('should return 204 for a non-existent stream', async () => {
      try {
        const result = streamFlow({
          url: `http://localhost:${port}/streamingFlowDurable`,
          input: { question: 'durable' },
          streamId: 'non-existent-stream-id',
        });
        for await (const _ of result.stream) {
        }
        assert.fail('should have thrown');
      } catch (err: unknown) {
        assert.strictEqual(
          (err as Error).message,
          'NOT_FOUND: Stream not found.'
        );
      }
    });

    it('stream a model', async () => {
      const result = streamFlow({
        url: `http://localhost:${port}/echoModel`,
        input: {
          messages: [{ role: 'user', content: [{ text: 'olleh' }] }],
        },
      });

      const gotChunks: unknown[] = [];
      for await (const chunk of result.stream) {
        gotChunks.push(chunk);
      }

      const output = await result.output;
      assert.strictEqual((output as GenerateResponseData).finishReason, 'stop');
      assert.deepStrictEqual((output as GenerateResponseData).message, {
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

describe('fetchHandlers (path routing)', () => {
  it('returns a handler that routes by path', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow('myFlow', async () => 'ok');
    const handler = fetchHandlers([flow]);
    const request = new Request('http://localhost/myFlow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: null }),
    });
    const response = await handler(request);
    assert.strictEqual(response.status, 200);
    const json = (await response.json()) as { result: string };
    assert.strictEqual(json.result, 'ok');
  });

  it('strips pathPrefix when provided', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow('myFlow', async () => 'ok');
    const handler = fetchHandlers([flow], '/api');
    const request = new Request('http://localhost/api/myFlow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: null }),
    });
    const response = await handler(request);
    assert.strictEqual(response.status, 200);
    const json = (await response.json()) as { result: string };
    assert.strictEqual(json.result, 'ok');
  });

  it('should return 404 when no flow matches path', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow('myFlow', async () => 'ok');
    const request = new Request('http://localhost/otherPath', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: null }),
    });
    const response = await fetchHandlers([flow])(request);
    assert.strictEqual(response.status, 404);
    const json = (await response.json()) as { status: string };
    assert.strictEqual(json.status, 'NOT_FOUND');
  });

  it('should route to flow by name', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow('myFlow', async () => 'ok');
    const request = new Request('http://localhost/myFlow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: null }),
    });
    const response = await fetchHandlers([flow])(request);
    assert.ok(response);
    assert.strictEqual(response!.status, 200);
    const json = (await response!.json()) as { result: string };
    assert.strictEqual(json.result, 'ok');
  });

  it('should route to flow with path option', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow('internalName', async () => 'ok');
    const request = new Request('http://localhost/customPath', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: null }),
    });
    const response = await fetchHandlers([
      { action: flow, options: { path: 'customPath' } },
    ])(request);
    assert.ok(response);
    assert.strictEqual(response!.status, 200);
    const json = (await response!.json()) as { result: string };
    assert.strictEqual(json.result, 'ok');
  });

  it('should strip pathPrefix and route', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow('myFlow', async () => 'ok');
    const request = new Request('http://localhost/api/genkit/myFlow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: null }),
    });
    const response = await fetchHandlers([flow], '/api/genkit')(request);
    assert.ok(response);
    assert.strictEqual(response!.status, 200);
    const json = (await response!.json()) as { result: string };
    assert.strictEqual(json.result, 'ok');
  });

  it('should return 404 when path does not match prefix', async () => {
    const ai = genkit({});
    const flow = ai.defineFlow('myFlow', async () => 'ok');
    const request = new Request('http://localhost/other/myFlow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: null }),
    });
    const response = await fetchHandlers([flow], '/api/genkit')(request);
    assert.strictEqual(response.status, 404);
    const json = (await response.json()) as { status: string };
    assert.strictEqual(json.status, 'NOT_FOUND');
  });
});

function defineEchoModel(ai: Genkit): ModelAction {
  return ai.defineModel(
    { name: 'echoModel' },
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

async function createServerWithFlows(
  port: number,
  voidInput: import('genkit/beta').Flow<any, any, any>,
  stringInput: import('genkit/beta').Flow<any, any, any>,
  objectInput: import('genkit/beta').Flow<any, any, any>,
  streamingFlow: import('genkit/beta').Flow<any, any, any>,
  flowWithAuth: import('genkit/beta').Flow<any, any, any>,
  echoModel: ModelAction
): Promise<http.Server> {
  const actions: (
    | ActionWithOptions<any, any, any>
    | import('genkit/beta').Flow<any, any, any>
  )[] = [
    voidInput,
    stringInput,
    objectInput,
    streamingFlow,
    {
      action: streamingFlow,
      options: {
        streamManager: new InMemoryStreamManager(),
        path: 'streamingFlowDurable',
      },
    },
    { action: flowWithAuth, options: { contextProvider } },
    echoModel,
    {
      action: echoModel,
      options: { contextProvider, path: 'echoModelWithAuth' },
    },
  ];
  return new Promise((resolve) => {
    const baseUrl = `http://localhost:${port}`;
    const server = http.createServer(async (req, res) => {
      try {
        const body = await getRequestBody(req);
        const request = nodeRequestToWebRequest(req, baseUrl, body);
        const response = await fetchHandlers(actions)(request);
        await writeWebResponseToNode(response, res);
      } catch (err) {
        res.writeHead(500);
        res.end(String(err));
      }
    });
    server.listen(port, () => resolve(server));
  });
}
