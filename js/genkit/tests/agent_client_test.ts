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

import * as assert from 'assert';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { AgentError, remoteAgent } from '../src/client/agent';

// ---------------------------------------------------------------------------
// Test transport: a fake `fetch` that drives the streamFlow/runFlow protocol.
// ---------------------------------------------------------------------------

interface RecordedRequest {
  url: string;
  body: any;
  headers: Record<string, string>;
}

/** Builds a streaming (SSE) Response from a list of chunk objects. */
function sseResponse(
  chunks: Array<{ message?: any } | { result?: any } | { error?: any }>
): Response {
  const encoder = new TextEncoder();
  const lines = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('');
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(lines));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

/** Builds a JSON Response (runFlow protocol). */
function jsonResponse(obj: any): Response {
  return new Response(JSON.stringify(obj), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** A scriptable fetch mock. */
class FetchMock {
  requests: RecordedRequest[] = [];
  private handlers: Array<(req: RecordedRequest) => Response> = [];

  /** Queues a response for the next matching request. */
  onNext(handler: (req: RecordedRequest) => Response): void {
    this.handlers.push(handler);
  }

  get fetch() {
    return async (url: any, init: any): Promise<Response> => {
      const req: RecordedRequest = {
        url: String(url),
        body: init?.body ? JSON.parse(init.body) : undefined,
        headers: init?.headers ?? {},
      };
      this.requests.push(req);
      const handler = this.handlers.shift();
      if (!handler) {
        throw new Error(`Unexpected fetch to ${req.url}`);
      }
      return handler(req);
    };
  }
}

describe('remoteAgent', () => {
  let mock: FetchMock;
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    mock = new FetchMock();
    originalFetch = globalThis.fetch;
    globalThis.fetch = mock.fetch as any;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function modelChunk(text: string) {
    return {
      message: {
        modelChunk: { role: 'model', content: [{ text }] },
      },
    };
  }

  function customChunk(customPatch: any) {
    return { message: { customPatch } };
  }

  function turnEndResult(out: any) {
    return { result: out };
  }

  it('streams text and resolves a response', async () => {
    mock.onNext(() =>
      sseResponse([
        modelChunk('Hello '),
        modelChunk('world'),
        turnEndResult({
          snapshotId: 'snap-1',
          message: {
            role: 'model',
            content: [{ text: 'Hello world' }],
          },
          finishReason: 'stop',
        }),
      ])
    );

    const agent = remoteAgent({ url: '/api/weatherAgent' });
    const chat = agent.chat();
    const turn = chat.sendStream('Weather in Tokyo?');

    const seen: string[] = [];
    for await (const chunk of turn.stream) {
      if (chunk.text) seen.push(chunk.accumulatedText);
    }
    assert.deepEqual(seen, ['Hello ', 'Hello world']);

    const res = await turn.response;
    assert.equal(res.text, 'Hello world');
    assert.equal(res.finishReason, 'stop');
    assert.equal(res.snapshotId, 'snap-1');
    assert.equal(chat.snapshotId, 'snap-1');

    // Request body shape.
    assert.equal(mock.requests.length, 1);
    assert.equal(mock.requests[0].url, '/api/weatherAgent');
    assert.deepEqual(mock.requests[0].body.data, {
      message: { role: 'user', content: [{ text: 'Weather in Tokyo?' }] },
    });
  });

  it('surfaces streamed custom-state updates as chunk.custom (post-patch)', async () => {
    mock.onNext(() =>
      sseResponse([
        // First patch of a turn is a whole-document replace re-basing onto the
        // server baseline.
        customChunk([
          { op: 'replace', path: '', value: { status: 'Decomposing…' } },
        ]),
        modelChunk('Answer '),
        // Subsequent patches are incremental, bare-rooted diffs.
        customChunk([{ op: 'replace', path: '/status', value: 'Done' }]),
        turnEndResult({
          snapshotId: 'snap-1',
          state: { custom: { status: 'Done' }, messages: [] },
          message: { role: 'model', content: [{ text: 'Answer ' }] },
          finishReason: 'stop',
        }),
      ])
    );

    const agent = remoteAgent<{ status: string }>({
      url: '/api/researchAgent',
    });
    const chat = agent.chat();
    const turn = chat.sendStream('Research X');

    const customs: Array<{ status: string } | undefined> = [];
    const texts: string[] = [];
    for await (const chunk of turn.stream) {
      if (chunk.custom) {
        customs.push(chunk.custom);
        // chunk.custom mirrors the live chat.state at yield time.
        assert.deepEqual(chunk.custom, chat.state);
      } else if (chunk.text) {
        texts.push(chunk.text);
      }
    }

    assert.deepEqual(customs, [{ status: 'Decomposing…' }, { status: 'Done' }]);
    assert.deepEqual(texts, ['Answer ']);
    // Distinct, fresh object references for `===` change detection.
    assert.notStrictEqual(customs[0], customs[1]);

    const res = await turn.response;
    assert.deepEqual(res.state, { status: 'Done' });
    assert.deepEqual(chat.state, { status: 'Done' });
  });

  it('carries snapshotId across multi-turn (server-managed)', async () => {
    mock.onNext(() =>
      sseResponse([
        turnEndResult({ snapshotId: 'snap-1', finishReason: 'stop' }),
      ])
    );
    mock.onNext((req) => {
      // Second turn must send init.snapshotId.
      assert.deepEqual(req.body.init, { snapshotId: 'snap-1' });
      return sseResponse([
        turnEndResult({ snapshotId: 'snap-2', finishReason: 'stop' }),
      ]);
    });

    const agent = remoteAgent({ url: '/api/a' });
    const chat = agent.chat();
    await chat.send('one');
    assert.equal(chat.snapshotId, 'snap-1');
    await chat.send('two');
    assert.equal(chat.snapshotId, 'snap-2');
  });

  it('keeps chat.state live across a non-streaming send() for server-managed agents and res.state mirrors it', async () => {
    // Server-managed agents return only a snapshotId on the wire (no `state`);
    // custom state arrives as streamed customPatch chunks. send() must drain the
    // stream so chat.state stays live and res.state falls back to it.
    mock.onNext(() =>
      sseResponse([
        customChunk([{ op: 'replace', path: '', value: { count: 1 } }]),
        turnEndResult({ snapshotId: 'snap-1', finishReason: 'stop' }),
      ])
    );
    mock.onNext(() =>
      sseResponse([
        customChunk([{ op: 'replace', path: '/count', value: 2 }]),
        turnEndResult({ snapshotId: 'snap-2', finishReason: 'stop' }),
      ])
    );

    const agent = remoteAgent<{ count: number }>({ url: '/api/counter' });
    const chat = agent.chat();

    const res1 = await chat.send('inc');
    assert.deepEqual(chat.state, { count: 1 });
    // res.state mirrors chat.state even though the wire output had no `state`.
    assert.deepEqual(res1.state, { count: 1 });
    assert.equal(res1.snapshotId, 'snap-1');

    // A second non-streaming send() must refresh chat.state (not go stale).
    const res2 = await chat.send('inc');
    assert.deepEqual(chat.state, { count: 2 });
    assert.deepEqual(res2.state, { count: 2 });
  });

  it('carries state across multi-turn (client-managed)', async () => {
    mock.onNext(() =>
      sseResponse([
        turnEndResult({
          state: { custom: { unit: 'celsius' }, messages: [] },
          finishReason: 'stop',
        }),
      ])
    );
    mock.onNext((req) => {
      assert.deepEqual(req.body.init, {
        state: { custom: { unit: 'celsius' }, messages: [] },
      });
      return sseResponse([
        turnEndResult({
          state: { custom: { unit: 'fahrenheit' }, messages: [] },
          finishReason: 'stop',
        }),
      ]);
    });

    const agent = remoteAgent<{ unit: string }>({ url: '/api/a' });
    const chat = agent.chat();
    const res1 = await chat.send('one');
    assert.deepEqual(res1.state, { unit: 'celsius' });
    assert.deepEqual(chat.state, { unit: 'celsius' });
    const res2 = await chat.send('two');
    assert.deepEqual(res2.state, { unit: 'fahrenheit' });
    assert.deepEqual(chat.state, { unit: 'fahrenheit' });
  });

  it('exposes interrupts and builds resume parts', async () => {
    mock.onNext(() =>
      sseResponse([
        turnEndResult({
          snapshotId: 'snap-1',
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'userApproval',
                  ref: 'r1',
                  input: { action: 'transfer' },
                },
                metadata: { interrupt: true },
              },
            ],
          },
          finishReason: 'interrupted',
        }),
      ])
    );

    const agent = remoteAgent({ url: '/api/bank' });
    const chat = agent.chat();
    const res = await chat.send('Transfer $500');

    assert.equal(res.finishReason, 'interrupted');
    assert.equal(res.interrupts.length, 1);
    const approval = res.interrupts[0];
    assert.equal(approval.name, 'userApproval');
    assert.deepEqual(approval.input, { action: 'transfer' });

    const respondPart = approval.respond({ approved: true });
    assert.deepEqual(respondPart, {
      toolResponse: {
        name: 'userApproval',
        ref: 'r1',
        output: { approved: true },
      },
    });

    const restartPart = approval.restart();
    assert.deepEqual(restartPart, {
      toolRequest: {
        name: 'userApproval',
        ref: 'r1',
        input: { action: 'transfer' },
      },
    });

    // chat.resume is sugar for send({ resume }).
    mock.onNext((req) => {
      assert.deepEqual(req.body.data.resume, { respond: [respondPart] });
      return sseResponse([
        turnEndResult({ snapshotId: 'snap-2', finishReason: 'stop' }),
      ]);
    });
    await chat.resume({ respond: [respondPart] });
    assert.equal(chat.snapshotId, 'snap-2');
  });

  it('throws AgentError on a failed turn and keeps last-good state', async () => {
    // First successful turn establishes last-good snapshot.
    mock.onNext(() =>
      sseResponse([
        turnEndResult({ snapshotId: 'snap-1', finishReason: 'stop' }),
      ])
    );
    // Second turn fails.
    mock.onNext(() =>
      sseResponse([
        turnEndResult({
          snapshotId: 'snap-1',
          finishReason: 'failed',
          error: { status: 'INTERNAL', message: 'boom' },
        }),
      ])
    );

    const agent = remoteAgent({ url: '/api/a' });
    const chat = agent.chat();
    await chat.send('one');
    assert.equal(chat.snapshotId, 'snap-1');

    await assert.rejects(
      () => chat.send('two'),

      (err: AgentError) => {
        assert.ok(err instanceof AgentError);
        assert.equal(err.status, 'INTERNAL');
        assert.equal(err.message, 'boom');
        assert.equal(err.snapshotId, 'snap-1');
        return true;
      }
    );
    // Chat recovered to last-good snapshot.
    assert.equal(chat.snapshotId, 'snap-1');
  });

  it('throws AgentError when the stream is iterated on a failed turn', async () => {
    mock.onNext(() =>
      sseResponse([
        turnEndResult({
          finishReason: 'failed',
          error: { status: 'UNAVAILABLE', message: 'down' },
        }),
      ])
    );
    const agent = remoteAgent({ url: '/api/a' });
    const chat = agent.chat();
    const turn = chat.sendStream('hi');
    await assert.rejects(async () => {
      for await (const _ of turn.stream) {
        // drain
      }
    }, AgentError);
  });

  it('loadChat() loads a snapshot and restores history', async () => {
    mock.onNext((req) => {
      assert.equal(req.url, '/api/a/getSnapshot');
      assert.deepEqual(req.body.data, { snapshotId: 'snap-9' });
      return jsonResponse({
        result: {
          snapshotId: 'snap-9',
          createdAt: '2026-01-01',
          state: {
            messages: [{ role: 'user', content: [{ text: 'earlier' }] }],
          },
          status: 'completed',
        },
      });
    });

    const agent = remoteAgent({ url: '/api/a' });
    const chat = await agent.loadChat({ snapshotId: 'snap-9' });
    assert.equal(chat.snapshotId, 'snap-9');
    assert.equal(chat.messages.length, 1);
    assert.equal(chat.messages[0].content[0].text, 'earlier');
  });

  it('getSnapshot reads via the getSnapshot endpoint', async () => {
    mock.onNext((req) => {
      assert.equal(req.url, '/api/a/getSnapshot');
      return jsonResponse({
        result: {
          snapshotId: 'snap-1',
          createdAt: '2026',
          state: {},
          status: 'completed',
        },
      });
    });
    const agent = remoteAgent({ url: '/api/a' });
    const snap = await agent.getSnapshot('snap-1');
    assert.equal(snap?.snapshotId, 'snap-1');
  });

  it('abort posts to the abort endpoint', async () => {
    mock.onNext((req) => {
      assert.equal(req.url, '/api/a/abort');
      assert.deepEqual(req.body.data, { snapshotId: 'snap-1' });
      return jsonResponse({
        result: { snapshotId: 'snap-1', status: 'pending' },
      });
    });
    const agent = remoteAgent({ url: '/api/a' });
    const status = await agent.abort('snap-1');
    assert.equal(status, 'pending');
  });

  it('detach submits a background task and reports detached', async () => {
    mock.onNext((req) => {
      assert.equal(req.body.data.detach, true);
      return sseResponse([
        turnEndResult({ snapshotId: 'bg-1', finishReason: 'detached' }),
      ]);
    });
    const agent = remoteAgent({ url: '/api/a' });
    const chat = agent.chat();
    const task = await chat.detach('long job');
    assert.equal(task.snapshotId, 'bg-1');

    // wait() polls /getSnapshot until terminal.
    mock.onNext(() =>
      jsonResponse({
        result: {
          snapshotId: 'bg-1',
          createdAt: '2026',
          state: {},
          status: 'completed',
        },
      })
    );
    const snap = await task.wait({ intervalMs: 1 });
    assert.equal(snap.status, 'completed');
  });

  it('applies static and async headers', async () => {
    mock.onNext((req) => {
      assert.equal(req.headers['Authorization'], 'Bearer xyz');
      return sseResponse([turnEndResult({ finishReason: 'stop' })]);
    });
    const agent = remoteAgent({
      url: '/api/a',
      headers: async () => ({ Authorization: 'Bearer xyz' }),
    });
    await agent.chat().send('hi');
  });
});
