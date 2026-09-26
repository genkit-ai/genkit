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
import { WebSocket } from 'ws';
import { REFLECTION_AUTH_ERROR_CODE } from '../src/reflection-auth.js';
import { HostEvent, ReflectionHost } from '../src/reflection-host.js';

/**
 * A minimal box runtime: dials the host, registers, and answers reflection
 * requests. Mirrors what a real Genkit runtime does over reflection v2, so we
 * can exercise the host without spawning a subprocess.
 */
class FakeRuntime {
  private ws: WebSocket;
  private abort?: AbortController;
  private currentRunId?: string;
  readonly ready: Promise<void>;

  constructor(
    host: ReflectionHost,
    private readonly id: string,
    private readonly handlers: {
      listActions?: () => unknown;
      runAction?: (
        params: any,
        emit: (chunk: unknown) => void,
        sendState: (traceId: string) => void,
        signal: AbortSignal
      ) => Promise<unknown> | unknown;
      cancelAction?: (params: any) => unknown;
    } = {},
    // An object, not a bare param, so `{ secret: undefined }` means "send none"
    // rather than falling back to the default.
    auth: { secret: string | undefined } = { secret: host.secret }
  ) {
    const secret = auth.secret;
    this.ws = new WebSocket(host.url);
    this.ready = new Promise((resolve, reject) => {
      this.ws.on('open', () => {
        this.send({
          jsonrpc: '2.0',
          method: 'register',
          params: { id: this.id, pid: process.pid, name: this.id, secret },
          id: 'reg',
        });
      });
      this.ws.on('message', (data) => {
        const msg = JSON.parse(data.toString());
        if (msg.id === 'reg') {
          if ('result' in msg) resolve();
          else reject(Object.assign(new Error(msg.error.message), msg.error));
          return;
        }
        void this.handle(msg);
      });
      this.ws.on('error', reject);
    });
  }

  private send(obj: unknown) {
    this.ws.send(JSON.stringify(obj));
  }

  private async handle(msg: any) {
    if (msg.method === 'listActions') {
      this.send({
        jsonrpc: '2.0',
        id: msg.id,
        result: { actions: this.handlers.listActions?.() ?? {} },
      });
    } else if (msg.method === 'runAction') {
      this.abort = new AbortController();
      this.currentRunId = msg.id;
      const emit = (chunk: unknown) =>
        this.send({
          jsonrpc: '2.0',
          method: 'streamChunk',
          params: { requestId: msg.id, chunk },
        });
      const sendState = (traceId: string) =>
        this.send({
          jsonrpc: '2.0',
          method: 'runActionState',
          params: { requestId: msg.id, state: { traceId } },
        });
      try {
        const result = await this.handlers.runAction?.(
          msg.params,
          emit,
          sendState,
          this.abort.signal
        );
        this.send({
          jsonrpc: '2.0',
          id: msg.id,
          result: { result, telemetry: { traceId: 'trace-1' } },
        });
      } catch (e: any) {
        this.send({
          jsonrpc: '2.0',
          id: msg.id,
          error: { code: -32000, message: e.message, data: { code: 13 } },
        });
      }
    } else if (msg.method === 'cancelAction') {
      this.handlers.cancelAction?.(msg.params);
      // Abort the in-flight body and reply to the runAction with CANCELLED, as
      // a real runtime does.
      this.abort?.abort();
      if (this.currentRunId) {
        this.send({
          jsonrpc: '2.0',
          id: this.currentRunId,
          error: { code: -32000, message: 'Action was cancelled', data: {} },
        });
        this.currentRunId = undefined;
      }
      this.send({
        jsonrpc: '2.0',
        id: msg.id,
        result: { message: 'Action cancelled' },
      });
    }
  }

  close() {
    this.ws.close();
  }
}

describe('ReflectionHost', () => {
  let host: ReflectionHost;

  beforeEach(async () => {
    host = new ReflectionHost();
    await host.start();
  });

  afterEach(async () => {
    await host.stop();
  });

  it('accepts a runtime registration and lists it', async () => {
    const rt = new FakeRuntime(host, 'rt1');
    const info = await host.waitForRuntime('rt1');
    await rt.ready;
    assert.strictEqual(info.id, 'rt1');
    assert.deepStrictEqual(host.listRuntimeIds(), ['rt1']);
    rt.close();
  });

  it('runs an action and returns its result', async () => {
    const rt = new FakeRuntime(host, 'rt1', {
      runAction: (params) => ({ echoed: params.input }),
    });
    await host.waitForRuntime('rt1');
    await rt.ready;
    const res = await host.runAction('rt1', {
      key: '/tool/echo',
      input: { hello: 'world' },
    });
    assert.deepStrictEqual(res.result, { echoed: { hello: 'world' } });
    assert.strictEqual(res.telemetry?.traceId, 'trace-1');
    rt.close();
  });

  it('lists actions from the runtime', async () => {
    const rt = new FakeRuntime(host, 'rt1', {
      listActions: () => ({
        '/tool/echo': { key: '/tool/echo', name: 'echo' },
      }),
    });
    await host.waitForRuntime('rt1');
    await rt.ready;
    const actions = await host.listActions('rt1');
    assert.deepStrictEqual(actions['/tool/echo'], {
      key: '/tool/echo',
      name: 'echo',
    });
    rt.close();
  });

  it('delivers streamed chunks and the early trace id', async () => {
    const rt = new FakeRuntime(host, 'rt1', {
      runAction: async (_params, emit, sendState) => {
        sendState('early-trace');
        emit({ n: 1 });
        emit({ n: 2 });
        return { done: true };
      },
    });
    await host.waitForRuntime('rt1');
    await rt.ready;
    const chunks: unknown[] = [];
    let traceId: string | undefined;
    const res = await host.runAction(
      'rt1',
      { key: '/flow/count' },
      {
        onChunk: (c) => chunks.push(c),
        onTraceId: (t) => (traceId = t),
      }
    );
    assert.deepStrictEqual(chunks, [{ n: 1 }, { n: 2 }]);
    assert.strictEqual(traceId, 'early-trace');
    assert.deepStrictEqual(res.result, { done: true });
    rt.close();
  });

  it('propagates runtime errors', async () => {
    const rt = new FakeRuntime(host, 'rt1', {
      runAction: () => {
        throw new Error('boom');
      },
    });
    await host.waitForRuntime('rt1');
    await rt.ready;
    await assert.rejects(
      () => host.runAction('rt1', { key: '/tool/fail' }),
      /boom/
    );
    rt.close();
  });

  it('cancels an in-flight action when the abort signal fires', async () => {
    let cancelled = false;
    const rt = new FakeRuntime(host, 'rt1', {
      runAction: async (_params, _emit, sendState, signal) => {
        sendState('trace-x');
        // Never resolve on its own; wait to be cancelled via the signal.
        await new Promise<void>((resolve) => {
          if (signal.aborted) return resolve();
          signal.addEventListener('abort', () => resolve());
        });
        throw new Error('Action was cancelled');
      },
      cancelAction: () => {
        cancelled = true;
      },
    });
    await host.waitForRuntime('rt1');
    await rt.ready;
    const ac = new AbortController();
    const p = host.runAction(
      'rt1',
      { key: '/tool/slow' },
      { abortSignal: ac.signal }
    );
    // Attach the rejection expectation up front so the cancellation (which can
    // arrive before we finish waiting) is never an unhandled rejection.
    const rejected = assert.rejects(p, /cancelled/i);
    // Give the runtime a moment to send its trace id, then abort.
    await new Promise((r) => setTimeout(r, 50));
    ac.abort();
    await rejected;
    assert.strictEqual(cancelled, true);
    rt.close();
  });

  it('emits disconnect when a runtime drops', async () => {
    const rt = new FakeRuntime(host, 'rt1');
    await host.waitForRuntime('rt1');
    await rt.ready;
    const gone = new Promise<void>((resolve) => {
      host.on(HostEvent.RUNTIME_DISCONNECT, (info) => {
        if (info.id === 'rt1') resolve();
      });
    });
    rt.close();
    await gone;
    assert.deepStrictEqual(host.listRuntimeIds(), []);
  });
});

describe('ReflectionHost auth', () => {
  let host: ReflectionHost;

  afterEach(async () => {
    await host.stop();
  });

  it('generates a secret per host by default', () => {
    host = new ReflectionHost();
    assert.ok(host.secret && host.secret.length >= 32);
    assert.notStrictEqual(new ReflectionHost().secret, host.secret);
  });

  for (const [label, secret] of [
    ['missing', undefined],
    ['wrong', 'not-the-secret'],
  ] as const) {
    it(`rejects a ${label} secret with the terminal auth code`, async () => {
      host = new ReflectionHost({ secret: 's3cret' });
      await host.start();
      const rt = new FakeRuntime(host, 'intruder', {}, { secret });
      await assert.rejects(rt.ready, { code: REFLECTION_AUTH_ERROR_CODE });
      assert.deepStrictEqual(host.listRuntimeIds(), []);
    });
  }

  it('accepts any runtime when auth is off', async () => {
    host = new ReflectionHost({ secret: false });
    await host.start();
    assert.strictEqual(host.secret, undefined);
    const rt = new FakeRuntime(host, 'legacy', {}, { secret: undefined });
    await rt.ready;
    assert.deepStrictEqual(host.listRuntimeIds(), ['legacy']);
    rt.close();
  });
});
