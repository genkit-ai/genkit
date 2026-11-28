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
import { GenkitError } from 'genkit';
import { PassThrough, Readable } from 'node:stream';
import { after, before, describe, it } from 'node:test';
import * as vm from 'node:vm';
import {
  Client,
  backoff,
  errorMessage,
  httpError,
  parseRetryAfter,
  type DecideRequest,
  type Endpoint,
} from '../src/client.js';
import { FakeServer } from './fake_server.js';

const request: DecideRequest = {
  state: 'My card was charged twice.',
  media: { images: [], audio: [], videos: [] },
  questions: {
    urgent: { type: 'noul', instructions: 'Is it urgent?' },
  },
};

function client(
  server: FakeServer,
  endpoint: Partial<Endpoint> = {},
  extra: Partial<ConstructorParameters<typeof Client>[0]> = {}
) {
  return new Client({
    fetch: globalThis.fetch,
    baseUrl: server.url,
    apiKey: 'key',
    endpoint: {
      name: 'test',
      path: '/v1/systemone',
      modelsPath: '',
      ...endpoint,
    },
    ...extra,
  });
}

describe('client', () => {
  let server: FakeServer;
  before(async () => {
    server = await new FakeServer().start();
  });
  after(() => server.stop());

  it('posts the native body and decodes the answers', async () => {
    const resp = await client(
      server,
      {},
      { headers: { 'X-Trace': 't1' } }
    ).decide('jev-1.13.0', {
      ...request,
      extra: { session_id: 's1', model: 'other' },
    });
    assert.deepStrictEqual(resp.answers, {
      urgent: { type: 'noul', noul: 0.93 },
    });
    const sent = server.bodies.at(-1);
    assert.deepStrictEqual(sent, {
      session_id: 's1',
      model: 'jev-1.13.0',
      state: 'My card was charged twice.',
      questions: request.questions,
    });
    const { url, headers } = server.requests.at(-1)!;
    assert.strictEqual(url, '/v1/systemone');
    assert.strictEqual(headers.authorization, 'Bearer key');
    assert.strictEqual(headers['x-trace'], 't1');
  });

  it('sends no authorization without a key', async () => {
    await client(server, {}, { apiKey: undefined }).decide('clef', request);
    assert.strictEqual(
      server.requests.at(-1)!.headers.authorization,
      undefined
    );
  });

  it('routes and unwraps through the hooks', async () => {
    server.wrap = (reply) => ({ result: reply });
    try {
      const resp = await client(server, {
        path: '/run',
        route: (model, body) => {
          delete body.model;
          return { path: model, body: { input: body } };
        },
        unwrap: (body: any) => body.result,
      }).decide('d1', request);
      assert.strictEqual(server.requests.at(-1)!.url, '/run/d1');
      assert.deepStrictEqual(Object.keys(server.bodies.at(-1)), ['input']);
      assert.deepStrictEqual(resp.answers.urgent, { type: 'noul', noul: 0.93 });
    } finally {
      server.wrap = undefined;
    }
  });

  it('names the endpoint in an unwrap error and keeps its status', async () => {
    await assert.rejects(
      client(server, {
        unwrap: () => {
          throw new Error('bad envelope');
        },
      }).decide('d1', request),
      (e: GenkitError) =>
        e.status === 'UNKNOWN' && e.originalMessage === 'test: bad envelope'
    );
    await assert.rejects(
      client(server, {
        unwrap: () => {
          throw new GenkitError({ status: 'PERMISSION_DENIED', message: 'no' });
        },
      }).decide('d1', request),
      (e: GenkitError) => e.status === 'PERMISSION_DENIED'
    );
  });

  it('retries an overload, honoring Retry-After', async () => {
    const waits: number[] = [];
    server.failures = [
      {
        status: 503,
        body: '{"error":{"message":"busy"}}',
        headers: { 'Retry-After': '0' },
      },
      { status: 429, body: '{"error":"slow down"}' },
    ];
    const before = server.requests.length;
    await client(
      server,
      {},
      {
        sleep: async (ms) => {
          waits.push(ms);
        },
      }
    ).decide('jev', request);
    assert.strictEqual(server.requests.length - before, 3);
    assert.strictEqual(waits[0], 0);
    assert.ok(waits[1] >= 750 && waits[1] <= 1000, `backoff ${waits[1]}`);
  });

  it('gives up after two retries and does not retry a client error', async () => {
    const sleep = async () => {};
    server.failures = [{ status: 500 }, { status: 502 }, { status: 503 }];
    await assert.rejects(
      client(server, {}, { sleep }).decide('jev', request),
      (e: GenkitError) => e.status === 'UNAVAILABLE'
    );
    assert.strictEqual(server.failures.length, 0);
    server.failures = [
      {
        status: 400,
        body: '{"detail":[{"loc":["body","state"],"msg":"field required"}]}',
      },
    ];
    const before = server.requests.length;
    await assert.rejects(
      client(server, {}, { sleep }).decide('jev', request),
      (e: GenkitError) =>
        e.status === 'INVALID_ARGUMENT' &&
        e.originalMessage === 'test: HTTP 400: body.state: field required'
    );
    assert.strictEqual(server.requests.length - before, 1);
  });

  it('reports a cancellation during backoff as one', async () => {
    const abort = new AbortController();
    server.failures = [{ status: 503 }];
    await assert.rejects(
      client(
        server,
        {},
        {
          sleep: async () => {
            abort.abort();
            throw new Error('aborted');
          },
        }
      ).decide('jev', request, { signal: abort.signal }),
      (e: GenkitError) =>
        e.status === 'CANCELLED' &&
        /last attempt: test: HTTP 503/.test(e.message)
    );
    server.failures = [];
  });

  it('fails once on a body past 8 MiB', async () => {
    let calls = 0;
    const big = new Client({
      fetch: async () => {
        calls++;
        return new Response('x'.repeat((8 << 20) + 1));
      },
      baseUrl: 'http://unused',
      endpoint: { name: 'test', path: '', modelsPath: '' },
      sleep: async () => {},
    });
    await assert.rejects(
      big.decide('jev', request),
      (e: GenkitError) =>
        e.status === 'INTERNAL' && /larger than 8388608 bytes/.test(e.message)
    );
    assert.strictEqual(calls, 1);
  });

  describe('a body from a fetch of your own', () => {
    // node-fetch v2 hands back a Node.js Readable as the body.
    const nodeBody = (body: unknown): Response =>
      ({
        status: 200,
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        body,
      }) as unknown as Response;
    const ownClient = (fetch: (init?: RequestInit) => Promise<Response>) =>
      new Client({
        fetch: async (_url, init) => fetch(init),
        baseUrl: 'http://unused',
        endpoint: { name: 'test', path: '', modelsPath: '' },
        sleep: async () => {},
      });

    it('reads a Node.js stream, with UTF-8 split across chunks', async () => {
      const answer = Buffer.from(
        JSON.stringify({
          model: 'jev',
          answers: { urgent: { noul: 0.93, note: 'café' } },
        })
      );
      // Cuts the two bytes of é apart.
      const cut = answer.indexOf(0xc3) + 1;
      const res = await ownClient(async () =>
        nodeBody(Readable.from([answer.subarray(0, cut), answer.subarray(cut)]))
      ).decide('jev', request);
      assert.deepStrictEqual(res.answers, {
        urgent: { noul: 0.93, note: 'café' },
      });
    });

    it('takes string chunks as text', async () => {
      const answer = JSON.stringify({ model: 'jev', answers: {} });
      const res = await ownClient(async () =>
        nodeBody(Readable.from([answer.slice(0, 5), answer.slice(5)]))
      ).decide('jev', request);
      assert.strictEqual(res.model, 'jev');
    });

    it(
      'stops reading an endless stream at 8 MiB, once, and ends the attempt',
      { timeout: 10_000 },
      async () => {
        let calls = 0;
        let pulled = 0;
        let signal: AbortSignal | undefined;
        await assert.rejects(
          ownClient(async (init) => {
            calls++;
            signal = init?.signal ?? undefined;
            return nodeBody(
              Readable.from(
                (function* () {
                  // Long enough to pass the cap, short enough that a read
                  // without one ends instead of filling memory.
                  for (let i = 0; i < 64; i++) {
                    pulled++;
                    yield Buffer.alloc(1 << 20, 0x78);
                  }
                })(),
                { highWaterMark: 1 }
              )
            );
          }).decide('jev', request),
          (e: GenkitError) =>
            e.status === 'INTERNAL' &&
            /larger than 8388608 bytes/.test(e.message)
        );
        assert.strictEqual(calls, 1);
        // 9 to pass the cap, plus what the stream reads ahead.
        assert.ok(pulled <= 12, `pulled ${pulled} chunks`);
        // node-fetch v2 closes the connection only when its signal aborts.
        assert.strictEqual(signal?.aborted, true);
      }
    );

    it('ends the attempt when a WHATWG body passes 8 MiB', async () => {
      let signal: AbortSignal | undefined;
      await assert.rejects(
        ownClient(async (init) => {
          signal = init?.signal ?? undefined;
          return new Response('x'.repeat((8 << 20) + 1));
        }).decide('jev', request),
        (e: GenkitError) => e.status === 'INTERNAL'
      );
      assert.strictEqual(signal?.aborted, true);
    });

    it('refuses chunks that are not bytes, once', async () => {
      let calls = 0;
      let signal: AbortSignal | undefined;
      await assert.rejects(
        ownClient(async (init) => {
          calls++;
          signal = init?.signal ?? undefined;
          return nodeBody(Readable.from([1, 2, 3]));
        }).decide('jev', request),
        (e: GenkitError) =>
          e.status === 'INTERNAL' && /other than bytes/.test(e.message)
      );
      assert.strictEqual(calls, 1);
      assert.strictEqual(signal?.aborted, true);
    });

    it('refuses a body that cannot be read, once', async () => {
      let calls = 0;
      let signal: AbortSignal | undefined;
      await assert.rejects(
        ownClient(async (init) => {
          calls++;
          signal = init?.signal ?? undefined;
          return nodeBody({});
        }).decide('jev', request),
        (e: GenkitError) =>
          e.status === 'INTERNAL' &&
          /neither a ReadableStream nor async iterable/.test(e.message)
      );
      assert.strictEqual(calls, 1);
      assert.strictEqual(signal?.aborted, true);
    });

    it('takes bytes made in another realm', async () => {
      const answer = JSON.stringify({ model: 'jev', answers: {} });
      const chunk = vm.runInNewContext('new Uint8Array(bytes)', {
        bytes: [...Buffer.from(answer)],
      });
      assert.ok(!(chunk instanceof Uint8Array));
      const res = await ownClient(async () =>
        nodeBody(Readable.from([chunk]))
      ).decide('jev', request);
      assert.strictEqual(res.model, 'jev');
    });

    it(
      'aborts a Node.js body past 8 MiB while still listening to it',
      { timeout: 10_000 },
      async () => {
        let uncaught: unknown;
        const onUncaught = (e: unknown) => (uncaught = e);
        process.on('uncaughtException', onUncaught);
        let body: PassThrough | undefined;
        try {
          await assert.rejects(
            ownClient(async (init) => {
              body = new PassThrough({ highWaterMark: 1 << 20 });
              // As node-fetch v2 does on abort: an error emitted on the
              // body, unguarded.
              init?.signal?.addEventListener('abort', () =>
                body!.emit('error', new Error('The user aborted a request.'))
              );
              for (let i = 0; i < 9; i++) body.write(Buffer.alloc(1 << 20));
              body.end();
              return nodeBody(body);
            }).decide('jev', request),
            (e: GenkitError) =>
              e.status === 'INTERNAL' &&
              /larger than 8388608 bytes/.test(e.message)
          );
          await new Promise((resolve) => setImmediate(resolve));
        } finally {
          process.off('uncaughtException', onUncaught);
        }
        assert.strictEqual(uncaught, undefined);
        assert.strictEqual(body?.destroyed, true);
      }
    );

    for (const [name, cancel] of [
      ['rejects', () => Promise.reject(new Error('cannot cancel'))],
      ['never settles', () => new Promise<void>(() => {})],
    ] as const) {
      it(
        `refuses a WHATWG body past 8 MiB whose cancel ${name}, once`,
        { timeout: 10_000 },
        async () => {
          let calls = 0;
          let signal: AbortSignal | undefined;
          await assert.rejects(
            ownClient(async (init) => {
              calls++;
              signal = init?.signal ?? undefined;
              let sent = 0;
              return nodeBody(
                new ReadableStream<Uint8Array>({
                  // Ends after 64 MiB, so a read without a cap fails
                  // instead of filling memory.
                  pull: (controller) => {
                    if (sent++ < 64) {
                      controller.enqueue(new Uint8Array(1 << 20));
                    } else {
                      controller.close();
                    }
                  },
                  cancel,
                })
              );
            }).decide('jev', request),
            (e: GenkitError) =>
              e.status === 'INTERNAL' &&
              /larger than 8388608 bytes/.test(e.message)
          );
          assert.strictEqual(calls, 1);
          assert.strictEqual(signal?.aborted, true);
        }
      );
    }

    it('retries a stream that fails midway', async () => {
      let calls = 0;
      await assert.rejects(
        ownClient(async () => {
          calls++;
          return nodeBody(
            Readable.from(
              (function* () {
                yield Buffer.from('{"model"');
                throw new Error('premature close');
              })()
            )
          );
        }).decide('jev', request),
        (e: GenkitError) =>
          e.status === 'UNAVAILABLE' && /premature close/.test(e.message)
      );
      assert.strictEqual(calls, 3);
    });

    it(
      'reports a caller abort during the read as CANCELLED',
      { timeout: 5_000 },
      async () => {
        const abort = new AbortController();
        let calls = 0;
        let body: PassThrough | undefined;
        await assert.rejects(
          ownClient(async (init) => {
            calls++;
            body = new PassThrough();
            // As node-fetch v2 does on abort.
            init?.signal?.addEventListener('abort', () =>
              body!.destroy(new Error('The user aborted a request.'))
            );
            body.write('{"model"');
            setImmediate(() => abort.abort());
            return nodeBody(body);
          }).decide('jev', request, { signal: abort.signal }),
          (e: GenkitError) =>
            e.status === 'CANCELLED' && /user aborted a request/.test(e.message)
        );
        assert.strictEqual(calls, 1);
        // The read ended because the transport destroyed the body.
        assert.strictEqual(body?.destroyed, true);
      }
    );
  });

  it('does not retry 501 or 505', async () => {
    for (const status of [501, 505]) {
      server.failures = [{ status }];
      const before = server.requests.length;
      await assert.rejects(
        client(server, {}, { sleep: async () => {} }).decide('jev', request),
        (e: GenkitError) => e.status === 'UNIMPLEMENTED'
      );
      assert.strictEqual(server.requests.length - before, 1, String(status));
    }
  });

  it('ends a Retry-After wait at once when the caller cancels', async () => {
    server.failures = [{ status: 503, headers: { 'Retry-After': '5' } }];
    const abort = new AbortController();
    const started = Date.now();
    setTimeout(() => abort.abort(), 100);
    await assert.rejects(
      client(server).decide('jev', request, { signal: abort.signal }),
      (e: GenkitError) => e.status === 'CANCELLED'
    );
    assert.ok(Date.now() - started < 1000, 'cancelled within a second');
    server.failures = [];
  });

  it('caps Retry-After at 30 seconds', async () => {
    const waits: number[] = [];
    server.failures = [{ status: 429, headers: { 'Retry-After': '100' } }];
    await client(
      server,
      {},
      {
        sleep: async (ms) => {
          waits.push(ms);
        },
      }
    ).decide('jev', request);
    assert.deepStrictEqual(waits, [30000]);
  });

  it('retries an attempt that times out', async () => {
    server.failures = [
      { status: 200, body: '{}', delayMs: 400 },
      { status: 200, body: '{}', delayMs: 400 },
      { status: 200, body: '{}', delayMs: 400 },
    ];
    const before = server.requests.length;
    await assert.rejects(
      client(server, {}, { timeoutMs: 50, sleep: async () => {} }).decide(
        'jev',
        request
      ),
      (e: GenkitError) => e.status === 'UNAVAILABLE'
    );
    assert.strictEqual(server.requests.length - before, 3);
    await new Promise((resolve) => setTimeout(resolve, 500));
  });

  it('retries a network error', async () => {
    let calls = 0;
    const down = new Client({
      fetch: async () => {
        calls++;
        throw new TypeError('fetch failed');
      },
      baseUrl: 'http://unused',
      endpoint: { name: 'test', path: '', modelsPath: '' },
      sleep: async () => {},
    });
    await assert.rejects(
      down.decide('jev', request),
      (e: GenkitError) =>
        e.status === 'UNAVAILABLE' && e.originalMessage === 'test: fetch failed'
    );
    assert.strictEqual(calls, 3);
  });

  it('refuses a response that is not a System One answer', async () => {
    const cases: [string, RegExp][] = [
      ['not json', /response is not a System One answer/],
      ['[1, 2]', /response is not a System One answer/],
      ['{"model": "jev"}', /response carries no answers/],
      ['{"model": "jev", "answers": [1]}', /response carries no answers/],
    ];
    for (const [body, message] of cases) {
      server.failures = [{ status: 200, body }];
      await assert.rejects(
        client(server).decide('jev', request),
        (e: GenkitError) => e.status === 'INTERNAL' && message.test(e.message),
        body
      );
    }
  });

  it('does not follow a redirect, and does not retry it', async () => {
    server.failures = [{ status: 302, headers: { Location: '/elsewhere' } }];
    const before = server.requests.length;
    await assert.rejects(
      client(server).decide('jev', request),
      (e: GenkitError) =>
        e.status === 'FAILED_PRECONDITION' && /HTTP 302/.test(e.message)
    );
    assert.strictEqual(server.requests.length - before, 1);
  });

  it('reports hook mistakes with a status', async () => {
    await assert.rejects(
      client(server, {
        route: () => {
          throw new Error('no route');
        },
      }).decide('jev', request),
      (e: GenkitError) =>
        e.status === 'UNKNOWN' && e.originalMessage === 'test: no route'
    );
    await assert.rejects(
      client(server, {
        route: () => {
          throw new GenkitError({
            status: 'INVALID_ARGUMENT',
            message: 'bad id',
          });
        },
      }).decide('jev', request),
      (e: GenkitError) =>
        e.status === 'INVALID_ARGUMENT' && e.originalMessage === 'bad id'
    );
    await assert.rejects(
      client(server, { route: () => ({ body: undefined }) }).decide(
        'jev',
        request
      ),
      /route must return \{path, body\}, with a string path if any and a body/
    );
    await assert.rejects(
      client(server, { unwrap: async (body) => body }).decide('jev', request),
      /unwrap must return the response, not a promise/
    );
  });

  it('refuses a body that is not JSON before sending it', async () => {
    const before = server.requests.length;
    await assert.rejects(
      client(server).decide('jev', {
        ...request,
        extra: { trace: { id: BigInt(1) } },
      }),
      (e: GenkitError) =>
        e.status === 'INVALID_ARGUMENT' && /request is not JSON/.test(e.message)
    );
    assert.strictEqual(server.requests.length, before);
  });

  it('sends a per-request key, and leaves a timeout to a caller fetch', async () => {
    let init: RequestInit | undefined;
    const own = new Client({
      fetch: async (url, i) => {
        init = i;
        return globalThis.fetch(url, i);
      },
      baseUrl: server.url,
      apiKey: 'plugin-key',
      endpoint: { name: 'test', path: '/v1/systemone', modelsPath: '' },
    });
    await own.decide('jev', request, { apiKey: 'request-key' });
    assert.strictEqual(
      server.requests.at(-1)!.headers.authorization,
      'Bearer request-key'
    );
    // No caller signal and no timeout: a signal of the attempt's own, which
    // ends it when its body is refused.
    assert.strictEqual(init?.signal?.aborted, false);
    assert.strictEqual(init?.redirect, 'manual');
  });

  it('replaces an authorization header given in another case', async () => {
    await client(
      server,
      {},
      { headers: { authorization: 'Bearer other', accept: 'text/plain' } }
    ).decide('jev', request);
    const { headers } = server.requests.at(-1)!;
    assert.strictEqual(headers.authorization, 'Bearer key');
    assert.strictEqual(headers.accept, 'application/json');
  });

  it('fails once, with a status, on a key HTTP cannot carry', async () => {
    let calls = 0;
    const c = new Client({
      fetch: async () => {
        calls++;
        return new Response('{}');
      },
      baseUrl: 'http://unused',
      endpoint: { name: 'test', path: '', modelsPath: '' },
      sleep: async () => {},
    });
    await assert.rejects(
      c.decide('jev', request, { apiKey: 'key\nwith-newline' }),
      (e: GenkitError) => e.status === 'INVALID_ARGUMENT'
    );
    assert.strictEqual(calls, 0);
  });

  it('refuses a route that returns no body', async () => {
    for (const body of [undefined, null]) {
      await assert.rejects(
        client(server, { route: () => ({ body }) }).decide('jev', request),
        (e: GenkitError) => e.status === 'INVALID_ARGUMENT',
        String(body)
      );
    }
  });

  it('says what a network error was', async () => {
    const down = new Client({
      fetch: async () => {
        throw new TypeError('fetch failed', {
          cause: new Error('connect ECONNREFUSED 127.0.0.1:9'),
        });
      },
      baseUrl: 'http://unused',
      endpoint: { name: 'test', path: '', modelsPath: '' },
      sleep: async () => {},
    });
    await assert.rejects(
      down.decide('jev', request),
      (e: GenkitError) =>
        e.originalMessage ===
        'test: fetch failed: connect ECONNREFUSED 127.0.0.1:9'
    );
  });

  it('refuses a promise from a hook without leaving its rejection unhandled', async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    try {
      await assert.rejects(
        client(server, {
          route: () => Promise.reject(new Error('route failed')) as any,
        }).decide('jev', request),
        (e: GenkitError) => e.status === 'INVALID_ARGUMENT'
      );
      await assert.rejects(
        client(server, {
          unwrap: () => Promise.reject(new Error('unwrap failed')),
        }).decide('jev', request),
        (e: GenkitError) =>
          e.status === 'INVALID_ARGUMENT' &&
          /unwrap must return the response, not a promise/.test(e.message)
      );
      await new Promise((resolve) => setTimeout(resolve, 20));
      assert.deepStrictEqual(unhandled, []);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });

  it('refuses a route path that is not a string', async () => {
    await assert.rejects(
      client(server, {
        route: (_model, body) => ({ path: 42 as any, body }),
      }).decide('jev', request),
      (e: GenkitError) =>
        e.status === 'INVALID_ARGUMENT' && /with a string path/.test(e.message)
    );
  });

  it("treats a browser's opaque redirect as a redirect", async () => {
    const browser = new Client({
      fetch: async () =>
        ({
          type: 'opaqueredirect',
          status: 0,
          ok: false,
          statusText: '',
          headers: new Headers(),
          body: null,
        }) as unknown as Response,
      baseUrl: 'http://unused',
      endpoint: { name: 'test', path: '', modelsPath: '' },
      sleep: async () => {},
    });
    await assert.rejects(
      browser.decide('jev', request),
      (e: GenkitError) =>
        e.status === 'FAILED_PRECONDITION' &&
        /redirects are not followed/.test(e.message)
    );
  });

  it('lists models from a bare array or an envelope', async () => {
    const c = client(server, { modelsPath: '/v1/models' });
    server.models = '[{"name":"jev-1.13.0"}]';
    assert.deepStrictEqual(await c.listModels(), [{ name: 'jev-1.13.0' }]);
    server.models = '{"data":[{"id":"liquid/d1","name":"Liquid: d1"}]}';
    assert.deepStrictEqual(await c.listModels(), [
      { id: 'liquid/d1', name: 'Liquid: d1' },
    ]);
    server.models = '{"nothing":[]}';
    await assert.rejects(c.listModels(), /has no models array/);
    server.models = undefined;
    await assert.rejects(
      c.listModels(),
      (e: GenkitError) => e.status === 'INTERNAL'
    );
  });
});

describe('errors', () => {
  it('map HTTP codes onto statuses', () => {
    const cases: [number, string][] = [
      [400, 'INVALID_ARGUMENT'],
      [401, 'UNAUTHENTICATED'],
      [402, 'RESOURCE_EXHAUSTED'],
      [403, 'PERMISSION_DENIED'],
      [404, 'NOT_FOUND'],
      [408, 'DEADLINE_EXCEEDED'],
      [409, 'ABORTED'],
      [422, 'INVALID_ARGUMENT'],
      [429, 'RESOURCE_EXHAUSTED'],
      [500, 'INTERNAL'],
      [501, 'UNIMPLEMENTED'],
      [502, 'UNAVAILABLE'],
      [503, 'UNAVAILABLE'],
      [504, 'DEADLINE_EXCEEDED'],
      [505, 'UNIMPLEMENTED'],
      [524, 'DEADLINE_EXCEEDED'],
    ];
    for (const [code, status] of cases) {
      assert.strictEqual(httpError('x', code, '').status, status, String(code));
    }
  });

  it('read the message out of every body shape', () => {
    const cases: [string, string][] = [
      ['{"error":{"message":"nested"}}', 'nested'],
      ['{"error":"flat"}', 'flat'],
      ['{"message":"plain"}', 'plain'],
      ['{"errors":[{"code":7000,"message":"No route"}]}', 'No route'],
      [
        '{"detail":[{"loc":["body","questions"],"msg":"too many"}]}',
        'body.questions: too many',
      ],
      ['{"detail":"Not authenticated"}', 'Not authenticated'],
      ['[{"path":["state"],"message":"Required"}]', 'state: Required'],
      // As in Go, a list with a record that is not an object is no list.
      ['[{"message":"a"},"oops"]', '[{"message":"a"},"oops"]'],
      ['  upstream exploded  ', 'upstream exploded'],
      ['', ''],
    ];
    for (const [body, message] of cases) {
      assert.strictEqual(errorMessage(body), message, body);
    }
    assert.strictEqual(errorMessage('x'.repeat(300)), 'x'.repeat(200) + '...');
    assert.strictEqual(
      errorMessage(JSON.stringify({ error: { message: 'y'.repeat(900) } })),
      'y'.repeat(500) + '...'
    );
    assert.strictEqual(
      httpError('x', 429, '').originalMessage,
      'x: HTTP 429: Too Many Requests'
    );
  });
});

describe('retry timing', () => {
  it('backs off with a quarter of jitter, capped', () => {
    assert.strictEqual(
      backoff(0, () => 0),
      500
    );
    assert.strictEqual(
      backoff(1, () => 1),
      750
    );
    assert.strictEqual(
      backoff(10, () => 0),
      5000
    );
  });

  it('reads Retry-After as seconds or a date', () => {
    const now = Date.parse('Sun, 22 Nov 2026 12:00:00 GMT');
    assert.strictEqual(parseRetryAfter(null), -1);
    assert.strictEqual(parseRetryAfter('0'), 0);
    assert.strictEqual(parseRetryAfter('3'), 3000);
    assert.strictEqual(
      parseRetryAfter('Sun, 22 Nov 2026 12:00:05 GMT', now),
      5000
    );
    assert.strictEqual(
      parseRetryAfter('Sun, 22 Nov 2026 11:00:00 GMT', now),
      0
    );
    assert.strictEqual(parseRetryAfter('soon'), -1);
    assert.strictEqual(parseRetryAfter('1.5', now), -1);
  });
});
