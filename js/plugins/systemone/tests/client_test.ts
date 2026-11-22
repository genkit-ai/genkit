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
import { after, before, describe, it } from 'node:test';
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
      ).decide('jev', request, abort.signal),
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
      ['  upstream exploded  ', 'upstream exploded'],
      ['', ''],
    ];
    for (const [body, message] of cases) {
      assert.strictEqual(errorMessage(body), message, body);
    }
    assert.strictEqual(errorMessage('x'.repeat(300)), 'x'.repeat(200) + '...');
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
