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
import { spawn, type ChildProcess } from 'node:child_process';
import { createServer } from 'node:net';
import path from 'node:path';
import { after, before, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { ReflectionClientV1 } from '../src/reflection-client-v1.js';

const here = path.dirname(fileURLToPath(import.meta.url));

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      srv.close(() => resolve(port));
    });
  });
}

const SECRET = 'test-reflection-secret';

/**
 * Drives the V1 client against a *real* Genkit reflection server (run as a
 * plain subprocess rather than a container, so this stays portable). Covers the
 * protocol details the podman runner depends on: GENKIT_REFLECTION_PORT being
 * honored, the secret header, chunk framing, trace headers, and error
 * envelopes. No GENKIT_ENV=dev: the pinned port alone starts the server.
 */
describe('ReflectionClientV1 against a real runtime', () => {
  let child: ChildProcess;
  let client: ReflectionClientV1;
  let baseUrl: string;

  before(async () => {
    const port = await freePort();
    const { GENKIT_ENV: _dropped, ...env } = process.env;
    child = spawn(
      process.execPath,
      ['--import', 'tsx', path.join(here, 'fixtures', 'v1-runtime-entry.ts')],
      {
        env: {
          ...env,
          // The runner relies on this being honored exactly.
          GENKIT_REFLECTION_PORT: String(port),
          GENKIT_REFLECTION_SECRET_TOKEN: SECRET,
        },
        stdio: ['ignore', 'inherit', 'inherit'],
      }
    );
    baseUrl = `http://127.0.0.1:${port}`;
    client = new ReflectionClientV1(baseUrl, { secret: SECRET });
    await client.waitForReady(30_000);
  });

  after(() => {
    child?.kill('SIGTERM');
  });

  it('is rejected without the secret', async () => {
    await assert.rejects(
      () => new ReflectionClientV1(baseUrl).listActions(),
      /401/
    );
  });

  it('lists actions', async () => {
    const actions = await client.listActions();
    assert.ok(Object.keys(actions).some((k) => k.includes('echo')));
  });

  it('runs a non-streaming action and returns a trace id', async () => {
    let traceId: string | undefined;
    const res = await client.runAction<{ echoed: string }>(
      { key: '/flow/echo', input: { text: 'hi' } },
      { onTraceId: (t) => (traceId = t) }
    );
    assert.deepStrictEqual(res.result, { echoed: 'boxed:hi' });
    assert.ok(traceId, 'expected a trace id from the response header');
    assert.strictEqual(res.telemetry?.traceId, traceId);
  });

  it('streams chunks and still returns the final result', async () => {
    const chunks: { i: number }[] = [];
    const res = await client.runAction<{ total: number }>(
      { key: '/flow/countTo', input: { n: 5 } },
      { onChunk: (c) => chunks.push(c as { i: number }) }
    );
    // The last line is the result envelope, not a chunk: it must not leak
    // into onChunk, and the chunks must all arrive.
    assert.deepStrictEqual(res.result, { total: 5 });
    assert.deepStrictEqual(
      chunks.map((c) => c.i),
      [1, 2, 3, 4, 5]
    );
  });

  it('surfaces a missing action as an error', async () => {
    await assert.rejects(() =>
      client.runAction({ key: '/flow/nope', input: {} })
    );
  });
});
