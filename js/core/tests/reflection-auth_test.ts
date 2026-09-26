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
import getPort from 'get-port';
import * as http from 'http';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { initNodeFeatures } from '../src/node.js';
import { REFLECTION_SECRET_HEADER } from '../src/reflection-config.js';
import { ReflectionServer } from '../src/reflection.js';
import { Registry } from '../src/registry.js';

initNodeFeatures();

const SECRET = 'test-secret';

/** Env keys this suite sets, restored after each test. */
const MANAGED_ENV = [
  'GENKIT_ENV',
  'GENKIT_REFLECTION_SECRET_TOKEN',
  'GENKIT_REFLECTION_PORT',
  'GENKIT_REFLECTION_HOST',
  'GENKIT_REFLECTION_DISABLED',
] as const;

function get(
  port: number,
  path: string,
  headers: Record<string, string> = {}
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    http
      .get({ host: '127.0.0.1', port, path, headers }, (res) => {
        let data = '';
        res.on('data', (chunk) => (data += chunk));
        res.on('end', () =>
          resolve({ status: res.statusCode ?? 0, body: data })
        );
      })
      .on('error', reject);
  });
}

describe('ReflectionServer auth', () => {
  let server: ReflectionServer | undefined;
  let saved: Record<string, string | undefined>;

  beforeEach(() => {
    saved = {};
    for (const key of MANAGED_ENV) {
      saved[key] = process.env[key];
      delete process.env[key];
    }
  });

  afterEach(async () => {
    await server?.stop();
    server = undefined;
    for (const key of MANAGED_ENV) {
      if (saved[key] === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = saved[key];
      }
    }
  });

  async function startWithSecret(secret?: string): Promise<number> {
    if (secret) {
      process.env.GENKIT_REFLECTION_SECRET_TOKEN = secret;
    }
    const port = await getPort();
    process.env.GENKIT_REFLECTION_PORT = String(port);
    server = new ReflectionServer(new Registry());
    await server.start();
    return port;
  }

  it('rejects a request with no secret', async () => {
    const port = await startWithSecret(SECRET);
    const res = await get(port, '/api/actions');
    assert.strictEqual(res.status, 401);
    assert.strictEqual(res.body, '');
  });

  it('rejects a request with the wrong secret', async () => {
    const port = await startWithSecret(SECRET);
    const res = await get(port, '/api/actions', {
      [REFLECTION_SECRET_HEADER]: 'nope',
    });
    assert.strictEqual(res.status, 401);
  });

  it('accepts a request with the right secret', async () => {
    const port = await startWithSecret(SECRET);
    const res = await get(port, '/api/actions', {
      [REFLECTION_SECRET_HEADER]: SECRET,
    });
    assert.strictEqual(res.status, 200);
  });

  it('leaves the health endpoint open', async () => {
    const port = await startWithSecret(SECRET);
    const res = await get(port, '/api/__health');
    assert.strictEqual(res.status, 200);
  });

  it('requires nothing when no secret is configured', async () => {
    const port = await startWithSecret();
    const res = await get(port, '/api/actions');
    assert.strictEqual(res.status, 200);
  });

  it('binds the pinned port exactly', async () => {
    const port = await startWithSecret();
    assert.strictEqual((server as any).server.address().port, port);
  });

  it('fails to start when the pinned port is taken', async () => {
    const taken = http.createServer();
    const port = await getPort();
    await new Promise<void>((resolve) =>
      taken.listen(port, '127.0.0.1', resolve)
    );
    try {
      process.env.GENKIT_REFLECTION_PORT = String(port);
      const blocked = new ReflectionServer(new Registry());
      await assert.rejects(() => blocked.start());
    } finally {
      await new Promise<void>((resolve) => taken.close(() => resolve()));
    }
  });

  it('does not serve quitquitquit outside dev', async () => {
    const port = await startWithSecret();
    const res = await get(port, '/api/__quitquitquit');
    assert.strictEqual(res.status, 404);
  });

  // The dev case is deliberately not exercised here: the handler awaits
  // stop(), which waits on the very connection making the request, so any
  // in-process call to it deadlocks. The 404 above is the assertion that
  // matters, since it is what keeps the endpoint out of non-dev processes.

  it('starts nothing when disabled', async () => {
    process.env.GENKIT_REFLECTION_DISABLED = 'true';
    process.env.GENKIT_ENV = 'dev';
    server = new ReflectionServer(new Registry());
    await server.start();
    assert.strictEqual((server as any).server, null);
  });
});
