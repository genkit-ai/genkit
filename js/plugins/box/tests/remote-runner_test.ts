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
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { spawn, type ChildProcess } from 'node:child_process';
import { createServer } from 'node:net';
import path from 'node:path';
import { after, before, describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { box } from '../src/box.js';
import { perRequest } from '../src/route.js';
import { remoteRunner } from '../src/runners/remote-runner.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const SECRET = 'remote-test-secret';

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

/** Starts the v1 runtime fixture the way a platform would (we don't own it). */
function startRuntime(port: number): ChildProcess {
  const { GENKIT_ENV: _dropped, ...env } = process.env;
  return spawn(
    process.execPath,
    ['--import', 'tsx', path.join(here, 'fixtures', 'v1-runtime-entry.ts')],
    {
      env: {
        ...env,
        GENKIT_REFLECTION_ENABLED: 'true',
        GENKIT_REFLECTION_PORT: String(port),
        GENKIT_REFLECTION_SECRET_TOKEN: SECRET,
      },
      stdio: ['ignore', 'inherit', 'inherit'],
    }
  );
}

describe('remoteRunner', () => {
  let child: ChildProcess;
  let url: string;

  before(async () => {
    const port = await freePort();
    url = `http://127.0.0.1:${port}/`; // trailing slash is tolerated
    child = startRuntime(port);
  });

  after(() => {
    child?.kill('SIGTERM');
  });

  it('proxies calls to an already-running runtime', async () => {
    const myBox = box(genkit({}), {
      runner: remoteRunner({ url, secret: SECRET }),
    });
    const echo = myBox.flow({
      name: 'echo',
      inputSchema: z.object({ text: z.string() }),
      outputSchema: z.object({ echoed: z.string() }),
    });
    assert.deepStrictEqual(await echo({ text: 'hi' }), { echoed: 'boxed:hi' });
    await myBox.close();
  });

  it('sends every routing key to the same runtime, and close leaves it running', async () => {
    const runner = remoteRunner({ url, secret: SECRET });
    const a = await runner.acquire('k1');
    const b = await runner.acquire('k2');
    assert.strictEqual(a, b);
    // perRequest releases after each call; nothing is torn down.
    const myBox = box(genkit({}), { runner, route: perRequest });
    const echo = myBox.flow({ name: 'echo' });
    await echo({ text: 'one' });
    await myBox.close();
    assert.ok(Object.keys(await a.listActions()).length > 0);
    assert.strictEqual(child.exitCode, null, 'runtime still up');
  });

  it('retries readiness after a failed check', async () => {
    const port = await freePort();
    const runner = remoteRunner({
      url: `http://127.0.0.1:${port}`,
      secret: SECRET,
      readyTimeoutMs: 300,
    });
    await assert.rejects(runner.acquire('k'), /Timed out/);
    // The runtime comes up later. Wait for it with a fresh runner, then the
    // original one must check again rather than replay its cached failure.
    const late = startRuntime(port);
    try {
      await remoteRunner({
        url: `http://127.0.0.1:${port}`,
        secret: SECRET,
      }).acquire('k');
      const conn = await runner.acquire('k');
      assert.ok(Object.keys(await conn.listActions()).length > 0);
    } finally {
      late.kill('SIGTERM');
    }
  });
});
