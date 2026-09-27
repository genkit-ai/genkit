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
import { spawnSync } from 'child_process';
import * as net from 'net';
import { describe, it } from 'node:test';
import * as path from 'path';
import { genkit, getClientHeader } from '../src/index.js';

describe('genkit', () => {
  it('sets the client headers', async () => {
    genkit({});

    assert.ok(getClientHeader().includes('genkit-node/'));
    assert.ok(!getClientHeader().includes('foo'));

    genkit({ clientHeader: 'foo' });

    assert.ok(getClientHeader().includes('genkit-node/'));
    assert.ok(getClientHeader().includes('foo'));
  });

  it('surfaces a reflection bind failure as an unhandled rejection, not an exit', async () => {
    const taken = net.createServer();
    await new Promise<void>((resolve) => taken.listen(0, '127.0.0.1', resolve));
    const address = taken.address();
    assert.ok(address && typeof address === 'object');
    try {
      const result = spawnSync(
        process.execPath,
        [
          '--import',
          'tsx',
          // Relative to the package root, where `pnpm test` runs; tsx loads
          // this file as ESM, so __dirname is not available.
          path.resolve('tests', 'fixtures', 'reflection-bind-failure.ts'),
        ],
        {
          env: {
            ...process.env,
            GENKIT_ENV: 'dev',
            GENKIT_REFLECTION_PORT: '',
            TAKEN_PORT: String(address.port),
          },
          encoding: 'utf8',
          timeout: 30_000,
        }
      );
      // Reaching the app's own handler proves nothing called process.exit.
      assert.match(
        result.stdout,
        /unhandledRejection: .*Reflection server failed to start: .*EADDRINUSE/
      );
      assert.strictEqual(result.status, 7);
    } finally {
      await new Promise<void>((resolve) => taken.close(() => resolve()));
    }
  });
});
