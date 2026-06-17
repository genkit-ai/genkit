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
import { ReflectionServer } from '../src/reflection.js';
import { Registry } from '../src/registry.js';

describe('ReflectionServer API', () => {
  let registry: Registry;
  let server: ReflectionServer;
  let port: number;

  beforeEach(async () => {
    registry = new Registry();
    server = new ReflectionServer(registry, { port: await getPort() });
    await server.start();
    port = (server as any).server.address().port;
  });

  afterEach(async () => {
    await server.stop();
  });

  async function fetchApi(path: string) {
    return new Promise<{ status: number; body: any }>((resolve, reject) => {
      http
        .get(`http://localhost:${port}${path}`, (res) => {
          let data = '';
          res.on('data', (chunk) => {
            data += chunk;
          });
          res.on('end', () => {
            try {
              resolve({
                status: res.statusCode || 200,
                body: JSON.parse(data),
              });
            } catch (e) {
              resolve({
                status: res.statusCode || 200,
                body: data,
              });
            }
          });
        })
        .on('error', reject);
    });
  }

  it('rejects missing type parameter for /api/values', async () => {
    const res = await fetchApi('/api/values');
    assert.strictEqual(res.status, 400);
    assert.strictEqual(res.body, 'Query parameter "type" is required.');
  });

  it('rejects unsupported type parameter for /api/values', async () => {
    const res = await fetchApi('/api/values?type=foo');
    assert.strictEqual(res.status, 400);
    assert.match(res.body, /is not supported/);
  });

  it('returns defaultModel values', async () => {
    registry.registerValue('defaultModel', 'testModel', 'my-model');
    const res = await fetchApi('/api/values?type=defaultModel');
    assert.strictEqual(res.status, 200);
    assert.deepStrictEqual(res.body, { testModel: 'my-model' });
  });

  it('returns middleware values mapped via toJson if available', async () => {
    registry.registerValue('middleware', 'mw1', {
      name: 'mw1',
      __def: {},
      toJson: () => ({ name: 'mw1', description: 'test mw1' }),
    });
    registry.registerValue('middleware', 'mw2', {
      name: 'mw2', // No toJson
    });

    const res = await fetchApi('/api/values?type=middleware');
    assert.strictEqual(res.status, 200);
    assert.deepStrictEqual(res.body, {
      mw1: { name: 'mw1', description: 'test mw1' },
      mw2: { name: 'mw2' },
    });
  });
});
