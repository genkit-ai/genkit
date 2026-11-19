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

import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import * as http from 'http';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { action } from '../src/action.js';
import { GenkitError } from '../src/error.js';
import { initNodeFeatures } from '../src/node.js';
import { ReflectionServer } from '../src/reflection.js';
import { Registry } from '../src/registry.js';
import { enableTelemetry } from '../src/tracing.js';
import { TestSpanExporter } from './utils.js';

initNodeFeatures();
const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessor: new SimpleSpanProcessor(spanExporter),
});

describe('ReflectionServer API', () => {
  let registry: Registry;
  let server: ReflectionServer;
  let port: number;

  beforeEach(async () => {
    registry = new Registry();
    // 0: OS-assigned. A port from getPort() would now be bound exactly and
    // could race other test files running in parallel.
    server = new ReflectionServer(registry, { port: 0 });
    await server.start();
    port = (server as any).server.address().port;
  });

  afterEach(async () => {
    await server.stop();
  });

  async function fetchApi(path: string) {
    return new Promise<{ status: number; body: any }>((resolve, reject) => {
      http
        .get(`http://127.0.0.1:${port}${path}`, (res) => {
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

  async function postApi(path: string, body: any) {
    return new Promise<{ status: number; body: any }>((resolve, reject) => {
      const payload = JSON.stringify(body);
      const req = http.request(
        `http://127.0.0.1:${port}${path}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(payload),
          },
        },
        (res) => {
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
        }
      );
      req.on('error', reject);
      req.write(payload);
      req.end();
    });
  }

  it('returns code 5 (NOT_FOUND) when an action throws NOT_FOUND via /api/runAction', async () => {
    registry.registerAction(
      'custom',
      action({ name: 'missingAction', actionType: 'custom' }, async () => {
        throw new GenkitError({
          status: 'NOT_FOUND',
          message: 'Snapshot not found for action test',
        });
      })
    );

    const res = await postApi('/api/runAction', {
      key: '/custom/missingAction',
      input: {},
    });
    assert.strictEqual(res.status, 200);
    assert.strictEqual(res.body.error.code, 5); // StatusCodes.NOT_FOUND === 5
    assert.match(
      res.body.error.message,
      /NOT_FOUND: Snapshot not found for action test/
    );
  });

  it('cancels the intended action when tracing is disabled', async () => {
    const traceId = '0'.repeat(32);
    const spanId = '0'.repeat(16);
    const finish = new Map<string, () => void>();
    const longAction = action(
      { name: 'untracedAction', actionType: 'custom' },
      async () => undefined
    );
    longAction.run = async (input, options) => {
      options?.onTraceStart?.({ traceId, spanId });
      const result = await new Promise<string>((resolve, reject) => {
        const id = (input as { id: string }).id;
        const onAbort = () => reject(new DOMException('Aborted', 'AbortError'));
        options?.abortSignal?.addEventListener('abort', onAbort, {
          once: true,
        });
        finish.set(id, () => {
          options?.abortSignal?.removeEventListener('abort', onAbort);
          resolve(id);
        });
      });
      return { result, telemetry: { traceId, spanId } };
    };
    registry.registerAction('custom', longAction);

    const startAction = async (id: string) => {
      const response = await fetch(`http://127.0.0.1:${port}/api/runAction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          key: '/custom/untracedAction',
          input: { id },
        }),
      });
      return {
        cancellationId: response.headers.get('x-genkit-trace-id'),
        response,
      };
    };

    const first = await startAction('first');
    const second = await startAction('second');
    const cancel = await postApi('/api/cancelAction', {
      traceId: first.cancellationId,
    });
    finish.get('first')?.();
    finish.get('second')?.();
    const [firstBody, secondBody] = await Promise.all([
      first.response.json(),
      second.response.json(),
    ]);

    assert.strictEqual(cancel.status, 200);
    assert.notStrictEqual(first.cancellationId, second.cancellationId);
    assert.strictEqual(firstBody.error.code, 1);
    assert.strictEqual(secondBody.result, 'second');
  });

  it('tolerates concurrent stop() calls', async () => {
    // A quit request racing a signal handler must not close the server twice
    // (the second close would reject with ERR_SERVER_NOT_RUNNING).
    await Promise.all([server.stop(), server.stop()]);
    await assert.rejects(() => fetchApi('/api/__health'));
  });
});
