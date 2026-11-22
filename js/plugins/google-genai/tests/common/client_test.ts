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
import { logger } from 'genkit/logging';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { TEST_ONLY as googleAI } from '../../src/googleai/client.js';
import { TEST_ONLY as vertexAI } from '../../src/vertexai/client.js';

for (const [name, client] of [
  ['Google AI', googleAI],
  ['Vertex AI', vertexAI],
] as const) {
  describe(`${name} request error logging`, () => {
    let fetchStub: sinon.SinonStub;
    let errors: unknown[][];
    let debug: unknown[][];
    const url = 'https://example.com/generate';

    beforeEach(() => {
      errors = [];
      debug = [];
      fetchStub = sinon.stub(global, 'fetch');
      logger.init({
        level: 'debug',
        debug: (...args: unknown[]) => debug.push(args),
        info: () => {},
        warn: () => {},
        error: (...args: unknown[]) => errors.push(args),
      });
    });

    afterEach(() => {
      sinon.restore();
      logger.init(logger.defaultLogger);
    });

    it('leaves HTTP error logging to the caller and retains error details', async () => {
      const body = { error: { message: 'Too many requests' } };
      fetchStub.resolves(
        new Response(JSON.stringify(body), {
          status: 429,
          statusText: 'Too Many Requests',
          headers: { 'retry-after': '2' },
        })
      );

      await assert.rejects(client.makeRequest(url, {}), (error: unknown) => {
        assert.ok(error instanceof GenkitError);
        assert.strictEqual(error.status, 'RESOURCE_EXHAUSTED');
        assert.deepStrictEqual(error.detail, body);
        assert.strictEqual(error.responseMetadata?.retryAfterMs, 2000);
        logger.error('Request failed', error);
        return true;
      });

      assert.strictEqual(errors.length, 1);
      assert.strictEqual(errors[0][0], 'Request failed');
      assert.strictEqual(debug.length, 1);
      assert.match(String(debug[0][0]), /Too many requests/);
    });

    it('does not log transport failures at ERROR while still throwing', async () => {
      fetchStub.rejects(new TypeError('Connection failed'));

      await assert.rejects(
        client.makeRequest(url, {}),
        /Failed to fetch from https:\/\/example.com\/generate: Connection failed/
      );

      assert.deepStrictEqual(errors, []);
      assert.strictEqual(debug.length, 1);
      assert.strictEqual(debug[0][0], 'Connection failed');
    });
  });
}
