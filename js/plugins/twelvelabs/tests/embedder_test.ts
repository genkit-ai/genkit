/**
 * Copyright 2025 Google LLC
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
import { genkit, type Genkit } from 'genkit';
import { beforeEach, describe, it } from 'node:test';
import { defineTwelveLabsEmbedder } from '../src/embedder.js';

// Mock fetch to simulate the TwelveLabs /embed endpoint.
global.fetch = (async (input: RequestInfo | URL) => {
  const url = typeof input === 'string' ? input : input.toString();
  if (url.includes('/embed')) {
    return {
      ok: true,
      json: async () => ({
        model_name: 'marengo3.0',
        text_embedding: { segments: [{ float: [0.1, 0.2, 0.3] }] },
      }),
    } as Response;
  }
  throw new Error('Unknown API endpoint');
}) as typeof fetch;

describe('defineTwelveLabsEmbedder', () => {
  let ai: Genkit;
  beforeEach(() => {
    ai = genkit({ plugins: [] });
  });

  it('returns the embedding vector from the response', async () => {
    const embedder = defineTwelveLabsEmbedder(ai, {
      apiKey: 'test-key',
      baseUrl: 'https://api.twelvelabs.io/v1.3',
      embedder: { name: 'marengo3.0', dimensions: 512 },
    });
    const result = await ai.embed({ embedder, content: 'a cat' });
    assert.deepStrictEqual(result, [{ embedding: [0.1, 0.2, 0.3] }]);
  });

  it('surfaces API errors', async () => {
    const prev = global.fetch;
    global.fetch = (async () =>
      ({
        ok: false,
        statusText: 'Internal Server Error',
        json: async () => ({ message: 'boom' }),
      }) as Response) as typeof fetch;
    const embedder = defineTwelveLabsEmbedder(ai, {
      apiKey: 'test-key',
      baseUrl: 'https://api.twelvelabs.io/v1.3',
      embedder: { name: 'marengo3.0', dimensions: 512 },
    });
    await assert.rejects(
      () => ai.embed({ embedder, content: 'a cat' }),
      (error) => {
        assert.ok(error instanceof Error);
        assert.match(error.message, /TwelveLabs.*Internal Server Error.*boom/);
        return true;
      }
    );
    global.fetch = prev;
  });
});
