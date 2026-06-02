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
import { Genkit, genkit } from 'genkit';
import assert from 'node:assert';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { ollama } from '../src';
import { defineOllamaEmbedder } from '../src/embeddings.js';
import type { OllamaPluginParams } from '../src/types.js';

// Store original fetch to restore after tests
const originalFetch = global.fetch;

/**
 * Creates a mock fetch function for embedding tests
 */
function createDefaultMockFetch(): typeof fetch {
  return async (input: RequestInfo | URL, options?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/api/embed')) {
      if (options?.body && JSON.stringify(options.body).includes('fail')) {
        return {
          ok: false,
          statusText: 'Internal Server Error',
          json: async () => ({}),
        } as Response;
      }

      const body = options?.body ? JSON.parse(options.body as string) : {};
      const inputCount = body.input ? body.input.length : 1;

      return {
        ok: true,
        json: async () => ({
          embeddings: Array(inputCount).fill([0.1, 0.2, 0.3]),
        }),
      } as Response;
    }
    throw new Error('Unknown API endpoint');
  };
}

const options: OllamaPluginParams = {
  models: [{ name: 'test-model' }],
  serverAddress: 'http://localhost:3000',
};

describe('defineOllamaEmbedder (without genkit initialization)', () => {
  beforeEach(() => {
    global.fetch = createDefaultMockFetch();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('should successfully return embeddings when called directly', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'test-embedder',
      modelName: 'test-model',
      dimensions: 123,
      options,
    });

    const result = await embedder({
      input: [{ content: [{ text: 'Hello, world!' }] }],
    });

    assert.deepStrictEqual(result, {
      embeddings: [{ embedding: [0.1, 0.2, 0.3] }],
    });
  });

  it('should handle API errors correctly when called directly', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'test-embedder',
      modelName: 'test-model',
      dimensions: 123,
      options,
    });

    await assert.rejects(
      async () => {
        await embedder({
          input: [{ content: [{ text: 'fail' }] }],
        });
      },
      (error) => {
        assert.ok(error instanceof Error);
        assert.strictEqual(
          error.message,
          'Error fetching embedding from Ollama: Internal Server Error. '
        );
        return true;
      }
    );
  });

  it('should handle multiple documents', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'test-embedder',
      modelName: 'test-model',
      dimensions: 123,
      options,
    });

    const result = await embedder({
      input: [
        { content: [{ text: 'First document' }] },
        { content: [{ text: 'Second document' }] },
      ],
    });

    assert.deepStrictEqual(result, {
      embeddings: [
        { embedding: [0.1, 0.2, 0.3] },
        { embedding: [0.1, 0.2, 0.3] },
      ],
    });
  });
});

describe('defineOllamaEmbedder (with genkit initialization)', () => {
  let ai: Genkit;

  beforeEach(() => {
    global.fetch = createDefaultMockFetch();
    ai = genkit({
      plugins: [ollama(options)],
    });
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('should successfully return embeddings', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'test-embedder',
      modelName: 'test-model',
      dimensions: 123,
      options,
    });

    const result = await ai.embed({
      embedder,
      content: 'Hello, world!',
    });

    assert.deepStrictEqual(result, [{ embedding: [0.1, 0.2, 0.3] }]);
  });

  it('should handle API errors correctly', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'test-embedder',
      modelName: 'test-model',
      dimensions: 123,
      options,
    });

    await assert.rejects(
      async () => {
        await ai.embed({
          embedder,
          content: 'fail',
        });

        await embedder({
          input: [{ content: [{ text: 'fail' }] }],
        });
      },
      (error) => {
        assert.ok(error instanceof Error);
        assert.strictEqual(
          error.message,
          'Error fetching embedding from Ollama: Internal Server Error. '
        );
        return true;
      }
    );
  });

  it('should support per-call embedder serverAddress configuration', async () => {
    // Override mock fetch for this specific test
    global.fetch = async (input: RequestInfo | URL, options?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();

      if (url.includes('/api/embed')) {
        // Verify the custom serverAddress was used
        assert.ok(url.includes('http://custom-server:11434'));
        return new Response(
          JSON.stringify({
            embeddings: [[0.1, 0.2, 0.3]],
          }),
          {
            headers: { 'Content-Type': 'application/json' },
          }
        );
      }

      throw new Error(`Unknown API endpoint: ${url}`);
    };

    const aiWithEmbedder = genkit({
      plugins: [
        ollama({
          serverAddress: 'http://localhost:3000',
          embedders: [{ name: 'test-embedder', dimensions: 768 }],
        }),
      ],
    });

    const result = await aiWithEmbedder.embed({
      embedder: 'ollama/test-embedder',
      content: 'test document',
      options: { serverAddress: 'http://custom-server:11434' },
    });

    assert.ok(result);
    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0].embedding.length, 3);
  });

  it('should initialize embedders when serverAddress is not explicitly provided (uses default)', async () => {
    // Override mock fetch for this specific test
    global.fetch = async (input: RequestInfo | URL, options?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();

      if (url.includes('/api/embed')) {
        // Verify the default serverAddress was used
        assert.ok(url.includes('http://localhost:11434'));
        return new Response(
          JSON.stringify({
            embeddings: [[0.4, 0.5, 0.6]],
          }),
          {
            headers: { 'Content-Type': 'application/json' },
          }
        );
      }

      throw new Error(`Unknown API endpoint: ${url}`);
    };

    // Initialize plugin WITHOUT serverAddress, relying on default
    const aiWithDefaultServer = genkit({
      plugins: [
        ollama({
          embedders: [{ name: 'default-embedder', dimensions: 768 }],
        }),
      ],
    });

    const result = await aiWithDefaultServer.embed({
      embedder: 'ollama/default-embedder',
      content: 'test with default server',
    });

    assert.ok(result);
    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0].embedding.length, 3);
    assert.deepStrictEqual(result[0].embedding, [0.4, 0.5, 0.6]);
  });
});
