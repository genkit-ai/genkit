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
import { Genkit, genkit } from 'genkit';
import { beforeEach, describe, it } from 'node:test';
import { defineOllamaEmbedder } from '../src/embeddings.js'; // Adjust the import path as necessary
import { ollama } from '../src/index.js';
import type { OllamaPluginParams } from '../src/types.js'; // Adjust the import path as necessary

// Utility function to parse command-line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  const serverAddress =
    args.find((arg) => arg.startsWith('--server-address='))?.split('=')[1] ||
    'http://localhost:11434';
  const modelName =
    args.find((arg) => arg.startsWith('--model-name='))?.split('=')[1] ||
    'nomic-embed-text';
  return { serverAddress, modelName };
}
const { serverAddress, modelName } = parseArgs();
describe('defineOllamaEmbedder - Live Tests (without genkit)', () => {
  const options: OllamaPluginParams = {
    models: [{ name: modelName }],
    serverAddress,
  };

  it('should successfully return embeddings for single document', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const result = await embedder({
      input: [{ content: [{ text: 'Hello, world!' }] }],
    });

    assert.strictEqual(result.embeddings.length, 1);
    assert.strictEqual(result.embeddings[0].embedding.length, 768);
    assert.ok(Array.isArray(result.embeddings[0].embedding));
    assert.ok(
      result.embeddings[0].embedding.every((val) => typeof val === 'number')
    );
  });

  it('should successfully return embeddings for multiple documents', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-multi',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const result = await embedder({
      input: [
        { content: [{ text: 'First document about machine learning' }] },
        {
          content: [{ text: 'Second document about artificial intelligence' }],
        },
        { content: [{ text: 'Third document about neural networks' }] },
      ],
    });

    assert.strictEqual(result.embeddings.length, 3);
    result.embeddings.forEach((embedding, index) => {
      assert.strictEqual(
        embedding.embedding.length,
        768,
        `Embedding ${index} should have 768 dimensions`
      );
      assert.ok(
        Array.isArray(embedding.embedding),
        `Embedding ${index} should be an array`
      );
      assert.ok(
        embedding.embedding.every((val) => typeof val === 'number'),
        `Embedding ${index} should contain only numbers`
      );
    });
  });

  it('should return different embeddings for different texts', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-different',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const result1 = await embedder({
      input: [
        { content: [{ text: 'The quick brown fox jumps over the lazy dog' }] },
      ],
    });

    const result2 = await embedder({
      input: [
        {
          content: [
            { text: 'Machine learning is a subset of artificial intelligence' },
          ],
        },
      ],
    });

    assert.strictEqual(result1.embeddings.length, 1);
    assert.strictEqual(result2.embeddings.length, 1);

    const embedding1 = result1.embeddings[0].embedding;
    const embedding2 = result2.embeddings[0].embedding;

    assert.notDeepStrictEqual(
      embedding1,
      embedding2,
      'Different texts should produce different embeddings'
    );

    assert.strictEqual(embedding1.length, 768);
    assert.strictEqual(embedding2.length, 768);
  });

  it('should handle empty text gracefully', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-empty',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const result = await embedder({
      input: [{ content: [{ text: '' }] }],
    });

    assert.strictEqual(result.embeddings.length, 1);
    assert.strictEqual(result.embeddings[0].embedding.length, 768);
    assert.ok(Array.isArray(result.embeddings[0].embedding));
  });

  it('should handle long text', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-long',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const longText =
      'Lorem ipsum dolor sit amet, consectetur adipiscing elit. '.repeat(100);

    const result = await embedder({
      input: [{ content: [{ text: longText }] }],
    });

    assert.strictEqual(result.embeddings.length, 1);
    assert.strictEqual(result.embeddings[0].embedding.length, 768);
    assert.ok(Array.isArray(result.embeddings[0].embedding));
    assert.ok(
      result.embeddings[0].embedding.every((val) => typeof val === 'number')
    );
  });
});

describe('defineOllamaEmbedder - Live Tests (with genkit)', () => {
  let ai: Genkit;
  const options: OllamaPluginParams = {
    models: [{ name: modelName }],
    serverAddress,
  };

  beforeEach(() => {
    ai = genkit({
      plugins: [ollama(options)],
    });
  });

  it('should successfully return embeddings through genkit', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-genkit',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const result = await ai.embed({
      embedder,
      content: 'Hello, world!',
    });

    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0].embedding.length, 768);
    assert.ok(Array.isArray(result[0].embedding));
    assert.ok(result[0].embedding.every((val) => typeof val === 'number'));
  });

  it('should handle multiple documents through genkit', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-genkit-multi',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const result = await ai.embedMany({
      embedder,
      content: [
        'First document about machine learning',
        'Second document about artificial intelligence',
        'Third document about neural networks',
      ],
    });

    assert.strictEqual(result.length, 3);
    result.forEach((embedding, index) => {
      assert.strictEqual(
        embedding.embedding.length,
        768,
        `Embedding ${index} should have 768 dimensions`
      );
      assert.ok(
        Array.isArray(embedding.embedding),
        `Embedding ${index} should be an array`
      );
      assert.ok(
        embedding.embedding.every((val) => typeof val === 'number'),
        `Embedding ${index} should contain only numbers`
      );
    });
  });

  it('should return different embeddings for different texts through genkit', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-genkit-different',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const result1 = await ai.embed({
      embedder,
      content: 'The quick brown fox jumps over the lazy dog',
    });

    const result2 = await ai.embed({
      embedder,
      content: 'Machine learning is a subset of artificial intelligence',
    });

    assert.strictEqual(result1.length, 1);
    assert.strictEqual(result2.length, 1);

    const embedding1 = result1[0].embedding;
    const embedding2 = result2[0].embedding;

    assert.notDeepStrictEqual(
      embedding1,
      embedding2,
      'Different texts should produce different embeddings'
    );

    assert.strictEqual(embedding1.length, 768);
    assert.strictEqual(embedding2.length, 768);
  });

  it('should handle empty text gracefully through genkit', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-genkit-empty',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const result = await ai.embed({
      embedder,
      content: '',
    });

    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0].embedding.length, 768);
    assert.ok(Array.isArray(result[0].embedding));
  });

  it('should handle long text through genkit', async () => {
    const embedder = defineOllamaEmbedder({
      name: 'live-test-embedder-genkit-long',
      modelName: modelName,
      dimensions: 768,
      options,
    });

    const longText =
      'Lorem ipsum dolor sit amet, consectetur adipiscing elit. '.repeat(100);

    const result = await ai.embed({
      embedder,
      content: longText,
    });

    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0].embedding.length, 768);
    assert.ok(Array.isArray(result[0].embedding));
    assert.ok(result[0].embedding.every((val) => typeof val === 'number'));
  });
});
