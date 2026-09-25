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
import { afterEach, describe, it } from 'node:test';
import {
  GENKIT_PINECONE_SOURCE_TAG,
  resolvePineconeConfig,
} from '../src/index.js';

describe('resolvePineconeConfig', () => {
  const originalApiKey = process.env.PINECONE_API_KEY;

  afterEach(() => {
    if (originalApiKey === undefined) {
      delete process.env.PINECONE_API_KEY;
    } else {
      process.env.PINECONE_API_KEY = originalApiKey;
    }
  });

  it('defaults sourceTag to genkit when using env API key', () => {
    process.env.PINECONE_API_KEY = 'test-key';
    const config = resolvePineconeConfig();
    assert.strictEqual(config.apiKey, 'test-key');
    assert.strictEqual(config.sourceTag, GENKIT_PINECONE_SOURCE_TAG);
    assert.strictEqual(config.sourceTag, 'genkit');
  });

  it('defaults sourceTag when clientParams omit it', () => {
    const config = resolvePineconeConfig({ apiKey: 'explicit-key' });
    assert.strictEqual(config.apiKey, 'explicit-key');
    assert.strictEqual(config.sourceTag, GENKIT_PINECONE_SOURCE_TAG);
  });

  it('preserves an explicit sourceTag from clientParams', () => {
    const config = resolvePineconeConfig({
      apiKey: 'explicit-key',
      sourceTag: 'custom_integration',
    });
    assert.strictEqual(config.sourceTag, 'custom_integration');
  });

  it('preserves other clientParams fields', () => {
    const config = resolvePineconeConfig({
      apiKey: 'explicit-key',
      controllerHostUrl: 'https://example.pinecone.io',
      additionalHeaders: { 'X-Custom': '1' },
    });
    assert.strictEqual(config.controllerHostUrl, 'https://example.pinecone.io');
    assert.deepStrictEqual(config.additionalHeaders, { 'X-Custom': '1' });
    assert.strictEqual(config.sourceTag, GENKIT_PINECONE_SOURCE_TAG);
  });

  it('throws when neither clientParams nor env API key is set', () => {
    delete process.env.PINECONE_API_KEY;
    assert.throws(
      () => resolvePineconeConfig(),
      /PINECONE_API_KEY/
    );
  });
});
