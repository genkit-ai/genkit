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
import { genkit, type ActionMetadata } from 'genkit';
import type { ModelInfo } from 'genkit/model';
import { describe, it } from 'node:test';
import anthropic from '../src/index.js';
import { TEST_ONLY } from '../src/models.js';
import { PluginOptions, __testClient } from '../src/types.js';

import { createMockAnthropicClient } from './mocks/anthropic-client.js';

const { KNOWN_MODELS } = TEST_ONLY;

function getModelInfo(
  metadata: ActionMetadata | undefined
): ModelInfo | undefined {
  return metadata?.metadata?.model as ModelInfo | undefined;
}

describe('Anthropic Plugin', () => {
  it('should register all supported Claude models', async () => {
    const mockClient = createMockAnthropicClient();

    const ai = genkit({
      plugins: [anthropic({ [__testClient]: mockClient } as PluginOptions)],
    });

    for (const modelName of Object.keys(KNOWN_MODELS)) {
      const modelPath = `/model/anthropic/${modelName}`;
      const expectedBaseName = `anthropic/${modelName}`;
      const model = await ai.registry.lookupAction(modelPath);
      assert.ok(model, `${modelName} should be registered at ${modelPath}`);
      assert.strictEqual(model?.__action.name, expectedBaseName);
    }
  });

  it('should throw error when API key is missing', () => {
    // Save original env var if it exists
    const originalApiKey = process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_API_KEY;

    try {
      assert.throws(() => {
        anthropic({} as PluginOptions);
      }, /Please pass in the API key or set the ANTHROPIC_API_KEY environment variable/);
    } finally {
      // Restore original env var
      if (originalApiKey !== undefined) {
        process.env.ANTHROPIC_API_KEY = originalApiKey;
      }
    }
  });

  it('should use API key from environment variable', () => {
    // Save original env var if it exists
    const originalApiKey = process.env.ANTHROPIC_API_KEY;
    const testApiKey = 'test-api-key-from-env';

    try {
      // Set test API key
      process.env.ANTHROPIC_API_KEY = testApiKey;

      // Plugin should initialize without throwing
      const plugin = anthropic({} as PluginOptions);
      assert.ok(plugin);
      assert.strictEqual(plugin.name, 'anthropic');
    } finally {
      // Restore original env var
      if (originalApiKey !== undefined) {
        process.env.ANTHROPIC_API_KEY = originalApiKey;
      } else {
        delete process.env.ANTHROPIC_API_KEY;
      }
    }
  });

  it('should resolve models dynamically via resolve function', async () => {
    const mockClient = createMockAnthropicClient();
    const plugin = anthropic({ [__testClient]: mockClient } as PluginOptions);

    assert.ok(plugin.resolve, 'Plugin should have resolve method');

    // Test resolving a valid model
    const validModel = plugin.resolve!('model', 'anthropic/claude-haiku-4-5');
    assert.ok(validModel, 'Should resolve valid model');
    assert.strictEqual(typeof validModel, 'function');

    // Test resolving an unknown model name - should return a model action
    // (following Google GenAI pattern: accept any model name, let API validate)
    const unknownModel = plugin.resolve!(
      'model',
      'anthropic/unknown-model-xyz'
    );
    assert.ok(unknownModel, 'Should resolve unknown model name');
    assert.strictEqual(
      typeof unknownModel,
      'function',
      'Should return a model action'
    );

    // Test resolving with invalid action type (using 'tool' as invalid for this context)
    const invalidActionType = plugin.resolve!(
      'tool',
      'anthropic/claude-haiku-4-5'
    );
    assert.strictEqual(
      invalidActionType,
      undefined,
      'Should return undefined for invalid action type'
    );
  });

  it('should list available models from API', async () => {
    const mockClient = createMockAnthropicClient({
      modelList: [
        { id: 'claude-haiku-4-5', display_name: 'Claude 4.5 Haiku' },
        {
          id: 'claude-haiku-4-5-latest',
          display_name: 'Claude 4.5 Haiku Latest',
        },
        { id: 'claude-sonnet-4-5', display_name: 'Claude 4.5 Sonnet' },
        { id: 'claude-sonnet-4', display_name: 'Claude 4 Sonnet' },
        { id: 'claude-new-5-20251212', display_name: 'Claude New 5' },
        { id: 'claude-experimental-latest' },
      ],
    });

    const plugin = anthropic({ [__testClient]: mockClient } as PluginOptions);
    assert.ok(plugin.list, 'Plugin should have list method');

    const models = await plugin.list!();

    assert.ok(Array.isArray(models), 'Should return an array');
    assert.ok(models.length > 0, 'Should return at least one model');

    const names = models.map((model) => model.name).sort();
    // Models are listed with their full IDs from the API (no normalization)
    assert.ok(
      names.includes('anthropic/claude-haiku-4-5'),
      'Known model should be listed with full model ID from API'
    );
    assert.ok(
      names.includes('anthropic/claude-haiku-4-5-latest'),
      'Latest variant should be listed separately'
    );
    assert.ok(
      names.includes('anthropic/claude-sonnet-4-5'),
      'Unknown Claude 4.5 Sonnet should be listed with full model ID'
    );
    assert.ok(
      names.includes('anthropic/claude-sonnet-4'),
      'Known Claude Sonnet 4 model should be listed with full model ID'
    );
    assert.ok(
      names.includes('anthropic/claude-new-5-20251212'),
      'Unknown model IDs should surface as-is'
    );
    assert.ok(
      names.includes('anthropic/claude-experimental-latest'),
      'Latest-suffixed unknown models should be surfaced'
    );

    const haikuMetadata = models.find(
      (model) => model.name === 'anthropic/claude-haiku-4-5'
    );
    assert.ok(haikuMetadata, 'Haiku metadata should exist');
    const haikuInfo = getModelInfo(haikuMetadata);
    assert.ok(haikuInfo, 'Haiku model info should exist');

    const newModelMetadata = models.find(
      (model) => model.name === 'anthropic/claude-new-5-20251212'
    );
    assert.ok(newModelMetadata, 'New model metadata should exist');

    const experimentalMetadata = models.find(
      (model) => model.name === 'anthropic/claude-experimental-latest'
    );
    assert.ok(experimentalMetadata, 'Experimental model metadata should exist');

    // Verify mock was called
    const listStub = mockClient.beta.models.list as any;
    assert.strictEqual(
      listStub.mock.calls.length,
      1,
      'beta.models.list should be called once'
    );
  });

  it('should cache list results on subsequent calls?', async () => {
    const mockClient = createMockAnthropicClient({
      modelList: [{ id: 'claude-haiku-4-5', display_name: 'Claude 4.5 Haiku' }],
    });

    const plugin = anthropic({ [__testClient]: mockClient } as PluginOptions);
    assert.ok(plugin.list, 'Plugin should have list method');

    // First call
    const firstResult = await plugin.list!();
    assert.ok(firstResult, 'First call should return results');

    // Second call
    const secondResult = await plugin.list!();
    assert.ok(secondResult, 'Second call should return results');

    // Verify both results are the same (reference equality for cache)
    assert.strictEqual(
      firstResult,
      secondResult,
      'Results should be cached (same reference)'
    );

    // Verify models.list was only called once due to caching
    const listStub = mockClient.beta.models.list as any;
    assert.strictEqual(
      listStub.mock.calls.length,
      1,
      'beta.models.list should only be called once due to caching'
    );
  });
});

describe('Anthropic resolve helpers', () => {
  it('should resolve model names without anthropic/ prefix', () => {
    const mockClient = createMockAnthropicClient();
    const plugin = anthropic({ [__testClient]: mockClient } as PluginOptions);

    const action = plugin.resolve?.('model', 'claude-haiku-4-5');
    assert.ok(action, 'Should resolve model without prefix');
    assert.strictEqual(typeof action, 'function');
  });

  it('anthropic.model should return model reference with config', () => {
    const reference = anthropic.model('claude-haiku-4-5', {
      temperature: 0.25,
    });

    const referenceAny = reference as any;
    assert.ok(referenceAny, 'Model reference should be created');
    assert.ok(referenceAny.name.includes('claude-haiku-4-5'));
    assert.strictEqual(referenceAny.config?.temperature, 0.25);
  });

  it('should apply system prompt caching when caching is enabled', async () => {
    const mockClient = createMockAnthropicClient();
    const plugin = anthropic({
      [__testClient]: mockClient,
    } as PluginOptions);

    const action = plugin.resolve?.('model', 'anthropic/claude-haiku-4-5');
    assert.ok(action, 'Action should be resolved');

    const abortSignal = new AbortController().signal;
    await (action as any)(
      {
        messages: [
          {
            role: 'system',
            content: [
              {
                text: 'You are helpful.',
                metadata: { cache_control: { type: 'ephemeral', ttl: '5m' } },
              },
            ],
          },
        ],
      },
      { streamingRequested: false, sendChunk: () => {}, abortSignal }
    );

    const createStub = mockClient.messages.create as any;
    assert.strictEqual(createStub.mock.calls.length, 1);
    const requestBody = createStub.mock.calls[0].arguments[0];
    assert.ok(Array.isArray(requestBody.system));
    assert.strictEqual(requestBody.system[0].cache_control?.type, 'ephemeral');
  });
});
