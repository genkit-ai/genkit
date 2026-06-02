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
import { Genkit, genkit } from 'genkit';
import { beforeEach, describe, it } from 'node:test';
import { ollama } from '../src/index.js';
import type {
  ListLocalModelsResponse,
  OllamaPluginParams,
} from '../src/types.js';

const MOCK_MODELS_RESPONSE: ListLocalModelsResponse = {
  models: [
    {
      name: 'llama3.2:latest',
      model: 'llama3.2:latest',
      modified_at: '2024-07-22T20:33:28.123648Z',
      size: 1234567890,
      digest: 'sha256:abcdef123456',
      details: {
        parent_model: '',
        format: 'gguf',
        family: 'llama',
        families: ['llama'],
        parameter_size: '8B',
        quantization_level: 'Q4_0',
      },
    },
    {
      name: 'gemma2:latest',
      model: 'gemma2:latest',
      modified_at: '2024-07-22T20:33:28.123648Z',
      size: 987654321,
      digest: 'sha256:fedcba654321',
      details: {
        parent_model: '',
        format: 'gguf',
        family: 'gemma',
        families: ['gemma'],
        parameter_size: '2B',
        quantization_level: 'Q4_0',
      },
    },
    {
      name: 'nomic-embed-text:latest',
      model: 'nomic-embed-text:latest',
      modified_at: '2024-07-22T20:33:28.123648Z',
      size: 456789123,
      digest: 'sha256:123456789abc',
      details: {
        parent_model: '',
        format: 'gguf',
        family: 'nomic',
        families: ['nomic'],
        parameter_size: '137M',
        quantization_level: 'Q4_0',
      },
    },
  ],
};

// Mock fetch to simulate the Ollama API response
global.fetch = async (input: RequestInfo | URL, options?: RequestInit) => {
  const url = typeof input === 'string' ? input : input.toString();

  if (url.includes('/api/tags')) {
    return new Response(JSON.stringify(MOCK_MODELS_RESPONSE), {
      headers: { 'Content-Type': 'application/json' },
    });
  }

  throw new Error(`Unknown API endpoint: ${url}`);
};

describe('ollama list', () => {
  const options: OllamaPluginParams = {
    serverAddress: 'http://localhost:3000',
  };

  let ai: Genkit;
  beforeEach(() => {
    ai = genkit({
      plugins: [ollama(options)],
    });
  });

  it('should return models with ollama/ prefix to maintain v1 compatibility', async () => {
    const result = await ollama().list!();

    // Should return 2 models (embedding models are filtered out)
    assert.strictEqual(result.length, 2);

    // Check that model names have the ollama/ prefix (maintaining v1 compatibility)
    const modelNames = result.map((m) => m.name);
    assert.ok(modelNames.includes('ollama/llama3.2:latest'));
    assert.ok(modelNames.includes('ollama/gemma2:latest'));
    assert.ok(!modelNames.includes('ollama/nomic-embed-text:latest')); // embedding model filtered out

    // Check that each model has the correct structure
    for (const model of result) {
      assert.ok(model.name);
      assert.ok(model.metadata);
      assert.ok(model.metadata.model);
      const modelInfo = model.metadata.model;
      assert.strictEqual(modelInfo.supports?.multiturn, true);
      assert.strictEqual(modelInfo.supports?.media, true);
      assert.strictEqual(modelInfo.supports?.tools, true);
      assert.strictEqual(modelInfo.supports?.toolChoice, true);
      assert.strictEqual(modelInfo.supports?.systemRole, true);
      assert.strictEqual(modelInfo.supports?.constrained, 'all');
    }
  });

  it('should list models through Genkit instance', async () => {
    const result = await ai.registry.listResolvableActions();

    // Should return 2 models (embedding models are filtered out)
    const modelActions = Object.values(result).filter(
      (action) => action.actionType === 'model'
    );
    assert.strictEqual(modelActions.length, 2);

    // Check that model names have the ollama/ prefix
    const modelNames = modelActions.map((m) => m.name);
    assert.ok(modelNames.includes('ollama/llama3.2:latest'));
    assert.ok(modelNames.includes('ollama/gemma2:latest'));
    assert.ok(!modelNames.includes('ollama/nomic-embed-text:latest')); // embedding model filtered out
  });
});
