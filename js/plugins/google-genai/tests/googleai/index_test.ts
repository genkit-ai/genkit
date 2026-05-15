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
import { genkit } from 'genkit';
import { afterEach, beforeEach, describe, it, mock } from 'node:test';
import {
  TEST_ONLY as EMBEDDER_TEST_ONLY,
  EmbeddingConfigSchema,
} from '../../src/googleai/embedder.js';
import {
  TEST_ONLY as GEMINI_TEST_ONLY,
  GeminiConfigSchema,
  GeminiImageConfigSchema,
  GeminiTtsConfigSchema,
  GemmaConfigSchema,
} from '../../src/googleai/gemini.js';
import {
  TEST_ONLY as IMAGEN_TEST_ONLY,
  ImagenConfigSchema,
} from '../../src/googleai/imagen.js';
import { googleAI } from '../../src/googleai/index.js';
import { Model } from '../../src/googleai/types.js';
import { MISSING_API_KEY_ERROR } from '../../src/googleai/utils.js';
import {
  TEST_ONLY as VEO_TEST_ONLY,
  VeoConfigSchema,
} from '../../src/googleai/veo.js';

describe('GoogleAI Plugin', () => {
  let originalEnv: NodeJS.ProcessEnv;
  let fetchMock: any;

  beforeEach(() => {
    originalEnv = { ...process.env };
    // Set a default API key for most tests
    process.env.GOOGLE_API_KEY = 'test-api-key';

    // Mock global.fetch for listActions tests
    fetchMock = mock.method(global, 'fetch');
  });

  afterEach(() => {
    // Restore environment variables
    process.env = originalEnv;
    // Restore mocks
    mock.restoreAll();
  });

  describe('Initializer', () => {
    it('should pre-register flagship Gemini models', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      const model1Name = Object.keys(GEMINI_TEST_ONLY.KNOWN_MODELS)[0];
      const model1Path = `/model/googleai/${model1Name}`;
      const expectedBaseName = `googleai/${model1Name}`;
      const model1 = await ai.registry.lookupAction(model1Path);
      assert.ok(model1, `${model1Name} should be registered at ${model1Path}`);
      assert.strictEqual(model1?.__action.name, expectedBaseName);
    });

    it('should register all known Gemini models', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      for (const modelName in GEMINI_TEST_ONLY.KNOWN_MODELS) {
        const modelPath = `/model/googleai/${modelName}`;
        const expectedBaseName = `googleai/${modelName}`;
        const model = await ai.registry.lookupAction(modelPath);
        assert.ok(model, `${modelName} should be registered at ${modelPath}`);
        assert.strictEqual(model?.__action.name, expectedBaseName);
      }
    });

    it('should register all known Imagen models', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      for (const modelName in IMAGEN_TEST_ONLY.KNOWN_MODELS) {
        const modelPath = `/model/googleai/${modelName}`;
        const expectedBaseName = `googleai/${modelName}`;
        const model = await ai.registry.lookupAction(modelPath);
        assert.ok(model, `${modelName} should be registered at ${modelPath}`);
        assert.strictEqual(model?.__action.name, expectedBaseName);
      }
    });

    it('should register all known Veo models', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      for (const modelName in VEO_TEST_ONLY.KNOWN_MODELS) {
        const modelPath = `/background-model/googleai/${modelName}`;
        const expectedBaseName = `googleai/${modelName}`;
        const model = await ai.registry.lookupAction(modelPath);
        assert.ok(model, `${modelName} should be registered at ${modelPath}`);
        assert.strictEqual(model?.__action.name, expectedBaseName);
      }
    });

    it('should pre-register flagship Embedder models', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      const modelKeys = Object.keys(EMBEDDER_TEST_ONLY.KNOWN_MODELS);
      if (modelKeys.length > 0) {
        const model1Name = modelKeys[0];
        const model1Path = `/embedder/googleai/${model1Name}`;
        const expectedBaseName = `googleai/${model1Name}`;
        const model1 = await ai.registry.lookupAction(model1Path);
        assert.ok(
          model1,
          `${model1Name} should be registered at ${model1Path}`
        );
        assert.strictEqual(model1?.__action.name, expectedBaseName);
      }
    });

    it('should register all known Embedder models', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      for (const modelName in EMBEDDER_TEST_ONLY.KNOWN_MODELS) {
        const modelPath = `/embedder/googleai/${modelName}`;
        const expectedBaseName = `googleai/${modelName}`;
        const model = await ai.registry.lookupAction(modelPath);
        assert.ok(model, `${modelName} should be registered at ${modelPath}`);
        assert.strictEqual(model?.__action.name, expectedBaseName);
      }
    });

    it('should throw from initializer if no API key is provided', async () => {
      delete process.env.GOOGLE_API_KEY;
      delete process.env.GEMINI_API_KEY;
      delete process.env.GOOGLE_GENAI_API_KEY;
      const plugin = googleAI({});
      await assert.rejects(async () => {
        await plugin.init!();
      }, MISSING_API_KEY_ERROR);
    });

    it('should NOT throw from initializer if apiKey is false', async () => {
      delete process.env.GOOGLE_API_KEY;
      delete process.env.GEMINI_API_KEY;
      delete process.env.GOOGLE_GENAI_API_KEY;
      const plugin = googleAI({ apiKey: false });
      await assert.doesNotReject(async () => {
        await plugin.init!();
      });
    });
  });

  describe('Resolver via lookupAction', () => {
    const testGeminiModelName = 'gemini-custom-pro';
    const testGeminiModelPath = `/model/googleai/${testGeminiModelName}`;
    const testImagenModelName = 'imagen-custom';
    const testImagenModelPath = `/model/googleai/${testImagenModelName}`;
    const testVeoModelName = 'veo-custom';
    const testVeoModelPath = `/background-model/googleai/${testVeoModelName}`;
    const testEmbedderName = 'embedding-custom-001';
    const testEmbedderPath = `/embedder/googleai/${testEmbedderName}`;

    it('should register a new Gemini model when looked up', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      const model = await ai.registry.lookupAction(testGeminiModelPath);
      assert.ok(
        model,
        `${testGeminiModelName} should be resolvable and registered`
      );
      assert.strictEqual(
        model?.__action.name,
        `googleai/${testGeminiModelName}`
      );
    });

    it('should register a new Imagen model when looked up', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      const model = await ai.registry.lookupAction(testImagenModelPath);
      assert.ok(
        model,
        `${testImagenModelName} should be resolvable and registered`
      );
      assert.strictEqual(
        model?.__action.name,
        `googleai/${testImagenModelName}`
      );
    });

    it('should register a new Veo model when looked up', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      const model = await ai.registry.lookupAction(testVeoModelPath);
      assert.ok(
        model,
        `${testVeoModelName} should be resolvable and registered`
      );
      assert.strictEqual(model?.__action.name, `googleai/${testVeoModelName}`);
    });

    it('should register a new Embedder when looked up', async () => {
      const ai = genkit({ plugins: [googleAI()] });
      const embedder = await ai.registry.lookupAction(testEmbedderPath);
      assert.ok(
        embedder,
        `${testEmbedderName} should be resolvable and registered`
      );
      assert.strictEqual(
        embedder?.__action.name,
        `googleai/${testEmbedderName}`
      );
    });
  });

  describe('googleAI.model', () => {
    it('should return a gemini ModelReference with correct schema', () => {
      const modelName = 'gemini-2.5-flash';
      const modelRef = googleAI.model(modelName);
      assert.strictEqual(
        modelRef.name,
        `googleai/${modelName}`,
        'Name should be prefixed'
      );
      assert.ok(
        modelRef.info?.supports?.multiturn,
        'Gemini model should support multiturn'
      );
      assert.strictEqual(
        modelRef.configSchema,
        GeminiConfigSchema,
        'Should have GeminiConfigSchema'
      );
    });

    it('should return a TTS model reference with correct schema', () => {
      const modelRef = googleAI.model('gemini-5.0-tts');
      assert.strictEqual(
        modelRef.configSchema,
        GeminiTtsConfigSchema,
        'Should have GeminiTTsConfigSchema'
      );
      assert.ok(
        !modelRef.info?.supports?.multiturn,
        'Gemini TTS model should not support multiturn'
      );
    });

    it('should have config values for gemini TTS', () => {
      const modelRef = googleAI.model('gemini-5.0-tts', {
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Algenib' },
          },
        },
      });
      assert.strictEqual(
        modelRef.configSchema,
        GeminiTtsConfigSchema,
        'Should have GeminiTTsConfigSchema'
      );
      assert.strictEqual(
        modelRef.config?.speechConfig?.voiceConfig?.prebuiltVoiceConfig
          ?.voiceName,
        'Algenib'
      );
    });

    it('should return a Gemma model reference with correct schema', () => {
      const modelRef = googleAI.model('gemma-new-model');
      assert.strictEqual(
        modelRef.configSchema,
        GemmaConfigSchema,
        'Should have GemmaConfigSchema'
      );
      assert.ok(
        modelRef.info?.supports?.multiturn,
        'Gemma model should support multiturn'
      );
    });

    it('should have config values for gemma', () => {
      const modelRef = googleAI.model('gemma-3-12b-it', {
        temperature: 0.7,
      });
      assert.strictEqual(modelRef.name, 'googleai/gemma-3-12b-it');
      assert.strictEqual(modelRef.config?.temperature, 0.7);
    });

    it('should return an Imagen model reference with correct schema', () => {
      const modelRef = googleAI.model('imagen-new-model');
      assert.strictEqual(
        modelRef.configSchema,
        ImagenConfigSchema,
        'Should have ImagenConfigSchema'
      );
    });

    it('should have config values for imagen model', () => {
      const modelRef = googleAI.model('imagen-new-model', {
        numberOfImages: 4,
        aspectRatio: '16:9',
      });
      assert.strictEqual(
        modelRef.configSchema,
        ImagenConfigSchema,
        'Should have ImagenConfigSchema'
      );
      assert.strictEqual(
        modelRef.config?.numberOfImages,
        4,
        'should have 4 images'
      );
      assert.strictEqual(
        modelRef.config?.aspectRatio,
        '16:9',
        'should be 16:9'
      );
    });

    it('should return an image model reference for new models', () => {
      const modelRef = googleAI.model('gemini-new-image-foo');
      assert.strictEqual(
        modelRef.configSchema,
        GeminiImageConfigSchema,
        'Should have GeminiImageConfigSchema'
      );
      assert.ok(
        modelRef.info?.supports?.multiturn,
        'Gemini Image model should support multiturn'
      );
    });

    it('should return an Image model reference with correct schema', () => {
      const modelRef = googleAI.model('gemini-2.5-flash-image');
      assert.strictEqual(
        modelRef.configSchema,
        GeminiImageConfigSchema,
        'Should have GeminiImageConfigSchema'
      );
      assert.ok(
        modelRef.info?.supports?.multiturn,
        'Gemini Image model should support multiturn'
      );
    });

    it('should have config values for image model', () => {
      const modelRef = googleAI.model('gemini-2.5-flash-image', {
        imageConfig: {
          aspectRatio: '16:9',
          imageSize: '1K',
        },
      });
      assert.strictEqual(
        modelRef.configSchema,
        GeminiImageConfigSchema,
        'Should have GeminiImageConfigSchema'
      );
      assert.deepStrictEqual(
        modelRef.config?.imageConfig,
        {
          aspectRatio: '16:9',
          imageSize: '1K',
        },
        'should have correct imageConfig'
      );
    });

    it('should return a Veo model reference with correct schema', () => {
      const modelRef = googleAI.model('veo-new-model');
      assert.strictEqual(
        modelRef.configSchema,
        VeoConfigSchema,
        'Should have VeoConfigSchema'
      );
    });

    it('should have config values for veo model', () => {
      const modelRef = googleAI.model('veo-new-model', {
        aspectRatio: '9:16',
        durationSeconds: 8,
      });
      assert.strictEqual(
        modelRef.configSchema,
        VeoConfigSchema,
        'Should have VeoConfigSchema'
      );
      assert.strictEqual(
        modelRef.config?.aspectRatio,
        '9:16',
        'should be 9:16'
      );
      assert.strictEqual(
        modelRef.config?.durationSeconds,
        8,
        'should be 8 seconds'
      );
    });

    it('should return a gemini model reference for unknown model names', () => {
      const modelRef = googleAI.model('foo-model');
      assert.strictEqual(
        modelRef.configSchema,
        GeminiConfigSchema,
        'Should have GeminiConfigSchema'
      );
    });

    it('should have gemini config values for unknown model', () => {
      const modelRef = googleAI.model('foo-model', { temperature: 0.3 });
      assert.strictEqual(
        modelRef.configSchema,
        GeminiConfigSchema,
        'Should have GeminiConfigSchema'
      );
      assert.strictEqual(modelRef.config?.temperature, 0.3);
    });

    it('should handle names with googleai/ prefix', () => {
      const modelName = 'googleai/gemini-2.0-pro';
      const modelRef = googleAI.model(modelName);
      assert.strictEqual(modelRef.name, modelName);
    });

    it('should handle names with models/ prefix', () => {
      const modelName = 'models/gemini-2.0-pro';
      const modelRef = googleAI.model(modelName);
      assert.strictEqual(modelRef.name, 'googleai/gemini-2.0-pro');
    });
  });

  describe('googleAI.embedder', () => {
    it('should return an EmbedderReference with correct schema', () => {
      const embedderName = 'gemini-embedding-001';
      const embedderRef = googleAI.embedder(embedderName);
      assert.strictEqual(embedderRef.name, `googleai/${embedderName}`);
      assert.ok(embedderRef.info, 'Should have info');
      assert.strictEqual(
        embedderRef.configSchema,
        EmbeddingConfigSchema,
        'Should have GoogleAIEmbeddingConfigSchema'
      );
    });

    it('should handle names with googleai/ prefix', () => {
      const embedderName = 'googleai/text-embedding-custom';
      const embedderRef = googleAI.embedder(embedderName);
      assert.strictEqual(embedderRef.name, embedderName);
    });

    it('should handle names with embedders/ prefix', () => {
      const embedderName = 'embedders/text-embedding-custom';
      const embedderRef = googleAI.embedder(embedderName);
      assert.strictEqual(embedderRef.name, 'googleai/text-embedding-custom');
    });
  });

  describe('listActions Function', () => {
    const createMockResponse = (models: Array<Partial<Model>>) => {
      const responseBody = { models: models };
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => responseBody,
        text: async () => JSON.stringify(responseBody),
      });
    };

    it('should return an empty array if no models are returned', async () => {
      fetchMock.mock.mockImplementation(async () => createMockResponse([]));
      const plugin = googleAI();
      const actions = await plugin.list!();
      assert.deepStrictEqual(actions, [], 'Should return an empty array');
    });

    it('should return metadata for models and embedders', async () => {
      const mockModels: Partial<Model>[] = [
        {
          name: 'models/gemini-2.5-pro',
          supportedGenerationMethods: ['generateContent'],
        },
        {
          name: 'models/gemini-embedding-001',
          supportedGenerationMethods: ['embedContent'],
        },
        {
          name: 'models/imagen-4.0-generate-001',
          supportedGenerationMethods: ['predict'],
        },
        {
          name: 'models/veo-3.1-generate-preview',
          supportedGenerationMethods: ['predictLongRunning'],
        },
        {
          name: 'models/other-model',
          supportedGenerationMethods: ['other'],
        },
      ];
      fetchMock.mock.mockImplementation(async () =>
        createMockResponse(mockModels)
      );

      const plugin = googleAI();
      const actions = await plugin.list!();
      const actionNames = actions.map((a) => a.name).sort();
      assert.deepStrictEqual(
        actionNames,
        [
          'googleai/gemini-2.5-pro',
          'googleai/imagen-4.0-generate-001',
          'googleai/gemini-embedding-001',
          'googleai/veo-3.1-generate-preview',
        ].sort()
      );

      const modelAction = actions.find(
        (a) => a.name === 'googleai/gemini-2.5-pro'
      );
      assert.strictEqual(modelAction?.actionType, 'model');

      const embedderAction = actions.find(
        (a) => a.name === 'googleai/gemini-embedding-001'
      );
      assert.strictEqual(embedderAction?.actionType, 'embedder');

      const imagenAction = actions.find(
        (a) => a.name === 'googleai/imagen-4.0-generate-001'
      );
      assert.strictEqual(imagenAction?.actionType, 'model');

      const veoAction = actions.find(
        (a) => a.name === 'googleai/veo-3.1-generate-preview'
      );
      assert.strictEqual(veoAction?.actionType, 'model');
    });

    it('should filter out deprecated models', async () => {
      const mockModels = [
        {
          name: 'models/gemini-2.5-flash',
          supportedGenerationMethods: ['generateContent'],
        },
        {
          name: 'models/gemini-pro-deprecated',
          supportedGenerationMethods: ['generateContent'],
          description: 'This model is deprecated.',
        },
        {
          name: 'models/imagen-deprecated',
          supportedGenerationMethods: ['predict'],
          description: 'This model is deprecated.',
        },
        {
          name: 'models/veo-deprecated',
          supportedGenerationMethods: ['predictLongRunning'],
          description: 'This model is deprecated.',
        },
        {
          name: 'models/text-embedding-deprecated',
          supportedGenerationMethods: ['embedContent'],
          description: 'deprecated',
        },
      ];
      fetchMock.mock.mockImplementation(async () =>
        createMockResponse(mockModels)
      );
      const plugin = googleAI();
      const actions = await plugin.list!();
      const actionNames = actions.map((a) => a.name);
      assert.deepStrictEqual(actionNames, ['googleai/gemini-2.5-flash']);
    });

    it('should handle fetch errors gracefully', async () => {
      fetchMock.mock.mockImplementation(async () => {
        return Promise.resolve({
          ok: false,
          status: 500,
          statusText: 'Internal Error',
          json: async () => ({ error: { message: 'API Error' } }),
          text: async () => JSON.stringify({ error: { message: 'API Error' } }),
        });
      });
      const plugin = googleAI();
      const actions = await plugin.list!();
      assert.deepStrictEqual(
        actions,
        [],
        'Should return empty array on fetch error'
      );
    });

    it('should return empty list if API key is missing for listActions', async () => {
      delete process.env.GOOGLE_API_KEY;
      delete process.env.GEMINI_API_KEY;
      delete process.env.GOOGLE_GENAI_API_KEY;

      const plugin = googleAI({ apiKey: false });
      const actions = await plugin.list!();
      assert.deepStrictEqual(
        actions,
        [],
        'Should return empty array if API key is not found'
      );
      assert.strictEqual(fetchMock.mock.callCount(), 0);
    });

    it('should use listActions cache', async () => {
      const mockModels = [
        {
          name: 'models/gemini-1.0-pro',
          supportedGenerationMethods: ['generateContent'],
        },
      ];
      fetchMock.mock.mockImplementation(async () =>
        createMockResponse(mockModels)
      );
      const plugin = googleAI();
      await plugin.list!();
      await plugin.list!();
      assert.strictEqual(
        fetchMock.mock.callCount(),
        1,
        'fetch should only be called once'
      );
    });
  });
});
