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
import { GenerateRequest } from 'genkit/model';
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it, mock } from 'node:test';
import {
  TEST_ONLY as EMBEDDER_TEST_ONLY,
  EmbeddingConfigSchema,
} from '../../src/vertexai/embedder.js';
import {
  TEST_ONLY as GEMINI_TEST_ONLY,
  GeminiConfigSchema,
  GeminiImageConfigSchema,
} from '../../src/vertexai/gemini.js';
import {
  TEST_ONLY as IMAGEN_TEST_ONLY,
  ImagenConfigSchema,
} from '../../src/vertexai/imagen.js';
import { vertexAI } from '../../src/vertexai/index.js';
import type {
  ExpressClientOptions,
  GlobalClientOptions,
  RegionalClientOptions,
} from '../../src/vertexai/types.js';
import {
  NOT_SUPPORTED_IN_EXPRESS_ERROR,
  TEST_ONLY as UTILS_TEST_ONLY,
} from '../../src/vertexai/utils.js';

describe('VertexAI Plugin', () => {
  const regionalMockDerivedOptions: RegionalClientOptions = {
    kind: 'regional' as const,
    location: 'us-central1',
    projectId: 'test-project',
    authClient: {
      getAccessToken: async () => 'fake-test-token',
    } as unknown as GoogleAuth,
  };
  const globalMockDerivedOptions: GlobalClientOptions = {
    kind: 'global' as const,
    location: 'global',
    projectId: 'test-project',
    authClient: {
      getAccessToken: async () => 'fake-test-token',
    } as unknown as GoogleAuth,
  };
  const expressMockDerivedOptions: ExpressClientOptions = {
    kind: 'express' as const,
    apiKey: 'test-express-api-key',
  };
  const notSupportedInExpressErrorMessage = {
    message: NOT_SUPPORTED_IN_EXPRESS_ERROR.message,
  };

  let ai: any;

  // Default to regional options for most tests
  beforeEach(() => {
    UTILS_TEST_ONLY.setMockDerivedOptions(regionalMockDerivedOptions);
    ai = genkit({ plugins: [vertexAI()] });
  });

  afterEach(() => {
    UTILS_TEST_ONLY.setMockDerivedOptions(undefined as any);
  });

  describe('Initializer', () => {
    it('should pre-register flagship Gemini models', async () => {
      const model1Name = Object.keys(GEMINI_TEST_ONLY.KNOWN_GEMINI_MODELS)[0];
      const model1Path = `/model/vertexai/${model1Name}`;
      const expectedBaseName = `vertexai/${model1Name}`;
      const model1 = await ai.registry.lookupAction(model1Path);
      assert.ok(model1, `${model1Name} should be registered at ${model1Path}`);
      assert.strictEqual(model1?.__action.name, expectedBaseName);
    });

    it('should register all known Gemini models', async () => {
      for (const modelName in GEMINI_TEST_ONLY.KNOWN_GEMINI_MODELS) {
        const modelPath = `/model/vertexai/${modelName}`;
        const expectedBaseName = `vertexai/${modelName}`;
        const model = await ai.registry.lookupAction(modelPath);
        assert.ok(model, `${modelName} should be registered at ${modelPath}`);
        assert.strictEqual(model?.__action.name, expectedBaseName);
      }
    });

    it('should register all known Image models', async () => {
      for (const modelName in GEMINI_TEST_ONLY.KNOWN_IMAGE_MODELS) {
        const modelPath = `/model/vertexai/${modelName}`;
        const expectedBaseName = `vertexai/${modelName}`;
        const model = await ai.registry.lookupAction(modelPath);
        assert.ok(model, `${modelName} should be registered at ${modelPath}`);
        assert.strictEqual(model?.__action.name, expectedBaseName);
      }
    });

    it('should pre-register flagship Imagen models', async () => {
      const modelKeys = Object.keys(IMAGEN_TEST_ONLY.KNOWN_MODELS);
      if (modelKeys.length > 0) {
        const model1Name = modelKeys[0];
        const model1Path = `/model/vertexai/${model1Name}`;
        const expectedBaseName = `vertexai/${model1Name}`;
        const model1 = await ai.registry.lookupAction(model1Path);
        assert.ok(
          model1,
          `${model1Name} should be registered at ${model1Path}`
        );
        assert.strictEqual(model1?.__action.name, expectedBaseName);
      }
    });

    it('should register all known Imagen models', async () => {
      for (const modelName in IMAGEN_TEST_ONLY.KNOWN_MODELS) {
        const modelPath = `/model/vertexai/${modelName}`;
        const expectedBaseName = `vertexai/${modelName}`;
        const model = await ai.registry.lookupAction(modelPath);
        assert.ok(model, `${modelName} should be registered at ${modelPath}`);
        assert.strictEqual(model?.__action.name, expectedBaseName);
      }
    });

    it('should pre-register flagship Embedder models', async () => {
      const modelKeys = Object.keys(EMBEDDER_TEST_ONLY.KNOWN_MODELS);
      if (modelKeys.length > 0) {
        const model1Name = modelKeys[0];
        const model1Path = `/embedder/vertexai/${model1Name}`;
        const expectedBaseName = `vertexai/${model1Name}`;
        const model1 = await ai.registry.lookupAction(model1Path);
        assert.ok(
          model1,
          `${model1Name} should be registered at ${model1Path}`
        );
        assert.strictEqual(model1?.__action.name, expectedBaseName);
      }
    });

    it('should register all known Embedder models', async () => {
      for (const modelName in EMBEDDER_TEST_ONLY.KNOWN_MODELS) {
        const modelPath = `/embedder/vertexai/${modelName}`;
        const expectedBaseName = `vertexai/${modelName}`;
        const model = await ai.registry.lookupAction(modelPath);
        assert.ok(model, `${modelName} should be registered at ${modelPath}`);
        assert.strictEqual(model?.__action.name, expectedBaseName);
      }
    });
  });

  describe('Resolver via lookupAction', () => {
    const testModelName = 'gemini-100.0-pro';
    const testModelPath = `/model/vertexai/${testModelName}`;
    const testImagenName = 'imagen-100.0-generate';
    const testImagenPath = `/model/vertexai/${testImagenName}`;
    const testEmbedderName = 'text-embedding-100';
    const testEmbedderPath = `/embedder/vertexai/${testEmbedderName}`;

    it('should register a new Gemini model when looked up', async () => {
      const model = await ai.registry.lookupAction(testModelPath);
      assert.ok(model, `${testModelName} should be resolvable and registered`);
      assert.strictEqual(model?.__action.name, `vertexai/${testModelName}`);
    });

    it('should register a new Imagen model when looked up', async () => {
      const model = await ai.registry.lookupAction(testImagenPath);
      assert.ok(model, `${testImagenName} should be resolvable and registered`);
      assert.strictEqual(model?.__action.name, `vertexai/${testImagenName}`);
    });

    it('should register a new Embedder when looked up', async () => {
      const embedder = await ai.registry.lookupAction(testEmbedderPath);
      assert.ok(
        embedder,
        `${testEmbedderName} should be resolvable and registered`
      );
      assert.strictEqual(
        embedder?.__action.name,
        `vertexai/${testEmbedderName}`
      );
    });
  });

  describe('Helper Functions', () => {
    it('vertexAI.model should return a ModelReference for Gemini with correct schema', () => {
      const modelName = 'gemini-2.5-flash';
      const modelRef = vertexAI.model(modelName);
      assert.strictEqual(
        modelRef.name,
        `vertexai/${modelName}`,
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

    it('vertexAI.model should return a ModelReference for Gemini Image model with correct schema', () => {
      const modelName = 'gemini-3-pro-image-preview';
      const modelRef = vertexAI.model(modelName);
      assert.strictEqual(
        modelRef.name,
        `vertexai/${modelName}`,
        'Name should be prefixed'
      );
      assert.ok(
        modelRef.info?.supports?.multiturn,
        'Gemini model should support multiturn'
      );
      assert.strictEqual(
        modelRef.configSchema,
        GeminiImageConfigSchema,
        'Should have GeminiImageConfigSchema'
      );
    });

    it('vertexAI.model should return a ModelReference for Imagen with correct schema', () => {
      const modelName = 'imagen-3.0-generate-002';
      const modelRef = vertexAI.model(modelName);
      assert.strictEqual(
        modelRef.name,
        `vertexai/${modelName}`,
        'Name should be prefixed'
      );
      assert.ok(modelRef.info, 'Imagen model should have info');
      assert.strictEqual(
        modelRef.configSchema,
        ImagenConfigSchema,
        'Should have ImagenConfigSchema'
      );
    });

    it('vertexAI.embedder should return an EmbedderReference with correct schema', () => {
      const embedderName = 'text-embedding-005';
      const embedderRef = vertexAI.embedder(embedderName);
      assert.strictEqual(embedderRef.name, `vertexai/${embedderName}`);
      assert.ok(embedderRef.info, 'Should have info');
      assert.strictEqual(
        embedderRef.configSchema,
        EmbeddingConfigSchema,
        'Should have VertexEmbeddingConfigSchema'
      );
    });
  });

  describe('listActions Function', () => {
    let fetchMock: any;

    beforeEach(() => {
      fetchMock = mock.method(global, 'fetch');
    });

    afterEach(() => {
      fetchMock.mock.restore();
    });

    const createMockResponse = (models: Array<{ name: string }>) => {
      return Promise.resolve({
        ok: true,
        json: async () => ({ publisherModels: models }),
      });
    };

    it('should return an empty array if no models are returned', async () => {
      fetchMock.mock.mockImplementation(async () => createMockResponse([]));
      const plugin = vertexAI();
      const actions = await plugin.list!();
      assert.deepStrictEqual(actions, [], 'Should return an empty array');
    });

    it('should return metadata for gemini and imagen models, filtering others', async () => {
      const mockModels = [
        { name: 'models/gemini-2.5-pro' },
        { name: 'models/imagen-3.0-generate-001' },
        { name: 'models/gemini-embedding-001' },
        { name: 'models/other-model' },
      ];
      fetchMock.mock.mockImplementation(async () =>
        createMockResponse(mockModels)
      );
      const plugin = vertexAI();
      const actions = await plugin.list!();
      const actionNames = actions.map((a) => a.name).sort();
      assert.deepStrictEqual(
        actionNames,
        ['vertexai/gemini-2.5-pro', 'vertexai/imagen-3.0-generate-001'].sort()
      );
      actions.forEach((action) => {
        assert.strictEqual(action.actionType, 'model');
      });
    });

    it('should call fetch with auth token and location-specific URL for local options', async () => {
      fetchMock.mock.mockImplementation(async () => createMockResponse([]));
      const plugin = vertexAI();
      await plugin.list!();

      const fetchCall = fetchMock.mock.calls[0];
      const headers = fetchCall.arguments[1].headers;
      const url = fetchCall.arguments[0];

      assert.strictEqual(headers['Authorization'], 'Bearer fake-test-token');
      assert.strictEqual(headers['x-goog-user-project'], 'test-project');
      assert.ok(
        url.startsWith('https://us-central1-aiplatform.googleapis.com')
      );
    });

    it('should call fetch with API key and global URL for global options', async () => {
      const globalWithOptions = {
        ...globalMockDerivedOptions,
        apiKey: 'test-api-key',
      };
      UTILS_TEST_ONLY.setMockDerivedOptions(globalWithOptions);
      ai = genkit({ plugins: [vertexAI()] }); // Re-init
      fetchMock.mock.mockImplementation(async () => createMockResponse([]));
      const plugin = vertexAI();
      await plugin.list!();

      const fetchCall = fetchMock.mock.calls[0];
      const headers = fetchCall.arguments[1].headers;
      const url = fetchCall.arguments[0];

      assert.strictEqual(headers['Authorization'], 'Bearer fake-test-token');
      assert.strictEqual(headers['x-goog-user-project'], 'test-project');
      assert.strictEqual(headers['x-goog-api-key'], 'test-api-key');
      assert.ok(url.startsWith('https://aiplatform.googleapis.com'));
      assert.ok(!url.includes('?key=test-api-key'));
      assert.ok(!url.includes('us-central1-'));
    });

    it('should throw for listActions with express options', async () => {
      UTILS_TEST_ONLY.setMockDerivedOptions(expressMockDerivedOptions);
      ai = genkit({ plugins: [vertexAI()] }); // Re-init
      fetchMock.mock.mockImplementation(async () => createMockResponse([]));
      const plugin = vertexAI();
      const actions = await plugin.list!();
      assert.strictEqual(actions.length, 0);
      assert.strictEqual(fetchMock.mock.calls.length, 0);
    });
  });

  describe('API Calls', () => {
    let fetchMock: any;

    beforeEach(() => {
      fetchMock = mock.method(global, 'fetch');
    });

    afterEach(() => {
      fetchMock.mock.restore();
    });

    const createMockApiResponse = (data: object) => {
      return Promise.resolve({
        ok: true,
        json: async () => data,
      });
    };

    describe('With Local Options', () => {
      beforeEach(() => {
        UTILS_TEST_ONLY.setMockDerivedOptions(regionalMockDerivedOptions);
        ai = genkit({ plugins: [vertexAI()] });
      });

      it('should use auth token for Gemini generateContent', async () => {
        const modelRef = vertexAI.model('gemini-2.5-flash');
        const generateAction = await ai.registry.lookupAction(
          '/model/' + modelRef.name
        );
        assert.ok(generateAction, `/model/${modelRef.name} action not found`);

        fetchMock.mock.mockImplementation(async () =>
          createMockApiResponse({
            candidates: [
              {
                index: 0,
                content: { role: 'model', parts: [{ text: 'response' }] },
              },
            ],
          })
        );

        await generateAction({
          messages: [{ role: 'user', content: [{ text: 'hi' }] }],
          config: {},
        } as GenerateRequest);

        const fetchCall = fetchMock.mock.calls[0];
        const headers = fetchCall.arguments[1].headers;
        const url = fetchCall.arguments[0];
        assert.strictEqual(headers['Authorization'], 'Bearer fake-test-token');
        assert.ok(url.includes('us-central1-aiplatform.googleapis.com'));
      });

      it('should use auth token for Embedder embedContent', async () => {
        const embedderRef = vertexAI.embedder('gemini-embedding-001');
        const embedAction = await ai.registry.lookupAction(
          '/embedder/' + embedderRef.name
        );
        assert.ok(
          embedAction,
          `/embedder/${embedderRef.name} action not found`
        );

        fetchMock.mock.mockImplementation(async () =>
          createMockApiResponse({
            predictions: [{ embeddings: { values: [0.1] } }],
          })
        );

        await embedAction({
          input: [{ content: [{ text: 'test' }] }],
        });

        const fetchCall = fetchMock.mock.calls[0];
        const headers = fetchCall.arguments[1].headers;
        assert.strictEqual(headers['Authorization'], 'Bearer fake-test-token');
      });

      it('should use auth token for Imagen predict', async () => {
        const modelRef = vertexAI.model('imagen-3.0-generate-001');
        const generateAction = await ai.registry.lookupAction(
          '/model/' + modelRef.name
        );
        assert.ok(generateAction, `/model/${modelRef.name} action not found`);

        fetchMock.mock.mockImplementation(async () =>
          createMockApiResponse({
            predictions: [{ bytesBase64Encoded: 'abc', mimeType: 'image/png' }],
          })
        );

        await generateAction({
          messages: [{ role: 'user', content: [{ text: 'a cat' }] }],
          config: {},
        } as GenerateRequest);
        const fetchCall = fetchMock.mock.calls[0];
        const headers = fetchCall.arguments[1].headers;
        assert.strictEqual(headers['Authorization'], 'Bearer fake-test-token');
      });
    });

    describe('With Global Options', () => {
      beforeEach(() => {
        const globalWithOptions = {
          ...globalMockDerivedOptions,
          apiKey: 'test-api-key',
        };
        UTILS_TEST_ONLY.setMockDerivedOptions(globalWithOptions);
        ai = genkit({ plugins: [vertexAI()] });
      });

      it('should use API key for Gemini generateContent', async () => {
        const modelRef = vertexAI.model('gemini-2.5-flash');
        const generateAction = await ai.registry.lookupAction(
          '/model/' + modelRef.name
        );
        assert.ok(generateAction, `/model/${modelRef.name} action not found`);

        fetchMock.mock.mockImplementation(async () =>
          createMockApiResponse({
            candidates: [
              {
                index: 0,
                content: { role: 'model', parts: [{ text: 'response' }] },
              },
            ],
          })
        );

        await generateAction({
          messages: [{ role: 'user', content: [{ text: 'hi' }] }],
          config: {},
        } as GenerateRequest);

        const fetchCall = fetchMock.mock.calls[0];
        const headers = fetchCall.arguments[1].headers;
        const url = fetchCall.arguments[0];
        assert.strictEqual(headers['Authorization'], 'Bearer fake-test-token');
        assert.strictEqual(headers['x-goog-user-project'], 'test-project');
        assert.strictEqual(headers['x-goog-api-key'], 'test-api-key');
        assert.ok(!url.includes('?key=test-api-key'));
        assert.ok(
          url.includes('aiplatform.googleapis.com') &&
            !url.includes('us-central1-')
        );
      });

      it('should use API key for Embedder embedContent', async () => {
        const embedderRef = vertexAI.embedder('gemini-embedding-001');
        const embedAction = await ai.registry.lookupAction(
          '/embedder/' + embedderRef.name
        );
        assert.ok(
          embedAction,
          `/embedder/${embedderRef.name} action not found`
        );

        fetchMock.mock.mockImplementation(async () =>
          createMockApiResponse({
            predictions: [{ embeddings: { values: [0.1] } }],
          })
        );

        await embedAction({
          input: [{ content: [{ text: 'test' }] }],
        });
        const fetchCall = fetchMock.mock.calls[0];
        const headers = fetchCall.arguments[1].headers;
        const url = fetchCall.arguments[0];
        assert.strictEqual(headers['Authorization'], 'Bearer fake-test-token');
        assert.strictEqual(headers['x-goog-api-key'], 'test-api-key');
        assert.ok(!url.includes('?key=test-api-key'));
      });

      it('should use API key for Imagen predict', async () => {
        const modelRef = vertexAI.model('imagen-3.0-generate-001');
        const generateAction = await ai.registry.lookupAction(
          '/model/' + modelRef.name
        );
        assert.ok(generateAction, `/model/${modelRef.name} action not found`);

        fetchMock.mock.mockImplementation(async () =>
          createMockApiResponse({
            predictions: [{ bytesBase64Encoded: 'abc', mimeType: 'image/png' }],
          })
        );

        await generateAction({
          messages: [{ role: 'user', content: [{ text: 'a cat' }] }],
          config: {},
        } as GenerateRequest);
        const fetchCall = fetchMock.mock.calls[0];
        const headers = fetchCall.arguments[1].headers;
        const url = fetchCall.arguments[0];
        assert.strictEqual(headers['Authorization'], 'Bearer fake-test-token');
        assert.strictEqual(headers['x-goog-api-key'], 'test-api-key');
        assert.ok(!url.includes('?key=test-api-key'));
      });
    });

    describe('With Express Options', () => {
      beforeEach(() => {
        UTILS_TEST_ONLY.setMockDerivedOptions(expressMockDerivedOptions);
        ai = genkit({ plugins: [vertexAI()] });
      });

      it('should use API key for Gemini generateContent', async () => {
        const modelRef = vertexAI.model('gemini-2.5-flash');
        const generateAction = await ai.registry.lookupAction(
          '/model/' + modelRef.name
        );
        assert.ok(generateAction, `/model/${modelRef.name} action not found`);

        fetchMock.mock.mockImplementation(async () =>
          createMockApiResponse({
            candidates: [
              {
                index: 0,
                content: { role: 'model', parts: [{ text: 'response' }] },
              },
            ],
          })
        );

        await generateAction({
          messages: [{ role: 'user', content: [{ text: 'hi' }] }],
          config: {},
        } as GenerateRequest);

        const fetchCall = fetchMock.mock.calls[0];
        const headers = fetchCall.arguments[1].headers;
        const url = fetchCall.arguments[0];

        assert.strictEqual(headers['Authorization'], undefined);
        assert.strictEqual(headers['x-goog-api-key'], 'test-express-api-key');
        assert.ok(!url.includes('test-express-api-key'));
        assert.ok(
          url.includes('aiplatform.googleapis.com') &&
            !url.includes('us-central1-')
        );
      });

      it('should not support Embedder embedContent', async () => {
        const embedderRef = vertexAI.embedder('gemini-embedding-001');
        const embedAction = await ai.registry.lookupAction(
          '/embedder/' + embedderRef.name
        );
        assert.ok(
          embedAction,
          `/embedder/${embedderRef.name} action not found`
        );

        await assert.rejects(async () => {
          await embedAction({
            input: [{ content: [{ text: 'test' }] }],
          });
        }, notSupportedInExpressErrorMessage);
      });

      it('should not support Imagen predict', async () => {
        const modelRef = vertexAI.model('imagen-3.0-generate-001');
        const generateAction = await ai.registry.lookupAction(
          '/model/' + modelRef.name
        );
        assert.ok(generateAction, `/model/${modelRef.name} action not found`);

        await assert.rejects(async () => {
          await generateAction({
            messages: [{ role: 'user', content: [{ text: 'a cat' }] }],
            config: {},
          } as GenerateRequest);
        }, notSupportedInExpressErrorMessage);
      });
    });
  });
});
