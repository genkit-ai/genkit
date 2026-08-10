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
import { GenerateRequest, getBasicUsageStats } from 'genkit/model';
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { getGenkitClientHeader } from '../../src/common/utils.js';
import { getVertexAIUrl } from '../../src/vertexai/client.js';
import {
  fromImagenResponse,
  toImagenPredictRequest,
} from '../../src/vertexai/converters.js';
import {
  ImagenConfig,
  ImagenConfigSchema,
  ImagenTryOnConfigSchema,
  TEST_ONLY,
  defineModel,
  isImagenModelName,
  model,
} from '../../src/vertexai/imagen.js';
import {
  ClientOptions,
  ImagenPredictRequest,
  ImagenPredictResponse,
  ImagenPrediction,
} from '../../src/vertexai/types.js';

// Helper function to escape special characters for use in a RegExp
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

describe('Vertex AI Imagen', () => {
  describe('KNOWN_IMAGEN_MODELS', () => {
    it('should contain non-zero number of models', () => {
      assert.ok(Object.keys(TEST_ONLY.KNOWN_MODELS).length > 0);
    });
  });

  describe('model()', () => {
    it('should return a ModelReference for a known model', () => {
      const modelName = 'imagen-3.0-generate-002';
      const ref = model(modelName);
      assert.strictEqual(ref.name, `vertexai/${modelName}`);
      assert.ok(ref.info?.supports?.media);
    });

    it('should return a ModelReference for an unknown model using generic info', () => {
      const modelName = 'imagen-unknown-model';
      const ref = model(modelName);
      assert.strictEqual(ref.name, `vertexai/${modelName}`);
      assert.deepStrictEqual(ref.info, TEST_ONLY.GENERIC_MODEL.info);
    });

    it('should apply config to a known model', () => {
      const modelName = 'imagen-3.0-generate-002';
      const config: ImagenConfig = { seed: 123 };
      const ref = model(modelName, config);
      assert.strictEqual(ref.name, `vertexai/${modelName}`);
      assert.deepStrictEqual(ref.config, config);
    });

    it('should apply config to an unknown model', () => {
      const modelName = 'imagen-unknown-model';
      const config: ImagenConfig = { aspectRatio: '16:9' };
      const ref = model(modelName, config);
      assert.strictEqual(ref.name, `vertexai/${modelName}`);
      assert.deepStrictEqual(ref.config, config);
    });

    it('should handle full model path', () => {
      const modelName = 'tunedModels/my-tuned-model';
      const ref = model(modelName);
      assert.strictEqual(ref.name, 'vertexai/tunedModels/my-tuned-model');
    });

    it('should return a ModelReference with TryOn schema', () => {
      const modelName = 'virtual-try-on-001';
      const ref = model(modelName);
      assert.strictEqual(ref.name, `vertexai/${modelName}`);
      assert.strictEqual(ref.configSchema, ImagenTryOnConfigSchema);
    });
  });

  describe('defineImagenModel()', () => {
    let fetchStub: sinon.SinonStub;
    const modelName = 'imagen-test-model';
    let authMock: sinon.SinonStubbedInstance<GoogleAuth>;

    const regionalClientOptions: ClientOptions = {
      kind: 'regional',
      projectId: 'test-project',
      location: 'us-central1',
      authClient: {} as any,
    };

    const globalClientOptions: ClientOptions = {
      kind: 'global',
      projectId: 'test-project',
      location: 'global',
      authClient: {} as any,
      apiKey: 'test-api-key',
    };

    beforeEach(() => {
      fetchStub = sinon.stub(global, 'fetch');
      authMock = sinon.createStubInstance(GoogleAuth);
      authMock.getAccessToken.resolves('test-token');
      regionalClientOptions.authClient = authMock as unknown as GoogleAuth;
      globalClientOptions.authClient = authMock as unknown as GoogleAuth;
    });

    afterEach(() => {
      sinon.restore();
    });

    function mockFetchResponse(body: any, status = 200) {
      const response = new Response(JSON.stringify(body), {
        status: status,
        statusText: status === 200 ? 'OK' : 'Error',
        headers: { 'Content-Type': 'application/json' },
      });
      fetchStub.resolves(Promise.resolve(response));
    }

    function captureModelRunner(
      clientOptions: ClientOptions
    ): (request: GenerateRequest, options: any) => Promise<any> {
      const model = defineModel(modelName, clientOptions);
      return model.run;
    }

    function getExpectedHeaders(
      clientOptions: ClientOptions
    ): Record<string, string | undefined> {
      const headers: Record<string, string | undefined> = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Client': getGenkitClientHeader(),
        'User-Agent': getGenkitClientHeader(),
        Authorization: 'Bearer test-token',
        'x-goog-user-project':
          clientOptions.kind != 'express' ? clientOptions.projectId : '',
      };
      if (clientOptions.apiKey) {
        headers['x-goog-api-key'] = clientOptions.apiKey;
      }
      return headers;
    }

    function runTestsForClientOptions(clientOptions: ClientOptions) {
      const expectedUrl = getVertexAIUrl({
        includeProjectAndLocation: true,
        resourcePath: `publishers/google/models/${modelName}`,
        resourceMethod: 'predict',
        clientOptions,
      });

      it(`should handle location override for ${clientOptions.kind}`, async () => {
        if (clientOptions.kind === 'express') {
          return; // Not applicable
        }
        const request: GenerateRequest<typeof ImagenConfigSchema> = {
          messages: [{ role: 'user', content: [{ text: 'A cat' }] }],
          config: { location: 'europe-west4' },
        };
        const mockPrediction: ImagenPrediction = {
          bytesBase64Encoded: 'abc',
          mimeType: 'image/png',
        };
        mockFetchResponse({ predictions: [mockPrediction] });
        const modelRunner = captureModelRunner(clientOptions);
        await modelRunner(request, {});

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const actualUrl = fetchArgs[0];
        assert.ok(actualUrl.includes('europe-west4'));
      });

      it(`should define a model and call fetch successfully for ${clientOptions.kind}`, async () => {
        const request: GenerateRequest<typeof ImagenConfigSchema> = {
          messages: [{ role: 'user', content: [{ text: 'A cat' }] }],
          candidates: 2,
          config: { seed: 42 },
        };

        const mockPrediction: ImagenPrediction = {
          bytesBase64Encoded: 'abc',
          mimeType: 'image/png',
        };
        const mockResponse: ImagenPredictResponse = {
          predictions: [mockPrediction, mockPrediction],
        };
        mockFetchResponse(mockResponse);

        const modelRunner = captureModelRunner(clientOptions);
        const result = await modelRunner(request, {});

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        let actualUrl = fetchArgs[0];
        assert.strictEqual(actualUrl, expectedUrl);
        assert.strictEqual(fetchArgs[1].method, 'POST');
        assert.deepStrictEqual(
          fetchArgs[1].headers,
          getExpectedHeaders(clientOptions)
        );

        const expectedImagenPredictRequest: ImagenPredictRequest =
          toImagenPredictRequest(request);

        assert.deepStrictEqual(
          JSON.parse(fetchArgs[1].body),
          expectedImagenPredictRequest
        );

        const expectedResponse = fromImagenResponse(mockResponse, request);
        const expectedCandidates = expectedResponse.candidates;
        assert.deepStrictEqual(result.result.candidates, expectedCandidates);
        assert.deepStrictEqual(result.result.usage, {
          ...getBasicUsageStats(request.messages, expectedCandidates as any),
          custom: { generations: 2 },
        });
        assert.deepStrictEqual(result.result.custom, mockResponse);
      });

      it(`should throw an error if model returns no predictions for ${clientOptions.kind}`, async () => {
        const request: GenerateRequest = {
          messages: [{ role: 'user', content: [{ text: 'A dog' }] }],
        };
        mockFetchResponse({ predictions: [] });

        const modelRunner = captureModelRunner(clientOptions);
        await assert.rejects(
          modelRunner(request, {}),
          /Model returned no predictions/
        );
        sinon.assert.calledOnce(fetchStub);
      });

      it(`should propagate network errors from fetch for ${clientOptions.kind}`, async () => {
        const request: GenerateRequest = {
          messages: [{ role: 'user', content: [{ text: 'A fish' }] }],
        };
        const error = new Error('Network Error');
        fetchStub.rejects(error);

        const modelRunner = captureModelRunner(clientOptions);
        await assert.rejects(modelRunner(request, {}), (err: any) => {
          assert.strictEqual(err.name, 'Error');
          assert.match(err.message, /Network Error/);
          return true;
        });
      });

      it(`should handle API error response for ${clientOptions.kind}`, async () => {
        const request: GenerateRequest = {
          messages: [{ role: 'user', content: [{ text: 'A bird' }] }],
        };
        const errorMsg = 'Invalid argument';
        const errorBody = { error: { message: errorMsg, code: 400 } };
        mockFetchResponse(errorBody, 400);

        const modelRunner = captureModelRunner(clientOptions);
        await assert.rejects(modelRunner(request, {}), (err: any) => {
          assert.strictEqual(err.name, 'GenkitError');
          assert.strictEqual(err.status, 'INVALID_ARGUMENT');
          assert.match(
            err.message,
            /Error fetching from .* \[400 Error\] Invalid argument/
          );
          return true;
        });
      });

      it(`should throw a resource exhausted error on 429 for ${clientOptions.kind}`, async () => {
        const request: GenerateRequest = {
          messages: [{ role: 'user', content: [{ text: 'A bird' }] }],
        };
        const errorMsg = 'Too many requests';
        const errorBody = { error: { message: errorMsg, code: 429 } };
        mockFetchResponse(errorBody, 429);

        const modelRunner = captureModelRunner(clientOptions);
        await assert.rejects(modelRunner(request, {}), (err: any) => {
          assert.strictEqual(err.name, 'GenkitError');
          assert.strictEqual(err.status, 'RESOURCE_EXHAUSTED');
          assert.match(
            err.message,
            /Error fetching from .* \[429 Error\] Too many requests/
          );
          return true;
        });
      });
    }

    describe('with RegionalClientOptions', () => {
      runTestsForClientOptions(regionalClientOptions);
    });

    describe('with GlobalClientOptions', () => {
      runTestsForClientOptions(globalClientOptions);
    });

    // ExpressClientOptions does not support Imagen
    // We have 'does not support' tests elsewhere
  });

  describe('ImagenTryOnConfigSchema', () => {
    it('should validate valid config', () => {
      const validConfig = {
        sampleCount: 1,
        storageUri: 'gs://bucket',
      };
      const result = ImagenTryOnConfigSchema.safeParse(validConfig);
      assert.ok(result.success);
    });
  });

  describe('isImagenModelName', () => {
    it('should return true for known models', () => {
      assert.ok(isImagenModelName('imagen-3.0-generate-002'));
    });

    it('should return true for imagen-* models', () => {
      assert.ok(isImagenModelName('imagen-future-model'));
    });

    it('should return true for virtual-try-on-* models', () => {
      assert.ok(isImagenModelName('virtual-try-on-future-model'));
    });

    it('should return false for other models', () => {
      assert.ok(!isImagenModelName('not-imagen-model'));
    });
  });
});
