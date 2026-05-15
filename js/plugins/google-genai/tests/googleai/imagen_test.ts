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
import { MessageData } from 'genkit';
import { GenerateRequest, getBasicUsageStats } from 'genkit/model';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { getGenkitClientHeader } from '../../src/common/utils.js';
import { getGoogleAIUrl } from '../../src/googleai/client.js';
import {
  ImagenConfig,
  ImagenConfigSchema,
  TEST_ONLY,
  defineModel,
  model,
} from '../../src/googleai/imagen.js';
import {
  ImagenPredictRequest,
  ImagenPredictResponse,
  ImagenPrediction,
} from '../../src/googleai/types.js';
import { MISSING_API_KEY_ERROR } from '../../src/googleai/utils.js';

const { toImagenParameters, fromImagenPrediction } = TEST_ONLY;

describe('Google AI Imagen', () => {
  describe('KNOWN_MODELS', () => {
    it('should contain non-zero number of models', () => {
      assert.ok(Object.keys(TEST_ONLY.KNOWN_MODELS).length > 0);
    });
  });

  describe('model()', () => {
    it('should return a ModelReference for a known model', () => {
      const modelName = 'imagen-4.0-generate-001';
      const ref = model(modelName);
      assert.strictEqual(ref.name, `googleai/${modelName}`);
      assert.ok(ref.info?.supports?.media);
    });

    it('should return a ModelReference for an unknown model using generic info', () => {
      const modelName = 'imagen-unknown-model';
      const ref = model(modelName);
      assert.strictEqual(ref.name, `googleai/${modelName}`);
      assert.deepStrictEqual(ref.info, TEST_ONLY.GENERIC_MODEL.info);
    });

    it('should apply config to a known model', () => {
      const modelName = 'imagen-4.0-generate-001';
      const config: ImagenConfig = { numberOfImages: 2 };
      const ref = model(modelName, config);
      assert.strictEqual(ref.name, `googleai/${modelName}`);
      assert.deepStrictEqual(ref.config, config);
    });

    it('should apply config to an unknown model', () => {
      const modelName = 'imagen-unknown-model';
      const config: ImagenConfig = { aspectRatio: '16:9' };
      const ref = model(modelName, config);
      assert.strictEqual(ref.name, `googleai/${modelName}`);
      assert.deepStrictEqual(ref.config, config);
    });

    it('should handle model name with prefix', () => {
      const modelName = 'models/imagen-4.0-generate-001';
      const ref = model(modelName);
      assert.strictEqual(ref.name, 'googleai/imagen-4.0-generate-001');
    });
  });

  describe('toImagenParameters', () => {
    const baseRequest: GenerateRequest<typeof ImagenConfigSchema> = {
      messages: [],
    };

    it('should set default sampleCount to 1', () => {
      const result = toImagenParameters(baseRequest);
      assert.strictEqual(result.sampleCount, 1);
    });

    it('should use config.numberOfImages for sampleCount', () => {
      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        ...baseRequest,
        config: { numberOfImages: 3 },
      };
      const result = toImagenParameters(request);
      assert.strictEqual(result.sampleCount, 3);
    });

    it('should include other config parameters but exclude apiKey', () => {
      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        ...baseRequest,
        config: {
          aspectRatio: '16:9',
          personGeneration: 'allow_adult',
          apiKey: 'test-key', // This should be excluded from the result
          numberOfImages: 1,
          addWatermark: false,
          foo: 0, // New passthrough with a valid 0 value.
          bar: undefined, // This should be excluded because its undefined
          baz: null, // This should be excluded because its null
        },
      };
      const result = toImagenParameters(request);
      assert.strictEqual(result.sampleCount, 1);
      assert.strictEqual(result.aspectRatio, '16:9');
      assert.strictEqual(result.personGeneration, 'allow_adult');
      assert.strictEqual(result.addWatermark, false);
      assert.strictEqual(result['foo'], 0);
      assert.strictEqual(
        result.hasOwnProperty('bar'),
        false,
        'bar should not be in parameters'
      );
      assert.strictEqual(
        result.hasOwnProperty('baz'),
        false,
        'baz should not be in parameters'
      );
      assert.strictEqual(
        result.hasOwnProperty('apiKey'),
        false,
        'apiKey should not be in parameters'
      );
    });

    it('should omit undefined or null config parameters', () => {
      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        ...baseRequest,
        config: {
          aspectRatio: undefined,
          personGeneration: null as any,
          numberOfImages: 1,
        },
      };
      const result = toImagenParameters(request);
      assert.strictEqual(result.sampleCount, 1);
      assert.strictEqual(result.hasOwnProperty('aspectRatio'), false);
      assert.strictEqual(result.hasOwnProperty('personGeneration'), false);
    });
  });

  describe('fromImagenPrediction', () => {
    it('should convert ImagenPrediction to MediaPart', () => {
      const prediction: ImagenPrediction = {
        bytesBase64Encoded: 'dGVzdGJ5dGVz',
        mimeType: 'image/png',
      };
      const result = fromImagenPrediction(prediction);

      assert.deepStrictEqual(result, {
        media: {
          url: 'data:image/png;base64,dGVzdGJ5dGVz',
          contentType: 'image/png',
        },
      });
    });
  });

  describe('defineModel()', () => {
    let fetchStub: sinon.SinonStub;
    let envStub: sinon.SinonStub;

    const modelName = 'imagen-test-model';
    const defaultApiKey = 'default-api-key';

    beforeEach(() => {
      fetchStub = sinon.stub(global, 'fetch');
      // Stub process.env to control environment variables
      envStub = sinon.stub(process, 'env').value({});
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
      defineOptions: {
        name?: string;
        apiKey?: string | false;
        apiVersion?: string;
        baseUrl?: string;
      } = {}
    ): (request: GenerateRequest, options: any) => Promise<any> {
      const name = defineOptions.name || modelName;
      const apiVersion = defineOptions.apiVersion;
      const baseUrl = defineOptions.baseUrl;
      const apiKey = defineOptions.apiKey;

      const model = defineModel(name, { apiKey, apiVersion, baseUrl });
      return model.run;
    }

    it('should define a model and call fetch successfully', async () => {
      const prompt = 'A cat';
      const requestApiKey = 'request-api-key';

      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        messages: [{ role: 'user', content: [{ text: prompt }] }],
        config: {
          numberOfImages: 2,
          aspectRatio: '1:1',
          apiKey: requestApiKey,
        },
      };

      const mockPrediction: ImagenPrediction = {
        bytesBase64Encoded: 'abc',
        mimeType: 'image/jpeg',
      };
      const mockResponse: ImagenPredictResponse = {
        predictions: [mockPrediction, mockPrediction],
      };
      mockFetchResponse(mockResponse);

      const modelRunner = captureModelRunner({ apiKey: defaultApiKey });
      const result = await modelRunner(request, {});

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;

      const expectedUrl = getGoogleAIUrl({
        resourcePath: `models/${modelName}`,
        resourceMethod: 'predict',
        clientOptions: { apiVersion: undefined, baseUrl: undefined },
      });
      assert.strictEqual(fetchArgs[0], expectedUrl);

      const expectedHeaders = {
        'Content-Type': 'application/json',
        'x-goog-api-key': requestApiKey, // Effective key is from request
        'x-goog-api-client': getGenkitClientHeader(),
      };
      assert.deepStrictEqual(fetchArgs[1].headers, expectedHeaders);
      assert.strictEqual(fetchArgs[1].method, 'POST');

      const expectedImagenPredictRequest: ImagenPredictRequest = {
        instances: [{ prompt: prompt }],
        parameters: toImagenParameters(request),
      };

      assert.deepStrictEqual(
        JSON.parse(fetchArgs[1].body),
        expectedImagenPredictRequest
      );

      const expectedContent =
        mockResponse.predictions.map(fromImagenPrediction);
      const expectedMessage: MessageData = {
        role: 'model',
        content: expectedContent,
      };
      assert.deepStrictEqual(result.result.message, expectedMessage);
      assert.strictEqual(result.result.finishReason, 'stop');
      assert.deepStrictEqual(
        result.result.usage,
        getBasicUsageStats(request.messages, expectedMessage)
      );
      assert.deepStrictEqual(result.result.custom, mockResponse);
    });

    it('should use default apiKey if no request apiKey provided', async () => {
      const prompt = 'A dog';
      const request: GenerateRequest<typeof ImagenConfigSchema> = {
        messages: [{ role: 'user', content: [{ text: prompt }] }],
        config: {
          numberOfImages: 1,
        },
      };
      mockFetchResponse({
        predictions: [{ bytesBase64Encoded: 'dog', mimeType: 'image/jpeg' }],
      });

      const modelRunner = captureModelRunner({ apiKey: defaultApiKey });
      await modelRunner(request, {});
      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      assert.strictEqual(fetchArgs[1].headers['x-goog-api-key'], defaultApiKey);
    });

    it('should handle custom apiVersion and baseUrl', async () => {
      mockFetchResponse({
        predictions: [{ bytesBase64Encoded: 'def', mimeType: 'image/png' }],
      });
      const apiVersion = 'v1test';
      const baseUrl = 'https://test.example.com';

      const modelRunner = captureModelRunner({
        apiKey: defaultApiKey,
        apiVersion,
        baseUrl,
      });
      await modelRunner(
        {
          messages: [{ role: 'user', content: [{ text: 'A dog' }] }],
        },
        {}
      );

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const expectedUrl = getGoogleAIUrl({
        resourcePath: `models/${modelName}`,
        resourceMethod: 'predict',
        clientOptions: { apiVersion, baseUrl },
      });
      assert.strictEqual(fetchArgs[0], expectedUrl);
    });

    it('should throw an error if model returns no predictions', async () => {
      mockFetchResponse({ predictions: [] });
      const modelRunner = captureModelRunner({ apiKey: defaultApiKey });
      await assert.rejects(
        modelRunner(
          {
            messages: [{ role: 'user', content: [{ text: 'A fish' }] }],
          },
          {}
        ),
        /Model returned no predictions/
      );
      sinon.assert.calledOnce(fetchStub);
    });

    it('should propagate network errors from fetch', async () => {
      const error = new Error('Network Error');
      fetchStub.rejects(error);

      const modelRunner = captureModelRunner({ apiKey: defaultApiKey });
      const expectedUrl = getGoogleAIUrl({
        resourcePath: `models/${modelName}`,
        resourceMethod: 'predict',
      });
      await assert.rejects(
        modelRunner(
          {
            messages: [{ role: 'user', content: [{ text: 'A bird' }] }],
          },
          {}
        ),
        new RegExp(`Failed to fetch from ${expectedUrl}: Network Error`)
      );
    });

    it('should handle API error response', async () => {
      const errorBody = { error: { message: 'Invalid argument', code: 400 } };
      mockFetchResponse(errorBody, 400);

      const modelRunner = captureModelRunner({ apiKey: defaultApiKey });
      const expectedUrl = getGoogleAIUrl({
        resourcePath: `models/${modelName}`,
        resourceMethod: 'predict',
      });
      await assert.rejects(
        modelRunner(
          {
            messages: [{ role: 'user', content: [{ text: 'A plane' }] }],
          },
          {}
        ),
        new RegExp(
          `Error fetching from ${expectedUrl}: \\[400 Error\\] Invalid argument`
        )
      );
    });

    it('apiKey false at init, provided in request', async () => {
      const requestApiKey = 'request-api-key';
      mockFetchResponse({
        predictions: [{ bytesBase64Encoded: 'jkl', mimeType: 'image/png' }],
      });

      const modelRunner = captureModelRunner({ apiKey: false });
      await modelRunner(
        {
          messages: [{ role: 'user', content: [{ text: 'A train' }] }],
          config: { apiKey: requestApiKey },
        },
        {}
      );

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      assert.strictEqual(fetchArgs[1].headers['x-goog-api-key'], requestApiKey);
    });

    it('works with apiKey false at init, missing in request', async () => {
      mockFetchResponse({
        predictions: [{ bytesBase64Encoded: 'jkl', mimeType: 'image/png' }],
      });
      const modelRunner = captureModelRunner({ apiKey: false });

      assert.ok(
        await modelRunner(
          {
            messages: [{ role: 'user', content: [{ text: 'A car' }] }],
            config: {},
          },
          {}
        )
      );
      sinon.assert.calledOnce(fetchStub);
    });

    it('defineImagenModel throws if key not found in env or args', async () => {
      // process.env is empty due to envStub in beforeEach
      assert.throws(() => {
        // Explicitly pass undefined for apiKey
        defineModel(modelName, undefined);
      }, MISSING_API_KEY_ERROR);
    });

    it('should use key from env if no key passed to defineImagenModel', async () => {
      const envKey = 'env-api-key';
      envStub.value({ GOOGLE_API_KEY: envKey });

      mockFetchResponse({
        predictions: [{ bytesBase64Encoded: 'mno', mimeType: 'image/png' }],
      });

      const modelRunner = captureModelRunner({ apiKey: undefined }); // No apiKey passed in init
      await modelRunner(
        {
          messages: [{ role: 'user', content: [{ text: 'A bike' }] }],
        },
        {}
      );

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      assert.strictEqual(fetchArgs[1].headers['x-goog-api-key'], envKey);
    });
  });
});
