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
import { GenerateRequest } from 'genkit/model';
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { getGenkitClientHeader } from '../../src/common/utils.js';
import { getVertexAIUrl } from '../../src/vertexai/client.js';
import {
  fromLyriaResponse,
  toLyriaPredictRequest,
} from '../../src/vertexai/converters.js';
import {
  LyriaConfigSchema,
  TEST_ONLY,
  defineModel,
  model,
} from '../../src/vertexai/lyria.js';
import {
  LyriaPredictResponse,
  RegionalClientOptions,
} from '../../src/vertexai/types.js';

const {
  GENERIC_MODEL,
  KNOWN_LYRIA_LEGACY_MODELS,
  KNOWN_LYRIA_INTERACTIONS_MODELS,
} = TEST_ONLY;

describe('Vertex AI Lyria', () => {
  let fetchStub: sinon.SinonStub;
  let authMock: sinon.SinonStubbedInstance<GoogleAuth>;

  const defaultRegionalClientOptions: RegionalClientOptions = {
    kind: 'regional',
    projectId: 'test-project',
    location: 'us-central1',
    authClient: {} as any,
  };

  beforeEach(() => {
    fetchStub = sinon.stub(global, 'fetch');
    authMock = sinon.createStubInstance(GoogleAuth);

    authMock.getAccessToken.resolves('test-token');
    defaultRegionalClientOptions.authClient = authMock as unknown as GoogleAuth;
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

  function getExpectedHeaders(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
      'X-Goog-Api-Client': getGenkitClientHeader(),
      'User-Agent': getGenkitClientHeader(),
      Authorization: 'Bearer test-token',
      'x-goog-user-project': defaultRegionalClientOptions.projectId,
    };
  }

  describe('model()', () => {
    it('should return a ModelReference for a known legacy model', () => {
      const knownModelName = Object.keys(KNOWN_LYRIA_LEGACY_MODELS)[0];
      const ref = model(knownModelName);
      assert.strictEqual(ref.name, `vertexai/${knownModelName}`);
      assert.ok(ref.info?.supports?.media);
      assert.deepStrictEqual(ref.info?.supports?.output, ['media']);
    });

    it('should return a ModelReference for a known lyria 3 model', () => {
      const knownModelName = Object.keys(KNOWN_LYRIA_INTERACTIONS_MODELS)[0];
      const ref = model(knownModelName);
      assert.strictEqual(ref.name, `vertexai/${knownModelName}`);
      assert.ok(ref.info?.supports?.media);
      assert.deepStrictEqual(ref.info?.supports?.output, ['text', 'media']);
    });

    it('should return a ModelReference for an unknown model using generic info', () => {
      const unknownModelName = 'lyria-unknown-model';
      const ref = model(unknownModelName);
      assert.strictEqual(ref.name, `vertexai/${unknownModelName}`);
      assert.deepStrictEqual(ref.info, GENERIC_MODEL.info);
      assert.deepStrictEqual(ref.info?.supports?.output, ['text', 'media']);
    });

    it('should apply config to a known model', () => {
      const knownModelName = Object.keys(KNOWN_LYRIA_LEGACY_MODELS)[0];
      const config = { negativePrompt: 'noisy' };
      const ref = model(knownModelName, config);
      assert.strictEqual(ref.name, `vertexai/${knownModelName}`);
      assert.deepStrictEqual(ref.config, config);
    });
  });

  describe('defineModel()', () => {
    const prompt = 'A funky bass line';
    const minimalRequest: GenerateRequest<typeof LyriaConfigSchema> = {
      messages: [{ role: 'user', content: [{ text: prompt }] }],
      config: { sampleCount: 2 },
    };

    // Test with a legacy model name so it hits the predict endpoint
    const legacyModelName = Object.keys(KNOWN_LYRIA_LEGACY_MODELS)[0];

    const mockPrediction: LyriaPredictResponse = {
      predictions: [
        {
          bytesBase64Encoded: 'base64audio1',
          mimeType: 'audio/wav',
        },
        {
          bytesBase64Encoded: 'base64audio2',
          mimeType: 'audio/wav',
        },
      ],
    };

    it('should call fetch with correct params and return lyria response for legacy models', async () => {
      mockFetchResponse(mockPrediction);

      const model = defineModel(legacyModelName, defaultRegionalClientOptions);
      const result = await model.run(minimalRequest);

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const url = fetchArgs[0];
      const options = fetchArgs[1];

      const expectedUrl = getVertexAIUrl({
        includeProjectAndLocation: true,
        resourcePath: `publishers/google/models/${legacyModelName}`,
        resourceMethod: 'predict',
        clientOptions: defaultRegionalClientOptions,
      });
      assert.strictEqual(url, expectedUrl);
      assert.strictEqual(options.method, 'POST');
      assert.deepStrictEqual(options.headers, getExpectedHeaders());

      const expectedPredictRequest = toLyriaPredictRequest(minimalRequest);
      assert.deepStrictEqual(JSON.parse(options.body), expectedPredictRequest);

      const expectedResponse = fromLyriaResponse(
        mockPrediction,
        minimalRequest
      );
      assert.deepStrictEqual(
        result.result.candidates,
        expectedResponse.candidates
      );
      assert.deepStrictEqual(result.result.usage, expectedResponse.usage);
      assert.deepStrictEqual(result.result.custom, expectedResponse.custom);
      assert.strictEqual(result.result.candidates?.length, 2);
      assert.strictEqual(
        result.result.candidates[0].message.content[0].media?.url,
        'data:audio/wav;base64,base64audio1'
      );
    });

    it('should call fetch with correct params and return interaction response for lyria-3 models', async () => {
      const interactionResponse = {
        id: '123',
        status: 'completed',
        outputs: [
          { type: 'audio', mime_type: 'audio/mpeg', data: 'base64audio1' },
          { type: 'text', text: 'Lyrics here' },
        ],
      };
      mockFetchResponse(interactionResponse);

      const lyria3ModelName = Object.keys(KNOWN_LYRIA_INTERACTIONS_MODELS)[0];
      const model = defineModel(lyria3ModelName, defaultRegionalClientOptions);

      const lyria3Request: GenerateRequest<typeof LyriaConfigSchema> = {
        messages: [{ role: 'user', content: [{ text: prompt }] }],
      };
      const result = await model.run(lyria3Request);

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const url = fetchArgs[0];
      const options = fetchArgs[1];

      const expectedUrl = getVertexAIUrl({
        includeProjectAndLocation: true,
        resourcePath: `interactions`,
        clientOptions: defaultRegionalClientOptions,
      });
      assert.strictEqual(url, expectedUrl);
      assert.strictEqual(options.method, 'POST');
      assert.deepStrictEqual(options.headers, getExpectedHeaders());

      assert.deepStrictEqual(JSON.parse(options.body), {
        model: lyria3ModelName,
        input: [
          {
            role: 'user',
            content: [{ type: 'text', text: prompt }],
          },
        ],
        response_modalities: ['audio', 'text'],
      });
      assert.strictEqual(result.result.message?.content.length, 2);
      assert.strictEqual(
        result.result.message?.content[0].media?.url,
        'data:audio/mpeg;base64,base64audio1'
      );
      assert.strictEqual(result.result.message?.content[1].text, 'Lyrics here');
    });

    it('should handle location override', async () => {
      mockFetchResponse(mockPrediction);
      const request: GenerateRequest<typeof LyriaConfigSchema> = {
        messages: [{ role: 'user', content: [{ text: prompt }] }],
        config: { location: 'global' },
      };
      const model = defineModel(legacyModelName, defaultRegionalClientOptions);
      await model.run(request);
      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const actualUrl = fetchArgs[0];
      assert.ok(actualUrl.includes('aiplatform.googleapis.com'));
      assert.ok(!actualUrl.includes('us-central1'));
    });

    it('should throw if no predictions are returned', async () => {
      mockFetchResponse({ predictions: [] });
      const model = defineModel(legacyModelName, defaultRegionalClientOptions);
      await assert.rejects(
        model.run(minimalRequest),
        /Model returned no predictions/
      );
    });

    it('should throw if interaction fails', async () => {
      mockFetchResponse({ status: 'failed' });
      const model = defineModel(
        Object.keys(KNOWN_LYRIA_INTERACTIONS_MODELS)[0],
        defaultRegionalClientOptions
      );
      await assert.rejects(
        model.run({
          messages: [{ role: 'user', content: [{ text: prompt }] }],
        }),
        /Interaction failed/
      );
    });

    it('should propagate API errors', async () => {
      const errorBody = { error: { message: 'Quota exceeded', code: 429 } };
      mockFetchResponse(errorBody, 429);

      const model = defineModel(legacyModelName, defaultRegionalClientOptions);
      await assert.rejects(
        model.run(minimalRequest),
        /Error fetching from .*predict.* Quota exceeded/
      );
    });

    it('should pass AbortSignal to fetch', async () => {
      mockFetchResponse(mockPrediction);
      const controller = new AbortController();
      const abortSignal = controller.signal;

      const clientOptionsWithSignal = {
        ...defaultRegionalClientOptions,
        signal: abortSignal,
      };
      const model = defineModel(legacyModelName, clientOptionsWithSignal);

      await model.run(minimalRequest, { abortSignal });

      sinon.assert.calledOnce(fetchStub);
      const fetchOptions = fetchStub.lastCall.args[1];
      assert.ok(fetchOptions.signal, 'Fetch options should have a signal');

      const fetchSignal = fetchOptions.signal;
      const abortSpy = sinon.spy();
      fetchSignal.addEventListener('abort', abortSpy);

      controller.abort();
      sinon.assert.calledOnce(abortSpy);
    });
  });
});
