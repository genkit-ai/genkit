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
import { Operation } from 'genkit';
import { GenerateRequest } from 'genkit/model';
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { getGenkitClientHeader } from '../../src/common/utils.js';
import { getVertexAIUrl } from '../../src/vertexai/client.js';
import {
  fromVeoOperation,
  toVeoOperationRequest,
  toVeoPredictRequest,
} from '../../src/vertexai/converters.js';
import {
  ClientOptions,
  RegionalClientOptions,
  VeoOperation,
  VeoOperationRequest,
  VeoPredictRequest,
} from '../../src/vertexai/types.js';
import {
  TEST_ONLY,
  VeoConfigSchema,
  defineModel,
  model,
} from '../../src/vertexai/veo.js';

const { GENERIC_MODEL, KNOWN_MODELS } = TEST_ONLY;

describe('Vertex AI Veo', () => {
  let fetchStub: sinon.SinonStub;
  let authMock: sinon.SinonStubbedInstance<GoogleAuth>;

  const modelName = 'veo-test-model';

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
    it('should return a ModelReference for a known model', () => {
      const knownModelName = Object.keys(KNOWN_MODELS)[0];
      const ref = model(knownModelName);
      assert.strictEqual(ref.name, `vertexai/${knownModelName}`);
      assert.ok(ref.info?.supports?.media);
    });

    it('should return a ModelReference for an unknown model using generic info', () => {
      const unknownModelName = 'veo-unknown-model';
      const ref = model(unknownModelName);
      assert.strictEqual(ref.name, `vertexai/${unknownModelName}`);
      assert.deepStrictEqual(ref.info, GENERIC_MODEL.info);
    });
  });

  describe('defineModel()', () => {
    function captureModelRunner(clientOptions: ClientOptions): {
      start: (
        request: GenerateRequest<typeof VeoConfigSchema>
      ) => Promise<Operation>;
      check: (operation: Operation) => Promise<Operation>;
    } {
      const model = defineModel(modelName, clientOptions);
      return {
        start: (req) => model.start(req),
        check: (op) => model.check(op),
      };
    }

    describe('start()', () => {
      const prompt = 'A unicycle on the moon';
      const request: GenerateRequest<typeof VeoConfigSchema> = {
        messages: [{ role: 'user', content: [{ text: prompt }] }],
        config: {
          aspectRatio: '16:9',
          durationSeconds: 5,
        },
      };

      it('should call fetch for veoPredict and return operation', async () => {
        const mockOp: VeoOperation = {
          name: `projects/test-project/locations/us-central1/publishers/google/models/${modelName}/operations/start123`,
          done: false,
        };
        mockFetchResponse(mockOp);

        const { start } = captureModelRunner(defaultRegionalClientOptions);
        const result = await start(request);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        const options = fetchArgs[1];

        const expectedUrl = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: `publishers/google/models/${modelName}`,
          resourceMethod: 'predictLongRunning',
          clientOptions: defaultRegionalClientOptions,
        });
        assert.strictEqual(url, expectedUrl);
        assert.strictEqual(options.method, 'POST');
        assert.deepStrictEqual(options.headers, getExpectedHeaders());

        const expectedPredictRequest: VeoPredictRequest =
          toVeoPredictRequest(request);
        assert.deepStrictEqual(
          JSON.parse(options.body),
          expectedPredictRequest
        );

        const expectedOp = fromVeoOperation(mockOp);
        assert.strictEqual(result.id, expectedOp.id);
        assert.strictEqual(result.done, expectedOp.done);
      });

      it('should propagate API errors', async () => {
        const errorBody = { error: { message: 'Invalid arg', code: 400 } };
        mockFetchResponse(errorBody, 400);

        const { start } = captureModelRunner(defaultRegionalClientOptions);
        await assert.rejects(
          start(request),
          /Error fetching from .*predictLongRunning.* Invalid arg/
        );
      });

      it('should pass AbortSignal to fetch', async () => {
        mockFetchResponse({ name: 'operations/abort', done: false });
        const controller = new AbortController();
        const abortSignal = controller.signal;

        const clientOptionsWithSignal = {
          ...defaultRegionalClientOptions,
          signal: abortSignal,
        };
        const { start } = captureModelRunner(clientOptionsWithSignal);
        await start(request);

        sinon.assert.calledOnce(fetchStub);
        const fetchOptions = fetchStub.lastCall.args[1];
        assert.ok(fetchOptions.signal, 'Fetch options should have a signal');

        const fetchSignal = fetchOptions.signal;
        const abortSpy = sinon.spy();
        fetchSignal.addEventListener('abort', abortSpy);

        // verify that aborting the original signal aborts the fetch signal
        controller.abort();
        sinon.assert.calledOnce(abortSpy);
      });
    });

    describe('check()', () => {
      const operationId = `projects/test-project/locations/us-central1/publishers/google/models/${modelName}/operations/check123`;
      const pendingOp: Operation = { id: operationId, done: false };

      it('should call fetch for veoCheckOperation and return updated operation', async () => {
        const mockResponse: VeoOperation = {
          name: operationId,
          done: true,
          response: {
            videos: [
              {
                gcsUri: 'gs://test-bucket/video.mp4',
                mimeType: 'video/mp4',
              },
            ],
          },
        };
        mockFetchResponse(mockResponse);

        const { check } = captureModelRunner(defaultRegionalClientOptions);
        const result = await check(pendingOp);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        const options = fetchArgs[1];

        const expectedUrl = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: `publishers/google/models/${modelName}`,
          resourceMethod: 'fetchPredictOperation',
          clientOptions: defaultRegionalClientOptions,
        });
        assert.strictEqual(url, expectedUrl);
        assert.strictEqual(options.method, 'POST');
        assert.deepStrictEqual(options.headers, getExpectedHeaders());

        const expectedCheckRequest: VeoOperationRequest =
          toVeoOperationRequest(pendingOp);
        assert.deepStrictEqual(JSON.parse(options.body), expectedCheckRequest);

        const expectedOp = fromVeoOperation(mockResponse);
        assert.strictEqual(result.id, expectedOp.id);
        assert.strictEqual(result.done, expectedOp.done);
        assert.deepStrictEqual(result.output, expectedOp.output);
      });

      it('should propagate API errors for check', async () => {
        const errorBody = { error: { message: 'Not found', code: 404 } };
        mockFetchResponse(errorBody, 404);

        const { check } = captureModelRunner(defaultRegionalClientOptions);
        await assert.rejects(
          check(pendingOp),
          /Error fetching from .*fetchPredictOperation.* Not found/
        );
      });

      it('should use clientOptions from operation metadata if available', async () => {
        const opClientOptions: ClientOptions = {
          kind: 'regional',
          projectId: 'op-project',
          location: 'europe-west1',
          authClient: authMock as any,
        };
        const opWithClientOptions: Operation = {
          ...pendingOp,
          metadata: { clientOptions: opClientOptions },
        };
        mockFetchResponse({ name: operationId, done: true });

        const { check } = captureModelRunner(defaultRegionalClientOptions);
        await check(opWithClientOptions);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        assert.ok(url.includes('europe-west1'));
        assert.ok(url.includes('op-project'));
      });
    });
  });
});
