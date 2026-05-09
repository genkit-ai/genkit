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
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { TextEncoder } from 'util';
import { getGenkitClientHeader } from '../../src/common/utils.js';
import {
  embedContent,
  generateContent,
  generateContentStream,
  getVertexAIUrl,
  imagenPredict,
  listModels,
  lyriaPredict,
  veoCheckOperation,
  veoPredict,
} from '../../src/vertexai/client.js';
import {
  ClientOptions,
  Content,
  EmbedContentRequest,
  EmbedContentResponse,
  GenerateContentRequest,
  GenerateContentResponse,
  GenerateContentStreamResult,
  ImagenPredictRequest,
  ImagenPredictResponse,
  LyriaPredictRequest,
  LyriaPredictResponse,
  Model,
  VeoOperation,
  VeoOperationRequest,
  VeoPredictRequest,
} from '../../src/vertexai/types.js';
import { NOT_SUPPORTED_IN_EXPRESS_ERROR } from '../../src/vertexai/utils.js';

describe('Vertex AI Client', () => {
  let fetchSpy: sinon.SinonStub;
  let authMock: sinon.SinonStubbedInstance<GoogleAuth>;

  const regionalClientOptions: ClientOptions = {
    kind: 'regional',
    projectId: 'test-project',
    location: 'us-central1',
    authClient: {} as GoogleAuth, // Will be replaced by mock
  };

  const globalClientOptions: ClientOptions = {
    kind: 'global',
    projectId: 'test-project',
    location: 'global',
    authClient: {} as GoogleAuth, // Will be replaced by mock
  };

  const expressClientOptions: ClientOptions = {
    kind: 'express',
    apiKey: 'test-api-key',
  };

  const notSupportedInExpressErrorMessage = {
    message: NOT_SUPPORTED_IN_EXPRESS_ERROR.message,
  };

  beforeEach(() => {
    fetchSpy = sinon.stub(global, 'fetch');
    authMock = sinon.createStubInstance(GoogleAuth);
    authMock.getAccessToken.resolves('test-token');
    (regionalClientOptions as any).authClient =
      authMock as unknown as GoogleAuth;
    (globalClientOptions as any).authClient = authMock as unknown as GoogleAuth;
  });

  afterEach(() => {
    sinon.restore();
  });

  function mockFetchResponse(
    body: any,
    ok = true,
    status = 200,
    statusText = 'OK',
    contentType = 'application/json'
  ) {
    const bodyString =
      body === null || body === undefined
        ? ''
        : contentType === 'application/json'
          ? JSON.stringify(body)
          : String(body);
    const response = new Response(bodyString, {
      status: status,
      statusText: statusText,
      headers: { 'Content-Type': contentType },
    });
    fetchSpy.resolves(response);
  }

  describe('getVertexAIUrl', () => {
    describe('Regional', () => {
      const opts: ClientOptions = {
        kind: 'regional',
        projectId: 'test-proj',
        location: 'us-east1',
        authClient: {} as any,
      };

      it('should build URL for listModels', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: false,
          resourcePath: 'publishers/google/models',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/publishers/google/models'
        );
      });

      it('should build URL for generateContent', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'publishers/google/models/gemini-2.0-pro',
          resourceMethod: 'generateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/us-east1/publishers/google/models/gemini-2.0-pro:generateContent'
        );
      });

      it('should build URL for generateContent with tuned model without project', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'endpoints/12345678',
          resourceMethod: 'generateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/us-east1/endpoints/12345678:generateContent'
        );
      });

      it('should build URL for generateContent with tuned model with project', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: false,
          resourcePath:
            'projects/project1/locations/location1/endpoints/12345678',
          resourceMethod: 'generateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/project1/locations/location1/endpoints/12345678:generateContent'
        );
      });

      it('should build URL for streamGenerateContent', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'publishers/google/models/gemini-2.5-flash',
          resourceMethod: 'streamGenerateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/us-east1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse'
        );
      });

      it('should build URL for streamGenerateContent with tuned model without project', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'endpoints/12345678',
          resourceMethod: 'streamGenerateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/us-east1/endpoints/12345678:streamGenerateContent?alt=sse'
        );
      });

      it('should build URL for streamGenerateContent with tuned model with project', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: false,
          resourcePath:
            'projects/project1/locations/location1/endpoints/12345678',
          resourceMethod: 'streamGenerateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/project1/locations/location1/endpoints/12345678:streamGenerateContent?alt=sse'
        );
      });

      it('should build URL for predict', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'publishers/google/models/imagen-3.0',
          resourceMethod: 'predict',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/us-east1/publishers/google/models/imagen-3.0:predict'
        );
      });

      it('should build URL for predictLongRunning', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'publishers/google/models/veo-2.0',
          resourceMethod: 'predictLongRunning',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/us-east1/publishers/google/models/veo-2.0:predictLongRunning'
        );
      });

      it('should build URL for fetchPredictOperation', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'publishers/google/models/veo-2.0',
          resourceMethod: 'fetchPredictOperation',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/us-east1/publishers/google/models/veo-2.0:fetchPredictOperation'
        );
      });

      it('should handle queryParams', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: false,
          resourcePath: 'publishers/google/models',
          clientOptions: opts,
          queryParams: 'pageSize=10',
        });
        assert.strictEqual(
          url,
          'https://us-east1-aiplatform.googleapis.com/v1beta1/publishers/google/models?pageSize=10'
        );
      });
    });

    describe('Global', () => {
      const opts: ClientOptions = {
        kind: 'global',
        projectId: 'test-proj',
        location: 'global',
        authClient: {} as any,
      };

      it('should build URL for listModels', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: false,
          resourcePath: 'publishers/google/models',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://aiplatform.googleapis.com/v1beta1/publishers/google/models'
        );
      });

      it('should build URL for generateContent', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'publishers/google/models/gemini-2.0-pro',
          resourceMethod: 'generateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://aiplatform.googleapis.com/v1beta1/projects/test-proj/locations/global/publishers/google/models/gemini-2.0-pro:generateContent'
        );
      });
    });

    describe('Express', () => {
      const opts: ClientOptions = {
        kind: 'express',
        apiKey: 'test-api-key',
      };

      it('should not support listModels', () => {
        assert.throws(() => {
          return getVertexAIUrl({
            includeProjectAndLocation: false,
            resourcePath: 'publishers/google/models',
            clientOptions: opts,
          });
        }, notSupportedInExpressErrorMessage);
      });

      it('should not support predict', () => {
        assert.throws(() => {
          return getVertexAIUrl({
            includeProjectAndLocation: true,
            resourcePath: 'publishers/google/models/imagen-3.0',
            resourceMethod: 'predict',
            clientOptions: opts,
          });
        }, notSupportedInExpressErrorMessage);
      });

      it('should not support predictLongRunning', () => {
        assert.throws(() => {
          return getVertexAIUrl({
            includeProjectAndLocation: true,
            resourcePath: 'publishers/google/models/veo-2.0',
            resourceMethod: 'predictLongRunning',
            clientOptions: opts,
          });
        }, notSupportedInExpressErrorMessage);
      });

      it('should build URL for generateContent', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true, // This is ignored for express
          resourcePath: 'publishers/google/models/gemini-2.0-pro',
          resourceMethod: 'generateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-2.0-pro:generateContent'
        );
      });

      it('should build URL for streamGenerateContent', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: true, // Ignored
          resourcePath: 'publishers/google/models/gemini-2.5-flash',
          resourceMethod: 'streamGenerateContent',
          clientOptions: opts,
        });
        assert.strictEqual(
          url,
          'https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse'
        );
      });

      it('should not support tuned models for generateContent', () => {
        assert.throws(() => {
          return getVertexAIUrl({
            includeProjectAndLocation: true,
            resourcePath: 'endpoints/12345678',
            resourceMethod: 'generateContent',
            clientOptions: opts,
          });
        }, notSupportedInExpressErrorMessage);
      });

      it('should handle queryParams', () => {
        const url = getVertexAIUrl({
          includeProjectAndLocation: false,
          resourcePath: 'publishers/google/models/gemini-2.5-flash',
          resourceMethod: 'generateContent',
          clientOptions: opts,
          queryParams: 'pageSize=10',
        });
        assert.strictEqual(
          url,
          'https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-2.5-flash:generateContent?pageSize=10'
        );
      });
    });
  });

  describe('Authentication Error', () => {
    it('should throw a specific error if getToken fails', async () => {
      authMock.getAccessToken.rejects(new Error('Auth failed'));
      await assert.rejects(
        listModels(regionalClientOptions),
        /Unable to authenticate your request/
      );
    });
  });

  describe('API Calls', () => {
    const testCases = [
      { name: 'Regional', options: regionalClientOptions },
      { name: 'Global', options: globalClientOptions },
      { name: 'Express', options: expressClientOptions },
    ];

    for (const testCase of testCases) {
      describe(`${testCase.name} Client - kind: ${testCase.options.kind}`, () => {
        const currentOptions = testCase.options;
        const isExpress = currentOptions.kind === 'express';
        const location =
          currentOptions.kind === 'regional'
            ? currentOptions.location
            : 'global';
        const projectId =
          currentOptions.kind !== 'express' ? currentOptions.projectId : '';

        const getExpectedHeaders = () => {
          const headers: Record<string, string | undefined> = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Client': getGenkitClientHeader(),
            'User-Agent': getGenkitClientHeader(),
          };
          if (isExpress) {
            return {
              ...headers,
              'x-goog-api-key': currentOptions.apiKey,
            };
          }
          return {
            ...headers,
            Authorization: 'Bearer test-token',
            'x-goog-user-project': projectId,
          };
        };

        const getBaseUrl = (path: string) => {
          if (isExpress) {
            return `https://aiplatform.googleapis.com/v1beta1/${path}`;
          }
          const domain =
            currentOptions.kind === 'regional'
              ? `${location}-aiplatform.googleapis.com`
              : 'aiplatform.googleapis.com';
          return `https://${domain}/v1beta1/${path}`;
        };

        const getResourceUrl = (model: string, method: string) => {
          const isStreaming = method.includes('streamGenerateContent');
          const isTuned = model.includes('endpoints/');
          let url;

          if (isExpress) {
            let resourcePath;
            if (isTuned) {
              resourcePath = model;
            } else {
              resourcePath = `publishers/google/models/${model}`;
            }
            url = `https://aiplatform.googleapis.com/v1beta1/${resourcePath}:${method}`;
          } else {
            const domain =
              currentOptions.kind === 'regional'
                ? `${location}-aiplatform.googleapis.com`
                : 'aiplatform.googleapis.com';

            let resourcePath;
            if (isTuned) {
              if (model.startsWith('projects/')) {
                resourcePath = model;
              } else {
                resourcePath = `projects/${projectId}/locations/${location}/${model}`;
              }
            } else {
              resourcePath = `projects/${projectId}/locations/${location}/publishers/google/models/${model}`;
            }
            url = `https://${domain}/v1beta1/${resourcePath}:${method}`;
          }

          if (isStreaming) {
            url += `?alt=sse`;
          }
          return url;
        };

        describe('listModels', () => {
          if (!isExpress) {
            it('should return a list of models', async () => {
              const mockModels: Model[] = [
                { name: 'gemini-2.0-pro', launchStage: 'GA' },
              ];
              mockFetchResponse({ publisherModels: mockModels });

              const result = await listModels(currentOptions);
              assert.deepStrictEqual(result, mockModels);

              const expectedUrl = getBaseUrl('publishers/google/models');
              sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
                method: 'GET',
                headers: getExpectedHeaders(),
              });

              if (!isExpress) {
                sinon.assert.calledOnce(authMock.getAccessToken);
              } else {
                sinon.assert.notCalled(authMock.getAccessToken);
              }
            });
          } else {
            it('should throw with unsupported for Express', async () => {
              await assert.rejects(
                listModels(currentOptions),
                notSupportedInExpressErrorMessage
              );
            });
          }
        });

        describe('generateContent', () => {
          const request: GenerateContentRequest = {
            contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
          };
          const model = 'gemini-2.0-pro';
          const tunedModel = 'endpoints/123456789';

          it('should return GenerateContentResponse for published model', async () => {
            const mockResponse: GenerateContentResponse = { candidates: [] };
            mockFetchResponse(mockResponse);

            const result = await generateContent(
              model,
              request,
              currentOptions
            );
            assert.deepStrictEqual(result, mockResponse);

            const expectedUrl = getResourceUrl(model, 'generateContent');
            sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
              method: 'POST',
              headers: getExpectedHeaders(),
              body: JSON.stringify(request),
            });
          });

          it('should return GenerateContentResponse for tuned model', async () => {
            if (isExpress) {
              await assert.rejects(
                generateContent(tunedModel, request, currentOptions),
                notSupportedInExpressErrorMessage
              );
            } else {
              const mockResponse: GenerateContentResponse = { candidates: [] };
              mockFetchResponse(mockResponse);

              const result = await generateContent(
                tunedModel,
                request,
                currentOptions
              );
              assert.deepStrictEqual(result, mockResponse);

              const expectedUrl = getResourceUrl(tunedModel, 'generateContent');
              sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
                method: 'POST',
                headers: getExpectedHeaders(),
                body: JSON.stringify(request),
              });
            }
          });

          it('should throw on API error', async () => {
            const errorResponse = { error: { message: 'Permission denied' } };
            mockFetchResponse(errorResponse, false, 403, 'Forbidden');

            await assert.rejects(
              generateContent(model, request, currentOptions),
              (err: any) => {
                assert.strictEqual(err.name, 'GenkitError');
                assert.strictEqual(err.status, 'UNKNOWN');
                assert.match(
                  err.message,
                  /Error fetching from .* \[403 Forbidden\] Permission denied/
                );
                return true;
              }
            );
          });
        });

        describe('embedContent', () => {
          const request: EmbedContentRequest = {
            instances: [{ content: 'test content' }],
            parameters: {},
          };
          const model = 'text-embedding-005';

          if (!isExpress) {
            it('should return EmbedContentResponse', async () => {
              const mockResponse: EmbedContentResponse = { predictions: [] };
              mockFetchResponse(mockResponse);

              await embedContent(model, request, currentOptions);
              const expectedUrl = getResourceUrl(model, 'predict');
              sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
                method: 'POST',
                headers: getExpectedHeaders(),
                body: JSON.stringify(request),
              });
            });
          } else {
            it('should throw with unsupported for Express', async () => {
              await assert.rejects(
                embedContent(model, request, currentOptions),
                notSupportedInExpressErrorMessage
              );
            });
          }
        });

        describe('imagenPredict', () => {
          const request: ImagenPredictRequest = {
            instances: [{ prompt: 'a cat' }],
            parameters: { sampleCount: 1 },
          };
          const model = 'imagen-3.0-generate-002';
          if (!isExpress) {
            it('should return ImagenPredictResponse', async () => {
              const mockResponse: ImagenPredictResponse = { predictions: [] };
              mockFetchResponse(mockResponse);
              await imagenPredict(model, request, currentOptions);

              const expectedUrl = getResourceUrl(model, 'predict');
              sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
                method: 'POST',
                headers: getExpectedHeaders(),
                body: JSON.stringify(request),
              });
            });
          } else {
            it('should throw with unsupported for Express', async () => {
              await assert.rejects(
                imagenPredict(model, request, currentOptions),
                notSupportedInExpressErrorMessage
              );
            });
          }
        });

        describe('lyriaPredict', () => {
          const request: LyriaPredictRequest = {
            instances: [{ prompt: 'a song' }],
            parameters: { sampleCount: 1 },
          };
          const model = 'lyria-002';
          if (!isExpress) {
            it('should return LyriaPredictResponse', async () => {
              const mockResponse: LyriaPredictResponse = { predictions: [] };
              mockFetchResponse(mockResponse);
              await lyriaPredict(model, request, currentOptions);

              const expectedUrl = getResourceUrl(model, 'predict');
              sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
                method: 'POST',
                headers: getExpectedHeaders(),
                body: JSON.stringify(request),
              });
            });
          } else {
            it('should throw with unsupported for Express', async () => {
              await assert.rejects(
                lyriaPredict(model, request, currentOptions),
                notSupportedInExpressErrorMessage
              );
            });
          }
        });

        describe('veoPredict', () => {
          const request: VeoPredictRequest = {
            instances: [{ prompt: 'a video' }],
            parameters: {},
          };
          const model = 'veo-2.0-generate-001';
          if (!isExpress) {
            it('should return VeoOperation', async () => {
              const mockResponse: VeoOperation = { name: 'operations/123' };
              mockFetchResponse(mockResponse);
              const result = await veoPredict(model, request, currentOptions);
              assert.deepStrictEqual(result, {
                ...mockResponse,
                clientOptions: currentOptions,
              });

              const expectedUrl = getResourceUrl(model, 'predictLongRunning');
              sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
                method: 'POST',
                headers: getExpectedHeaders(),
                body: JSON.stringify(request),
              });
            });
          } else {
            it('should throw with unsupported for Express', async () => {
              await assert.rejects(
                veoPredict(model, request, currentOptions),
                notSupportedInExpressErrorMessage
              );
            });
          }
        });

        describe('veoCheckOperation', () => {
          const request: VeoOperationRequest = {
            operationName: 'operations/123',
          };
          const model = 'veo-2.0-generate-001';
          if (!isExpress) {
            it('should return VeoOperation', async () => {
              const mockResponse: VeoOperation = {
                name: 'operations/123',
                done: true,
              };
              mockFetchResponse(mockResponse);
              const result = await veoCheckOperation(
                model,
                request,
                currentOptions
              );
              assert.deepStrictEqual(result, {
                ...mockResponse,
                clientOptions: currentOptions,
              });

              const expectedUrl = getResourceUrl(
                model,
                'fetchPredictOperation'
              );
              sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
                method: 'POST',
                headers: getExpectedHeaders(),
                body: JSON.stringify(request),
              });
            });
          } else {
            it('should throw with unsupported for Express', async () => {
              await assert.rejects(
                veoCheckOperation(model, request, currentOptions),
                notSupportedInExpressErrorMessage
              );
            });
          }
        });

        describe('generateContentStream', () => {
          const model = 'gemini-2.5-flash';
          const tunedModel = 'endpoints/123456789';

          const request: GenerateContentRequest = {
            contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
          };

          function mockStream() {
            const chunks = [
              'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "Hello "}]}}]}\n\n',
            ];
            const stream = new ReadableStream({
              start(controller) {
                for (const chunk of chunks) {
                  controller.enqueue(new TextEncoder().encode(chunk));
                }
                controller.close();
              },
            });
            fetchSpy.resolves(
              new Response(stream, {
                headers: { 'Content-Type': 'application/json' },
              })
            );
          }

          it('should process stream for published model', async () => {
            mockStream();

            await generateContentStream(model, request, currentOptions);

            const expectedUrl = getResourceUrl(model, 'streamGenerateContent');
            sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
              method: 'POST',
              headers: getExpectedHeaders(),
              body: JSON.stringify(request),
            });
          });

          it('should process stream for tuned model', async () => {
            if (isExpress) {
              await assert.rejects(
                generateContentStream(tunedModel, request, currentOptions),
                notSupportedInExpressErrorMessage
              );
            } else {
              mockStream();

              await generateContentStream(tunedModel, request, currentOptions);

              const expectedUrl = getResourceUrl(
                tunedModel,
                'streamGenerateContent'
              );
              sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
                method: 'POST',
                headers: getExpectedHeaders(),
                body: JSON.stringify(request),
              });
            }
          });
        });
      });
    }
  });

  describe('Error Handling Extras', () => {
    it('listModels should throw an error if fetch fails with non-JSON error', async () => {
      mockFetchResponse(
        '<h1>Gateway Timeout</h1>',
        false,
        504,
        'Gateway Timeout',
        'text/html'
      );

      await assert.rejects(listModels(regionalClientOptions), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'UNKNOWN');
        assert.match(
          err.message,
          /Error fetching from .* \[504 Gateway Timeout\] <h1>Gateway Timeout<\/h1>/
        );
        return true;
      });
    });

    it('listModels should throw an error if fetch fails with empty response body', async () => {
      mockFetchResponse(null, false, 502, 'Bad Gateway');

      await assert.rejects(listModels(regionalClientOptions), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'UNKNOWN');
        assert.match(err.message, /Error fetching from .* \[502 Bad Gateway\]/);
        return true;
      });
    });

    it('listModels should throw an error on network failure', async () => {
      fetchSpy.rejects(new Error('Network Error'));
      await assert.rejects(
        listModels(regionalClientOptions),
        /Failed to fetch from .* Network Error/
      );
    });

    it('should throw a resource exhausted error on 429', async () => {
      const errorResponse = { error: { message: 'Too many requests' } };
      mockFetchResponse(errorResponse, false, 429, 'Too Many Requests');

      await assert.rejects(listModels(regionalClientOptions), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'RESOURCE_EXHAUSTED');
        assert.match(
          err.message,
          /Error fetching from .* \[429 Too Many Requests\] Too many requests/
        );
        return true;
      });
    });
  });

  describe('generateContentStream full aggregation tests', () => {
    function createMockStream(chunks: string[]): Response {
      const stream = new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(new TextEncoder().encode(chunk));
          }
          controller.close();
        },
      });
      return new Response(stream, {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    it('should process stream and return stream and aggregated response', async () => {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
      };
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "Hello "}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "World!"}]}}], "usageMetadata": {"totalTokenCount": 10}}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));

      const result: GenerateContentStreamResult = await generateContentStream(
        'gemini-2.5-flash',
        request,
        regionalClientOptions
      );

      const streamResults: GenerateContentResponse[] = [];
      for await (const item of result.stream) {
        streamResults.push(item);
      }

      assert.deepStrictEqual(streamResults, [
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [{ text: 'Hello ' }],
              } as Content,
            } as any,
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [{ text: 'World!' }],
              } as Content,
            } as any,
          ],
          usageMetadata: { totalTokenCount: 10 },
        },
      ]);

      const aggregated = await result.response;
      assert.deepStrictEqual(aggregated, {
        candidates: [
          {
            index: 0,
            content: {
              role: 'model',
              parts: [{ text: 'Hello ' }, { text: 'World!' }],
            },
          },
        ],
        usageMetadata: { totalTokenCount: 10 },
      });
    });

    it('should handle stream with malformed JSON', async () => {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
      };
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "Hello "}]}}]}\n\n',
        'data: {"candi dates": []}}}}\n\n', // Malformed
      ];
      fetchSpy.resolves(createMockStream(chunks));

      const result: GenerateContentStreamResult = await generateContentStream(
        'gemini-2.5-flash',
        request,
        regionalClientOptions
      );

      try {
        for await (const _ of result.stream) {
          // Consume stream
        }
        assert.fail('Stream should have thrown an error');
      } catch (e: any) {
        assert.match(e.message, /Error parsing JSON response/);
      }

      try {
        await result.response;
        assert.fail('Response promise should have thrown an error');
      } catch (e: any) {
        assert.match(e.message, /Error parsing JSON response/);
      }
    });

    it('should aggregate parts for multiple candidates', async () => {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
      };
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "C0 A"}]}}, {"index": 1, "content": {"role": "model", "parts": [{"text": "C1 A"}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": " C0 B"}]}}, {"index": 1, "content": {"role": "model", "parts": [{"text": " C1 B"}]}}]}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(
        'gemini-2.5-flash',
        request,
        regionalClientOptions
      );
      const aggregated = await result.response;

      assert.strictEqual(aggregated.candidates?.length, 2);
      const sortedCandidates = aggregated.candidates!.sort(
        (a, b) => a.index - b.index
      );

      assert.deepStrictEqual(sortedCandidates[0].content.parts, [
        { text: 'C0 A' },
        { text: ' C0 B' },
      ]);
      assert.deepStrictEqual(sortedCandidates[1].content.parts, [
        { text: 'C1 A' },
        { text: ' C1 B' },
      ]);
    });

    it('should aggregate functionCall parts, keeping text first', async () => {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
      };
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "A"}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"functionCall": {"name": "tool1", "args": {}}}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "B"}]}}]}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(
        'gemini-2.5-flash',
        request,
        regionalClientOptions
      );
      const aggregated = await result.response;
      assert.deepStrictEqual(aggregated.candidates![0].content.parts, [
        { text: 'A' },
        { functionCall: { name: 'tool1', args: {} } },
        { text: 'B' },
      ]);
    });

    it('should remove empty first text part if other parts exist', async () => {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
      };
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"functionCall": {"name": "tool1", "args": {}}}]}}]}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(
        'gemini-2.5-flash',
        request,
        regionalClientOptions
      );
      const aggregated = await result.response;
      assert.deepStrictEqual(aggregated.candidates![0].content.parts, [
        { functionCall: { name: 'tool1', args: {} } },
      ]);
    });

    it('should aggregate citation metadata', async () => {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
      };
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "."}] }, "citationMetadata": { "citations": [{"uri": "u1"}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "."}] }, "citationMetadata": { "citations": [{"uri": "u2"}]}}]}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(
        'gemini-2.5-flash',
        request,
        regionalClientOptions
      );
      const aggregated = await result.response;
      assert.deepStrictEqual(aggregated.candidates![0].citationMetadata, {
        citations: [{ uri: 'u1' }, { uri: 'u2' }],
      });
    });

    it('should aggregate grounding metadata', async () => {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
      };
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "."}] }, "groundingMetadata": { "webSearchQueries": ["q1"]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "."}] }, "groundingMetadata": { "webSearchQueries": ["q2"], "retrievalQueries": ["rq1"], "searchEntryPoint": { "renderedContent": "test" } }}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "."}] }, "groundingMetadata": { "groundingChunks": [{"web": {"uri": "u1"}}], "groundingSupports": [{"segment": {"text": "s1"}}]}}]}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(
        'gemini-2.5-flash',
        request,
        regionalClientOptions
      );
      const aggregated = await result.response;
      assert.deepStrictEqual(aggregated.candidates![0].groundingMetadata, {
        webSearchQueries: ['q1', 'q2'],
        retrievalQueries: ['rq1'],
        groundingChunks: [{ web: { uri: 'u1' } }],
        groundingSupports: [{ segment: { text: 's1' } }],
        searchEntryPoint: { renderedContent: 'test' },
      });
    });

    it('should take last finishReason, finishMessage, and safetyRatings', async () => {
      const request: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream' }] }],
      };
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "A"}] }, "finishReason": "STOP" }]}\n\n',
        'data: {"candidates": [{"index": 0, "content": { "role": "model", "parts": [{"text": "B"}] }, "finishReason": "MAX_TOKENS", "finishMessage": "Done", "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "LOW"}] }]}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(
        'gemini-2.5-flash',
        request,
        regionalClientOptions
      );
      const aggregated = await result.response;
      assert.strictEqual(aggregated.candidates![0].finishReason, 'MAX_TOKENS');
      assert.strictEqual(aggregated.candidates![0].finishMessage, 'Done');
      assert.deepStrictEqual(aggregated.candidates![0].safetyRatings, [
        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', probability: 'LOW' },
      ]);
    });
  });
});
