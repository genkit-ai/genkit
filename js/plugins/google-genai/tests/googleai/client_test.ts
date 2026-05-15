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
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { TextEncoder } from 'util';
import {
  FinishReason,
  HarmCategory,
  HarmProbability,
} from '../../src/common/types.js';
import { getGenkitClientHeader } from '../../src/common/utils.js';
import {
  TEST_ONLY,
  embedContent,
  generateContent,
  generateContentStream,
  getGoogleAIUrl,
  imagenPredict,
  listModels,
  veoCheckOperation,
  veoPredict,
} from '../../src/googleai/client.js';
import {
  ClientOptions,
  EmbedContentRequest,
  EmbedContentResponse,
  GenerateContentRequest,
  GenerateContentResponse,
  GenerateContentStreamResult,
  ImagenPredictRequest,
  ImagenPredictResponse,
  Model,
  VeoOperation,
  VeoPredictRequest,
} from '../../src/googleai/types.js';

const { getAbortSignal } = TEST_ONLY;

describe('Google AI Client', () => {
  let fetchSpy: sinon.SinonStub;
  const apiKey = 'test-api-key';
  const defaultBaseUrl = 'https://generativelanguage.googleapis.com';
  const defaultApiVersion = 'v1beta';

  beforeEach(() => {
    fetchSpy = sinon.stub(global, 'fetch');
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

  describe('getGoogleAIUrl', () => {
    it('should build basic URL', () => {
      const url = getGoogleAIUrl({ resourcePath: 'models' });
      assert.strictEqual(url, `${defaultBaseUrl}/${defaultApiVersion}/models`);
    });

    it('should build URL with resourceMethod', () => {
      const url = getGoogleAIUrl({
        resourcePath: 'models/gemini-2.0-pro',
        resourceMethod: 'generateContent',
      });
      assert.strictEqual(
        url,
        `${defaultBaseUrl}/${defaultApiVersion}/models/gemini-2.0-pro:generateContent`
      );
    });

    it('should build URL with queryParams', () => {
      const url = getGoogleAIUrl({
        resourcePath: 'models',
        queryParams: 'pageSize=10',
      });
      assert.strictEqual(
        url,
        `${defaultBaseUrl}/${defaultApiVersion}/models?pageSize=10`
      );
    });

    it('should add alt=sse for streamGenerateContent', () => {
      const url = getGoogleAIUrl({
        resourcePath: 'models/gemini-2.5-flash',
        resourceMethod: 'streamGenerateContent',
      });
      assert.strictEqual(
        url,
        `${defaultBaseUrl}/${defaultApiVersion}/models/gemini-2.5-flash:streamGenerateContent?alt=sse`
      );
    });

    it('should add alt=sse for streamGenerateContent with other queryParams', () => {
      const url = getGoogleAIUrl({
        resourcePath: 'models/gemini-2.5-flash',
        resourceMethod: 'streamGenerateContent',
        queryParams: 'test=123',
      });
      assert.strictEqual(
        url,
        `${defaultBaseUrl}/${defaultApiVersion}/models/gemini-2.5-flash:streamGenerateContent?test=123&alt=sse`
      );
    });

    it('should use custom apiVersion and baseUrl from RequestOptions', () => {
      const clientOptions: ClientOptions = {
        apiVersion: 'v1',
        baseUrl: 'https://custom.googleapis.com',
      };
      const url = getGoogleAIUrl({ resourcePath: 'models', clientOptions });
      assert.strictEqual(url, 'https://custom.googleapis.com/v1/models');
    });
  });

  describe('listModels', () => {
    it('should return a list of models', async () => {
      const mockModels: Model[] = [
        { name: 'models/gemini-2.0-pro' } as Model,
        { name: 'models/gemini-2.5-flash' } as Model,
      ];
      mockFetchResponse({ models: mockModels });

      const result = await listModels(apiKey);
      assert.deepStrictEqual(result, mockModels);

      const expectedUrl =
        'https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000';
      sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        },
      });
    });

    it('should throw an error if fetch fails with JSON error', async () => {
      const errorResponse = { error: { message: 'Internal Error' } };
      mockFetchResponse(errorResponse, false, 500, 'Internal Server Error');

      await assert.rejects(listModels(apiKey), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'INTERNAL');
        assert.match(
          err.message,
          /Error fetching from .* \[500 Internal Server Error\] Internal Error/
        );
        return true;
      });
    });

    it('should throw an error if fetch fails with non-JSON error', async () => {
      mockFetchResponse(
        '<html><body><h1>Server Error</h1></body></html>',
        false,
        500,
        'Internal Server Error',
        'text/html'
      );

      await assert.rejects(listModels(apiKey), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'INTERNAL');
        assert.match(
          err.message,
          /Error fetching from .* \[500 Internal Server Error\] <html><body><h1>Server Error<\/h1><\/body><\/html>/
        );
        return true;
      });
    });

    it('should throw an error if fetch fails with empty response body', async () => {
      mockFetchResponse(null, false, 502, 'Bad Gateway');

      await assert.rejects(listModels(apiKey), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'UNKNOWN');
        assert.match(err.message, /Error fetching from .* \[502 Bad Gateway\]/);
        return true;
      });
    });

    it('should throw a resource exhausted error on 429', async () => {
      const errorResponse = { error: { message: 'Too many requests' } };
      mockFetchResponse(errorResponse, false, 429, 'Too Many Requests');

      await assert.rejects(listModels(apiKey), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'RESOURCE_EXHAUSTED');
        assert.match(
          err.message,
          /Error fetching from .* \[429 Too Many Requests\] Too many requests/
        );
        return true;
      });
    });

    it('should throw an error on network failure', async () => {
      fetchSpy.rejects(new Error('Network connection failed'));

      await assert.rejects(
        listModels(apiKey),
        /Failed to fetch from .* Network connection failed/
      );
    });

    it('should include custom headers', async () => {
      mockFetchResponse({ models: [] });
      const clientOptions: ClientOptions = {
        customHeaders: { 'X-Custom-Header': 'test' },
      };
      await listModels(apiKey, clientOptions);

      sinon.assert.calledOnce(fetchSpy);
      const headers = fetchSpy.firstCall.args[1].headers;
      assert.strictEqual(headers['X-Custom-Header'], 'test');
      assert.strictEqual(headers['x-goog-api-key'], apiKey);
    });
  });

  describe('generateContent', () => {
    const model = 'gemini-2.0-pro';
    const request: GenerateContentRequest = {
      contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
    };

    it('should return GenerateContentResponse', async () => {
      const mockResponse: GenerateContentResponse = {
        candidates: [
          { index: 0, content: { role: 'model', parts: [{ text: 'world' }] } },
        ],
      };
      mockFetchResponse(mockResponse);

      const result = await generateContent(apiKey, model, request);
      assert.deepStrictEqual(result, mockResponse);

      const expectedUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
      sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        },
        body: JSON.stringify(request),
      });
    });

    it('should throw on API error with JSON body', async () => {
      const errorResponse = { error: { message: 'Invalid Request' } };
      mockFetchResponse(errorResponse, false, 400, 'Bad Request');

      await assert.rejects(
        generateContent(apiKey, model, request),
        (err: any) => {
          assert.strictEqual(err.name, 'GenkitError');
          assert.strictEqual(err.status, 'INVALID_ARGUMENT');
          assert.match(
            err.message,
            /Error fetching from .* \[400 Bad Request\] Invalid Request/
          );
          return true;
        }
      );
    });

    it('should throw on API error with non-JSON body', async () => {
      mockFetchResponse('Bad Request', false, 400, 'Bad Request', 'text/plain');

      await assert.rejects(
        generateContent(apiKey, model, request),
        (err: any) => {
          assert.strictEqual(err.name, 'GenkitError');
          assert.strictEqual(err.status, 'INVALID_ARGUMENT');
          assert.match(
            err.message,
            /Error fetching from .* \[400 Bad Request\] Bad Request/
          );
          return true;
        }
      );
    });

    it('should throw on network failure', async () => {
      fetchSpy.rejects(new TypeError('Failed to fetch'));
      await assert.rejects(
        generateContent(apiKey, model, request),
        /Failed to fetch from .* Failed to fetch/
      );
    });
  });

  describe('embedContent', () => {
    const model = 'text-embedding-005';
    const request: EmbedContentRequest = {
      content: { role: 'user', parts: [{ text: 'test content' }] },
    };

    it('should return EmbedContentResponse', async () => {
      const mockResponse: EmbedContentResponse = {
        embedding: { values: [0.1, 0.2, 0.3] },
      };
      mockFetchResponse(mockResponse);

      const result = await embedContent(apiKey, model, request);
      assert.deepStrictEqual(result, mockResponse);

      const expectedUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent`;
      sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        },
        body: JSON.stringify(request),
      });
    });

    it('should throw on API error with JSON body', async () => {
      const errorResponse = { error: { message: 'Embedding failed' } };
      mockFetchResponse(errorResponse, false, 500, 'Internal Server Error');

      await assert.rejects(embedContent(apiKey, model, request), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'INTERNAL');
        assert.match(
          err.message,
          /Error fetching from .* \[500 Internal Server Error\] Embedding failed/
        );
        return true;
      });
    });

    it('should throw on API error with non-JSON body', async () => {
      mockFetchResponse(
        'Internal Server Error',
        false,
        500,
        'Internal Server Error',
        'text/plain'
      );

      await assert.rejects(embedContent(apiKey, model, request), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'INTERNAL');
        assert.match(
          err.message,
          /Error fetching from .* \[500 Internal Server Error\] Internal Server Error/
        );
        return true;
      });
    });

    it('should throw on network failure', async () => {
      fetchSpy.rejects(new TypeError('Network error'));
      await assert.rejects(
        embedContent(apiKey, model, request),
        /Failed to fetch from .* Network error/
      );
    });
  });

  describe('generateContentStream', () => {
    const model = 'gemini-2.5-flash';
    const defaultRequest: GenerateContentRequest = {
      contents: [{ role: 'user', parts: [{ text: 'stream test' }] }],
    };

    it('should process stream and return stream and aggregated response', async () => {
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "Hello "}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "World!"}]}}], "usageMetadata": {"totalTokenCount": 10}}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));

      const result: GenerateContentStreamResult = await generateContentStream(
        apiKey,
        model,
        defaultRequest
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
              content: { role: 'model', parts: [{ text: 'Hello ' }] },
            },
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: { role: 'model', parts: [{ text: 'World!' }] },
            },
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

      const expectedUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse`;
      sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        },
        body: JSON.stringify(defaultRequest),
      });
    });

    it('should aggregate parts for multiple candidates', async () => {
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "C0 A"}]}}, {"index": 1, "content": {"role": "model", "parts": [{"text": "C1 A"}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": " C0 B"}]}}, {"index": 1, "content": {"role": "model", "parts": [{"text": " C1 B"}]}}]}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(apiKey, model, defaultRequest);
      const aggregated = await result.response;

      assert.strictEqual(aggregated.candidates?.length, 2);
      const sortedCandidates = aggregated.candidates!.sort(
        (a, b) => a.index - b.index
      );

      assert.deepStrictEqual(sortedCandidates[0], {
        index: 0,
        content: {
          role: 'model',
          parts: [{ text: 'C0 A' }, { text: ' C0 B' }],
        },
      });
      assert.deepStrictEqual(sortedCandidates[1], {
        index: 1,
        content: {
          role: 'model',
          parts: [{ text: 'C1 A' }, { text: ' C1 B' }],
        },
      });
    });

    it('should use latest metadata from chunks for each candidate', async () => {
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "A"}]}, "finishReason": "STOP"}, {"index": 1, "content": {"role": "model", "parts": [{"text": "C1 "}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "B"}]}, "finishReason": "MAX_TOKENS", "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "LOW"}]}, {"index": 1, "content": {"role": "model", "parts": [{"text": "D1"}]}, "finishReason": "STOP" }], "usageMetadata": {"totalTokenCount": 20}}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(apiKey, model, defaultRequest);
      const aggregated = await result.response;

      assert.strictEqual(aggregated.candidates?.length, 2);
      const sortedCandidates = aggregated.candidates!.sort(
        (a, b) => a.index - b.index
      );

      const cand0 = sortedCandidates[0];
      assert.strictEqual(cand0.index, 0);
      assert.strictEqual(cand0.finishReason, FinishReason.MAX_TOKENS);
      assert.deepStrictEqual(cand0.safetyRatings, [
        {
          category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
          probability: HarmProbability.LOW,
        },
      ]);
      assert.deepStrictEqual(cand0.content.parts, [
        { text: 'A' },
        { text: 'B' },
      ]);

      const cand1 = sortedCandidates[1];
      assert.strictEqual(cand1.index, 1);
      assert.strictEqual(cand1.finishReason, FinishReason.STOP);
      assert.strictEqual(cand1.safetyRatings, undefined);
      assert.deepStrictEqual(cand1.content.parts, [
        { text: 'C1 ' },
        { text: 'D1' },
      ]);

      assert.deepStrictEqual(aggregated.usageMetadata, { totalTokenCount: 20 });
    });

    it('should aggregate different part types', async () => {
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "Action: "}]}}]}\n\n',
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"functionCall": {"name": "tool_call", "args": {}}}]}}]}\n\n',
      ];
      fetchSpy.resolves(createMockStream(chunks));
      const result = await generateContentStream(apiKey, model, defaultRequest);
      const aggregated = await result.response;
      const cand = aggregated.candidates![0];

      assert.deepStrictEqual(cand, {
        index: 0,
        content: {
          role: 'model',
          parts: [
            { text: 'Action: ' },
            { functionCall: { name: 'tool_call', args: {} } },
          ],
        },
      });
    });

    it('should handle stream with malformed JSON', async () => {
      const chunks = [
        'data: {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "Hello "}]}}]}\n\n',
        'data: {"candi dates": []}}}}\n\n', // Malformed
      ];
      fetchSpy.resolves(createMockStream(chunks));

      const result: GenerateContentStreamResult = await generateContentStream(
        apiKey,
        model,
        defaultRequest
      );

      const streamResults: GenerateContentResponse[] = [];
      try {
        for await (const item of result.stream) {
          streamResults.push(item);
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

    it('should throw error if fetch rejects for stream', async () => {
      fetchSpy.rejects(new Error('Stream failed to connect'));

      await assert.rejects(
        generateContentStream(apiKey, model, defaultRequest),
        /Failed to fetch from .* Stream failed to connect/
      );
    });

    it('should respect AbortSignal', async () => {
      const controller = new AbortController(); // The test's controller
      const clientOptions: ClientOptions = { signal: controller.signal };
      const model = 'gemini-2.5-flash';
      const defaultRequest: GenerateContentRequest = {
        contents: [{ role: 'user', parts: [{ text: 'stream test' }] }],
      };
      const apiKey = 'test-api-key';

      let capturedSignal: AbortSignal | undefined;

      fetchSpy.callsFake((url, options) => {
        return new Promise((resolve, reject) => {
          const signal = options?.signal;
          capturedSignal = signal; // Capture the signal passed to fetch

          if (signal) {
            if (signal.aborted) {
              return reject(
                new DOMException('The operation was aborted.', 'AbortError')
              );
            }
            signal.addEventListener(
              'abort',
              () => {
                reject(
                  new DOMException('The operation was aborted.', 'AbortError')
                );
              },
              { once: true }
            );
          }
          // Keep promise pending to simulate a real network request
        });
      });

      const promise = generateContentStream(
        apiKey,
        model,
        defaultRequest,
        clientOptions
      );

      // Short delay to allow fetchSpy to be called
      await new Promise((resolve) => setTimeout(resolve, 0));

      sinon.assert.calledOnce(fetchSpy);
      assert.ok(capturedSignal, 'A signal should be passed to fetch');
      assert.notStrictEqual(
        capturedSignal,
        controller.signal,
        'The signal passed to fetch should be a new signal, not the original one'
      );
      assert.strictEqual(
        capturedSignal!.aborted,
        false,
        'The passed signal should not be aborted initially'
      );

      // Abort the test's controller
      controller.abort();

      // Verify the promise from generateContentStream rejects due to the abort
      await assert.rejects(promise, (err: Error) => {
        assert.match(
          err.message,
          /Failed to fetch from .* The operation was aborted./
        );
        return true;
      });

      // Verify the signal captured within fetchSpy is now aborted
      assert.strictEqual(
        capturedSignal!.aborted,
        true,
        'The passed signal should be aborted after the original controller is aborted'
      );
    });
  });

  describe('imagenPredict', () => {
    const model = 'imagen-3.0-generate-001';
    const request: ImagenPredictRequest = {
      instances: [{ prompt: 'A cat sitting on a mat' }],
      parameters: { sampleCount: 1 },
    };

    it('should return ImagenPredictResponse', async () => {
      const mockResponse: ImagenPredictResponse = {
        predictions: [
          { bytesBase64Encoded: 'base64string', mimeType: 'image/png' },
        ],
      };
      mockFetchResponse(mockResponse);

      const result = await imagenPredict(apiKey, model, request);
      assert.deepStrictEqual(result, mockResponse);

      const expectedUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:predict`;
      sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        },
        body: JSON.stringify(request),
      });
    });

    it('should throw on API error', async () => {
      const errorResponse = { error: { message: 'Imagen failed' } };
      mockFetchResponse(errorResponse, false, 400, 'Bad Request');

      await assert.rejects(
        imagenPredict(apiKey, model, request),
        (err: any) => {
          assert.strictEqual(err.name, 'GenkitError');
          assert.strictEqual(err.status, 'INVALID_ARGUMENT');
          assert.match(
            err.message,
            /Error fetching from .* \[400 Bad Request\] Imagen failed/
          );
          return true;
        }
      );
    });

    it('should throw on network error', async () => {
      fetchSpy.rejects(new Error('Network issue'));
      await assert.rejects(
        imagenPredict(apiKey, model, request),
        /Failed to fetch from .* Network issue/
      );
    });
  });

  describe('veoPredict', () => {
    const model = 'veo-1.5-flash-0804';
    const request: VeoPredictRequest = {
      instances: [{ prompt: 'A video of a sunset' }],
      parameters: {},
    };

    it('should return VeoOperation', async () => {
      const mockResponse: VeoOperation = {
        name: 'operations/12345',
        done: false,
      };
      mockFetchResponse(mockResponse);

      const result = await veoPredict(apiKey, model, request);
      assert.deepStrictEqual(result, mockResponse);

      const expectedUrl = `https://generativelanguage.googleapis.com/v1beta/models/${model}:predictLongRunning`;
      sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        },
        body: JSON.stringify(request),
      });
    });

    it('should throw on API error', async () => {
      const errorResponse = { error: { message: 'Veo failed to start' } };
      mockFetchResponse(errorResponse, false, 500, 'Internal Server Error');

      await assert.rejects(veoPredict(apiKey, model, request), (err: any) => {
        assert.strictEqual(err.name, 'GenkitError');
        assert.strictEqual(err.status, 'INTERNAL');
        assert.match(
          err.message,
          /Error fetching from .* \[500 Internal Server Error\] Veo failed to start/
        );
        return true;
      });
    });
  });

  describe('veoCheckOperation', () => {
    const operationName = 'operations/12345';

    it('should return VeoOperation state', async () => {
      const mockResponse: VeoOperation = {
        name: operationName,
        done: true,
        response: {
          generateVideoResponse: {
            generatedSamples: [{ video: { uri: 'gs://bucket/video.mp4' } }],
          },
        },
      };
      mockFetchResponse(mockResponse);

      const result = await veoCheckOperation(apiKey, operationName);
      assert.deepStrictEqual(result, mockResponse);

      const expectedUrl = `https://generativelanguage.googleapis.com/v1beta/${operationName}`;
      sinon.assert.calledOnceWithExactly(fetchSpy, expectedUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': apiKey,
          'x-goog-api-client': getGenkitClientHeader(),
        },
      });
    });

    it('should throw on API error', async () => {
      const errorResponse = { error: { message: 'Operation not found' } };
      mockFetchResponse(errorResponse, false, 404, 'Not Found');

      await assert.rejects(
        veoCheckOperation(apiKey, operationName),
        (err: any) => {
          assert.strictEqual(err.name, 'GenkitError');
          assert.strictEqual(err.status, 'UNKNOWN');
          assert.match(
            err.message,
            /Error fetching from .* \[404 Not Found\] Operation not found/
          );
          return true;
        }
      );
    });
  });

  describe('TEST_ONLY.getAbortSignal', () => {
    let clock: sinon.SinonFakeTimers;

    beforeEach(() => {
      clock = sinon.useFakeTimers();
    });

    afterEach(() => {
      clock.restore();
    });

    it('should return undefined if no signal or timeout', () => {
      const signal = getAbortSignal({});
      assert.strictEqual(signal, undefined);
    });

    it('should return a signal that aborts after timeout', () => {
      const clientOptions: ClientOptions = { timeout: 100 };
      const signal = getAbortSignal(clientOptions);
      assert.ok(signal);
      assert.strictEqual(signal!.aborted, false);
      clock.tick(100);
      assert.strictEqual(signal!.aborted, true);
    });

    it('should return a signal linked to the provided signal', () => {
      const controller = new AbortController();
      const clientOptions: ClientOptions = { signal: controller.signal };
      const signal = getAbortSignal(clientOptions);
      assert.ok(signal);
      assert.strictEqual(signal!.aborted, false);
      controller.abort();
      assert.strictEqual(signal!.aborted, true);
    });

    it('should return a signal that aborts on timeout or provided signal', () => {
      const controller = new AbortController();
      const clientOptions: ClientOptions = {
        signal: controller.signal,
        timeout: 100,
      };
      const signal = getAbortSignal(clientOptions);
      assert.ok(signal);

      clock.tick(50);
      assert.strictEqual(signal!.aborted, false);
      controller.abort();
      assert.strictEqual(signal!.aborted, true);

      const signal2 = getAbortSignal(clientOptions);
      clock.tick(100);
      assert.strictEqual(signal2!.aborted, true);
    });
  });
});
