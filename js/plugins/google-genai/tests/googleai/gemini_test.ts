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
import { z } from 'genkit';
import { GenerateRequest } from 'genkit/model';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import {
  GeminiConfigSchema,
  GeminiImageConfigSchema,
  GeminiTtsConfigSchema,
  defineModel,
  model,
} from '../../src/googleai/gemini.js';
import {
  FinishReason,
  GenerateContentRequest,
  GenerateContentResponse,
  GoogleAIPluginOptions,
} from '../../src/googleai/types.js';
import { MISSING_API_KEY_ERROR } from '../../src/googleai/utils.js';

describe('Google AI Gemini', () => {
  const ORIGINAL_ENV = { ...process.env };

  let fetchStub: sinon.SinonStub;

  beforeEach(() => {
    process.env = { ...ORIGINAL_ENV };
    delete process.env.GEMINI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.GOOGLE_GENAI_API_KEY;

    fetchStub = sinon.stub(global, 'fetch');
  });

  afterEach(() => {
    sinon.restore();
    process.env = { ...ORIGINAL_ENV };
  });

  // Mock fetch for non-streaming responses
  function mockFetchResponse(body: any, status = 200) {
    const response = new Response(JSON.stringify(body), {
      status: status,
      statusText: status === 200 ? 'OK' : 'Error',
      headers: { 'Content-Type': 'application/json' },
    });
    fetchStub.resolves(response);
  }

  // Mock fetch for streaming responses (SSE)
  function mockFetchStreamResponse(responses: GenerateContentResponse[]) {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        for (const response of responses) {
          const chunk = `data: ${JSON.stringify(response)}\n\n`;
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    });

    const response = new Response(stream, {
      status: 200,
      statusText: 'OK',
      headers: { 'Content-Type': 'text/event-stream' },
    });
    fetchStub.resolves(response);
  }

  const defaultPluginOptions: GoogleAIPluginOptions = {
    apiKey: 'test-api-key-plugin',
  };

  const minimalRequest: GenerateRequest<typeof GeminiConfigSchema> = {
    messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
  };

  const mockCandidate = {
    index: 0,
    content: {
      role: 'model',
      parts: [{ text: 'Hi there', thoughtSignature: 'test-signature' }],
    },
    finishReason: 'STOP' as FinishReason,
  };

  const defaultApiResponse: GenerateContentResponse = {
    candidates: [mockCandidate],
  };

  describe('defineModel', () => {
    describe('API Key Handling', () => {
      it('throws if no API key is provided', () => {
        assert.throws(() => {
          defineModel('gemini-2.5-flash');
        }, MISSING_API_KEY_ERROR);
      });

      it('uses API key from pluginOptions', async () => {
        const model = defineModel('gemini-2.5-flash', {
          apiKey: 'plugin-key',
        });
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);
        sinon.assert.calledOnce(fetchStub);
        const fetchOptions = fetchStub.lastCall.args[1];
        assert.strictEqual(
          fetchOptions.headers['x-goog-api-key'],
          'plugin-key'
        );
      });

      it('uses API key from GEMINI_API_KEY env var', async () => {
        process.env.GEMINI_API_KEY = 'gemini-key';
        const model = defineModel('gemini-2.5-flash');
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);
        const fetchOptions = fetchStub.lastCall.args[1];
        assert.strictEqual(
          fetchOptions.headers['x-goog-api-key'],
          'gemini-key'
        );
      });

      it('works if apiKey is false and not in call config', async () => {
        mockFetchResponse(defaultApiResponse);
        const model = defineModel('gemini-2.5-flash', { apiKey: false });
        assert.ok(await model.run(minimalRequest));
        sinon.assert.calledOnce(fetchStub);
      });

      it('uses API key from call config if apiKey is false', async () => {
        const model = defineModel('gemini-2.5-flash', { apiKey: false });
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { apiKey: 'call-time-key' },
        };
        await model.run(request);
        const fetchOptions = fetchStub.lastCall.args[1];
        assert.strictEqual(
          fetchOptions.headers['x-goog-api-key'],
          'call-time-key'
        );
      });
    });

    describe('Request Formation and API Calls', () => {
      it('calls fetch for non-streaming requests', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);
        sinon.assert.calledOnce(fetchStub);

        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        const options = fetchArgs[1];

        assert.ok(url.includes('models/gemini-2.5-flash:generateContent'));
        assert.strictEqual(options.method, 'POST');
        assert.strictEqual(
          options.headers['x-goog-api-key'],
          'test-api-key-plugin'
        );
        const body = JSON.parse(options.body);
        assert.deepStrictEqual(body.contents, [
          { role: 'user', parts: [{ text: 'Hello' }] },
        ]);
      });

      it('calls fetch for streaming requests', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchStreamResponse([defaultApiResponse]);

        const sendChunkSpy = sinon.spy();
        await model.run(minimalRequest, { onChunk: sendChunkSpy });

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        assert.ok(
          url.includes('models/gemini-2.5-flash:streamGenerateContent')
        );
        assert.ok(url.includes('alt=sse'));

        await new Promise((resolve) => setTimeout(resolve, 10)); // Allow stream to process

        sinon.assert.calledOnce(sendChunkSpy);
        const chunkArg = sendChunkSpy.lastCall.args[0];
        assert.deepStrictEqual(chunkArg, {
          index: 0,
          content: [
            {
              text: 'Hi there',
              metadata: { thoughtSignature: 'test-signature' },
            },
          ],
        });
      });

      it('passes AbortSignal to fetch', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const controller = new AbortController();
        const abortSignal = controller.signal;
        await model.run(minimalRequest, {
          abortSignal,
        });
        sinon.assert.calledOnce(fetchStub);
        const fetchOptions = fetchStub.lastCall.args[1];
        assert.ok(fetchOptions.signal, 'Fetch options should have a signal');
        assert.notStrictEqual(
          fetchOptions.signal,
          abortSignal,
          'Fetch signal should be a new signal, not the original'
        );

        const fetchSignal = fetchOptions.signal;
        const abortSpy = sinon.spy();
        fetchSignal.addEventListener('abort', abortSpy);
        controller.abort();
        sinon.assert.calledOnce(abortSpy);
      });

      it('handles system instructions', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          messages: [
            { role: 'system', content: [{ text: 'Be concise' }] },
            { role: 'user', content: [{ text: 'Hello' }] },
          ],
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.systemInstruction, {
          role: 'user',
          parts: [{ text: 'Be concise' }],
        });
        assert.deepStrictEqual(apiRequest.contents, [
          { role: 'user', parts: [{ text: 'Hello' }] },
        ]);
      });

      it('constructs tools array correctly', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          tools: [
            {
              name: 'myFunc',
              description: 'Does something',
              inputSchema: z.object({ foo: z.string() }),
              outputSchema: z.string(),
            },
          ],
          config: {
            codeExecution: true,
            googleSearchRetrieval: {},
            fileSearch: {
              fileSearchStoreNames: ['foo'],
            },
            urlContext: {},
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 5);
        assert.deepStrictEqual(apiRequest.tools?.[1], { codeExecution: {} });
        assert.deepStrictEqual(apiRequest.tools?.[2], {
          googleSearch: {},
        });
        assert.deepStrictEqual(apiRequest.tools?.[3], {
          fileSearch: {
            fileSearchStoreNames: ['foo'],
          },
        });
        assert.deepStrictEqual(apiRequest.tools?.[4], {
          urlContext: {},
        });
      });

      it('uses baseUrl and apiVersion from call config', async () => {
        const model = defineModel('gemini-2.5-flash', {
          ...defaultPluginOptions,
          baseUrl: 'https://my.custom.base.path',
          apiVersion: 'v1custom',
        });
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
        };
        await model.run(request);
        sinon.assert.calledOnce(fetchStub);

        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        assert.ok(
          url.startsWith('https://my.custom.base.path/v1custom/models'),
          `Expected URL to start with "https://my.custom.base.path/v1custom/models", but it was "${url}"`
        );
      });

      it('passes thinkingLevel to the API', async () => {
        const model = defineModel('gemini-3-pro-preview', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            thinkingConfig: {
              thinkingLevel: 'HIGH',
            },
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generationConfig, {
          thinkingConfig: {
            thinkingLevel: 'HIGH',
          },
        });
      });

      it('passes imageConfig to the API', async () => {
        const model = defineModel(
          'gemini-2.5-flash-image',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            imageConfig: {
              aspectRatio: '16:9',
              imageSize: '2K',
            },
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generationConfig, {
          imageConfig: {
            aspectRatio: '16:9',
            imageSize: '2K',
          },
        });
      });
    });

    describe('Media Handling', () => {
      const imageUrl = 'https://example.com/image.png';

      it('passes external URLs for non-Gemini 2.0 models', async () => {
        const model = defineModel(
          'gemini-3-flash-preview',
          defaultPluginOptions
        );

        fetchStub.callsFake(async (url: string | Request) => {
          if (typeof url === 'string' && url === imageUrl) {
            return new Response('image-data', {
              headers: { 'Content-Type': 'image/png' },
              status: 200,
            });
          }
          return new Response(JSON.stringify(defaultApiResponse), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        });

        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          messages: [
            {
              role: 'user',
              content: [{ media: { url: imageUrl, contentType: 'image/png' } }],
            },
          ],
        };

        await model.run(request);

        // Verify image was NOT downloaded
        assert.ok(
          !fetchStub.calledWith(imageUrl),
          'Should NOT attempt to download image for Gemini 3.0'
        );

        // Verify API request contained fileData
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        const part = apiRequest.contents[0].parts[0];
        assert.ok(part.fileData, 'Should be fileData');
        assert.strictEqual(part.fileData?.mimeType, 'image/png');
        assert.strictEqual(part.fileData?.fileUri, imageUrl);
      });
    });

    describe('Error Handling', () => {
      it('throws if no candidates are returned', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse({ candidates: [] });
        await assert.rejects(
          model.run(minimalRequest),
          /No valid candidates returned/
        );
      });

      it('throws on fetch error', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        fetchStub.rejects(new Error('Network error'));
        await assert.rejects(model.run(minimalRequest), /Failed to fetch/);
      });
    });

    describe('Debug Traces', () => {
      it('API call works with debugTraces: true', async () => {
        const model = defineModel('gemini-2.5-flash', {
          ...defaultPluginOptions,
          experimental_debugTraces: true,
        });

        mockFetchResponse(defaultApiResponse);
        await assert.doesNotReject(model.run(minimalRequest));
        sinon.assert.calledOnce(fetchStub);
      });

      it('API call works with debugTraces: false', async () => {
        const model = defineModel('gemini-2.5-flash', {
          ...defaultPluginOptions,
          experimental_debugTraces: false,
        });

        mockFetchResponse(defaultApiResponse);
        await assert.doesNotReject(model.run(minimalRequest));
        sinon.assert.calledOnce(fetchStub);
      });
    });
  });

  describe('gemini() function', () => {
    it('returns a ModelReference for a known model string', () => {
      const name = 'gemini-2.5-flash';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.info?.supports?.multiturn, true);
      assert.strictEqual(modelRef.configSchema, GeminiConfigSchema);
    });

    it('returns a ModelReference for a tts type model string', () => {
      const name = 'gemini-2.5-flash-preview-tts';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.info?.supports?.multiturn, false);
      assert.strictEqual(modelRef.configSchema, GeminiTtsConfigSchema);
    });

    it('returns a ModelReference for an image type model string', () => {
      const name = 'gemini-2.5-flash-image';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.info?.supports?.multiturn, true);
      assert.strictEqual(modelRef.configSchema, GeminiImageConfigSchema);
    });

    it('returns a ModelReference for an unknown model string', () => {
      const name = 'gemini-42.0-flash';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.info?.supports?.multiturn, true);
      assert.strictEqual(modelRef.configSchema, GeminiConfigSchema);
    });
  });
});
