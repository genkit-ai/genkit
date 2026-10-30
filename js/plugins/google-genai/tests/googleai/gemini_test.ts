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
import { logger } from 'genkit/logging';
import { GenerateRequest } from 'genkit/model';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { CreateInteractionRequest } from '../../src/common/interaction-types.js';
import {
  GeminiConfigSchema,
  GeminiImageConfigSchema,
  GeminiTtsConfigSchema,
  GeminiVideoConfigSchema,
  GemmaConfigSchema,
  TEST_ONLY,
  defineModel,
  isGeminiModelName,
  isImageModelName,
  isVideoModelName,
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

  const jsonOutputSchema = {
    type: 'object',
    properties: { name: { type: 'string' } },
  };

  const constrainedJsonRequest: GenerateRequest<typeof GeminiConfigSchema> = {
    ...minimalRequest,
    output: {
      format: 'json',
      contentType: 'application/json',
      constrained: true,
      schema: jsonOutputSchema,
    },
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
        const model = defineModel('gemini-3.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          tools: [
            {
              name: 'myFunc',
              description: 'Does something',
              inputSchema: {
                type: 'object',
                properties: { foo: { type: 'string' } },
                required: ['foo'],
              },
              outputSchema: { type: 'string' },
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

      it('constructs google_search tool for images with empty searchTypes', async () => {
        const model = defineModel(
          'gemini-3.1-flash-image',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            google_search: {},
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          google_search: {},
        });
      });

      it('constructs google_search tool for images with webSearch', async () => {
        const model = defineModel(
          'gemini-3.1-flash-image',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            google_search: {
              searchTypes: { webSearch: {} },
            },
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          google_search: { searchTypes: { webSearch: {} } },
        });
      });

      it('constructs google_search tool for images with imageSearch', async () => {
        const model = defineModel(
          'gemini-3.1-flash-image',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            google_search: {
              searchTypes: { imageSearch: {} },
            },
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          google_search: { searchTypes: { imageSearch: {} } },
        });
      });

      it('constructs google_search tool for images with both webSearch and imageSearch', async () => {
        const model = defineModel(
          'gemini-3.1-flash-image',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            google_search: {
              searchTypes: { webSearch: {}, imageSearch: {} },
            },
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          google_search: { searchTypes: { webSearch: {}, imageSearch: {} } },
        });
      });

      it('constructs google_search tool for images with empty searchTypes (Interactions API)', async () => {
        const model = defineModel(
          'gemini-flash-latest', // uses Interactions API
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            google_search: {},
          },
        };
        await model.run(request);

        const apiRequest: any = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          type: 'google_search',
        });
      });

      it('constructs google_search tool for images with webSearch (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            google_search: {
              searchTypes: { webSearch: {} },
            },
          },
        };
        await model.run(request);

        const apiRequest: any = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          type: 'google_search',
          search_types: ['web_search'],
        });
      });

      it('constructs google_search tool for images with imageSearch (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            google_search: {
              searchTypes: { imageSearch: {} },
            },
          },
        };
        await model.run(request);

        const apiRequest: any = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          type: 'google_search',
          search_types: ['image_search'],
        });
      });

      it('constructs google_search tool for images with both webSearch and imageSearch (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiImageConfigSchema> = {
          ...minimalRequest,
          config: {
            google_search: {
              searchTypes: { webSearch: {}, imageSearch: {} },
            },
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          type: 'google_search',
          search_types: ['web_search', 'image_search'],
        });
      });

      it('constructs generation_config.tool_choice for functionCallingConfig (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            functionCallingConfig: {
              mode: 'ANY',
              allowedFunctionNames: ['myFunc'],
            },
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generation_config?.tool_choice, {
          allowed_tools: {
            mode: 'any',
            tools: ['myFunc'],
          },
        });
      });

      it('omits tool_choice when mode is MODE_UNSPECIFIED (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            functionCallingConfig: {
              mode: 'MODE_UNSPECIFIED',
            },
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.generation_config?.tool_choice,
          undefined
        );
      });

      it('constructs generation_config.tool_choice for toolChoice (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          toolChoice: 'required',
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generation_config?.tool_choice, {
          allowed_tools: {
            mode: 'any',
          },
        });
      });

      it('constructs generation_config.tool_choice for toolChoice none (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          toolChoice: 'none',
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generation_config?.tool_choice, {
          allowed_tools: {
            mode: 'none',
          },
        });
      });

      it('sends responseModalities as response_format entries (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            responseModalities: ['TEXT', 'AUDIO'],
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.response_format, [
          { type: 'text' },
          { type: 'audio' },
        ]);
        // Deprecated by the API: not sent.
        assert.strictEqual(apiRequest.response_modalities, undefined);
      });

      it('sends imageConfig on the image entry with text, for image models (Interactions API)', async () => {
        const model = defineModel(
          'gemini-nano-banana-2.1',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model
          .run({
            ...minimalRequest,
            config: { imageConfig: { aspectRatio: '9:16', imageSize: '2K' } },
          })
          .catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        // The same outputs as before (text and image), with the image
        // settings in the image entry (seen live to apply, Oct 2026).
        assert.deepStrictEqual(apiRequest.response_format, [
          { type: 'text' },
          { type: 'image', aspect_ratio: '9:16', image_size: '2K' },
        ]);
        assert.strictEqual(apiRequest.response_modalities, undefined);
        assert.ok(!('image_config' in (apiRequest.generation_config ?? {})));
      });

      it('asks image models for text and image by default (Interactions API)', async () => {
        const model = defineModel(
          'gemini-3.1-flash-lite-image',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest).catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.response_format, [
          { type: 'text' },
          { type: 'image' },
        ]);
      });

      it('keeps JSON output when audioFormat is set too (Interactions API)', async () => {
        // Used to be lost: the audio entry replaced the JSON text entry.
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model
          .run({
            ...minimalRequest,
            config: {
              responseModalities: ['TEXT', 'AUDIO'],
              audioFormat: { mimeType: 'audio/wav' },
            },
            output: {
              format: 'json',
              contentType: 'application/json',
              constrained: true,
              schema: jsonOutputSchema,
            },
          })
          .catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.response_format, [
          {
            type: 'text',
            mime_type: 'application/json',
            schema: jsonOutputSchema,
          },
          { type: 'audio', mime_type: 'audio/wav' },
        ]);
      });

      it('constructs toolConfig with retrievalConfig and googleMaps tool correctly', async () => {
        const model = defineModel(
          'gemini-3.1-pro-preview',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            tools: [{ googleMaps: {} }],
            retrievalConfig: {
              latLng: {
                latitude: 43.0896,
                longitude: -79.0849,
              },
            },
          } as any,
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.deepStrictEqual(apiRequest.tools?.[0], {
          googleMaps: {},
        });
        assert.deepStrictEqual(apiRequest.toolConfig, {
          retrievalConfig: {
            latLng: {
              latitude: 43.0896,
              longitude: -79.0849,
            },
          },
        });
        assert.strictEqual(
          (apiRequest.generationConfig as any).retrievalConfig,
          undefined,
          'retrievalConfig should not be in generationConfig'
        );
      });

      it('applies retrievalConfig.latLng to the google_maps tool (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run({
          ...minimalRequest,
          config: {
            tools: [{ googleMaps: { enableWidget: true } }],
            retrievalConfig: {
              latLng: { latitude: 43.0896, longitude: -79.0849 },
            },
          },
        });

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.tools, [
          {
            type: 'google_maps',
            enable_widget: true,
            latitude: 43.0896,
            longitude: -79.0849,
          },
        ]);
        assert.strictEqual(
          'retrieval_config' in (apiRequest.generation_config ?? {}),
          false
        );
      });

      it('warns that candidates > 1 is ignored (Interactions API)', async () => {
        const warnStub = sinon.stub(logger, 'warn');
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run({ ...minimalRequest, candidates: 2 });

        assert.ok(
          warnStub
            .getCalls()
            .some((c) => String(c.args[0]).includes('Multiple candidates')),
          'expected a warning about multiple candidates'
        );
      });

      // On generateContent, `store` controls whether the server logs the
      // request for the project; it's sent as is when set.
      for (const store of [true, false]) {
        it(`sends store: ${store} to generateContent models, without a warning`, async () => {
          const warnStub = sinon.stub(logger, 'warn');
          const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          await model.run({ ...minimalRequest, config: { store } });

          const apiRequest: GenerateContentRequest = JSON.parse(
            fetchStub.lastCall.args[1].body
          );
          assert.strictEqual(apiRequest.store, store);
          assert.ok(
            !warnStub
              .getCalls()
              .some((c) => String(c.args[0]).includes('store')),
            'store should not warn'
          );
        });
      }

      it('does not send store to generateContent models when it is not set', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual('store' in apiRequest, false);
      });

      it('sends the plugin store option to generateContent models; the request config wins', async () => {
        const model = defineModel('gemini-2.5-flash', {
          ...defaultPluginOptions,
          store: false,
        });
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);
        const fromPlugin: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(fromPlugin.store, false);

        mockFetchResponse(defaultApiResponse);
        await model.run({ ...minimalRequest, config: { store: true } });
        const fromRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(fromRequest.store, true);
      });

      it('warns that previousInteractionId is ignored on generateContent models', async () => {
        const warnStub = sinon.stub(logger, 'warn');
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run({
          ...minimalRequest,
          config: { previousInteractionId: 'v1_abc123' },
        });

        assert.ok(
          warnStub
            .getCalls()
            .some((c) =>
              String(c.args[0]).includes(
                'previousInteractionId is not supported'
              )
            ),
          'previousInteractionId should warn'
        );
      });

      it('never sends thinkingBudget or contextCache (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run({
          ...minimalRequest,
          config: {
            thinkingConfig: { thinkingLevel: 'HIGH', thinkingBudget: 1024 },
            contextCache: true,
          },
        });

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generation_config, {
          thinking_level: 'high',
        });
      });

      it('warns once per model when thinkingBudget is ignored (Interactions API)', async () => {
        const warnStub = sinon.stub(logger, 'warn');
        // An unlisted model (routed to Interactions) not used by other tests,
        // since the warning is only logged once per model and field.
        const model = defineModel(
          'gemini-thinking-budget-test',
          defaultPluginOptions
        );
        const request = {
          ...minimalRequest,
          config: { thinkingConfig: { thinkingBudget: 1024 } },
        };
        mockFetchResponse(defaultApiResponse);
        await model.run(request);
        mockFetchResponse(defaultApiResponse);
        await model.run(request);

        const budgetWarnings = warnStub
          .getCalls()
          .filter((c) => String(c.args[0]).includes('thinkingBudget'));
        assert.strictEqual(budgetWarnings.length, 1);
        assert.ok(String(budgetWarnings[0].args[0]).includes('thinkingLevel'));
      });

      it('drops topK with one warning per model (Interactions API)', async () => {
        const warnStub = sinon.stub(logger, 'warn');
        // An unlisted model (routed to Interactions) not used by other tests,
        // since the warning is only logged once per model and field.
        const model = defineModel('gemini-top-k-test', defaultPluginOptions);
        const request = {
          ...minimalRequest,
          config: { topK: 40, temperature: 0.5 },
        };
        mockFetchResponse(defaultApiResponse);
        await model.run(request).catch(() => undefined);
        mockFetchResponse(defaultApiResponse);
        await model.run(request).catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generation_config, {
          temperature: 0.5,
        });
        const topKWarnings = warnStub
          .getCalls()
          .filter((c) => String(c.args[0]).includes('topK'));
        assert.strictEqual(topKWarnings.length, 1);
      });

      it('still sends topK to generateContent models', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run({ ...minimalRequest, config: { topK: 40 } });

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.generationConfig?.topK, 40);
      });

      it('handles every config schema field explicitly (Interactions API)', () => {
        // The Interactions API rejects unknown `generation_config` fields with
        // a 400, so every field in the plugin's config schemas must be either
        // handled in gemini.ts / toInteractionGenerationConfig, or known to be
        // accepted as a snake_case `generation_config` field. If this fails
        // after adding a schema field, map it explicitly (translate it, or
        // drop it with a warning) and add it to the right list below.
        const handledInGemini = [
          'apiKey',
          'baseUrl',
          'apiVersion',
          'version',
          'safetySettings',
          'codeExecution',
          'functionCallingConfig',
          'responseModalities',
          'googleSearchRetrieval',
          'googleSearch',
          'google_search',
          'fileSearch',
          'urlContext',
          'retrievalConfig',
          'serviceTier',
          'previousInteractionId',
          'store',
          'speechConfig',
          'audioFormat',
          'videoFormat',
          'thinkingConfig',
          'contextCache',
          // Dropped with a warning (no `top_k` in generation_config).
          'topK',
          'toolConfig',
          'tools',
          // Sent as a `response_format` image entry.
          'imageConfig',
        ];
        // Verified live to be accepted in `generation_config`.
        const acceptedByApi = [
          'temperature',
          'topP',
          'maxOutputTokens',
          'stopSequences',
        ];
        // A `generation_config` field in the Omni guide (video models); not
        // yet verified live.
        const documentedForApi = ['videoConfig'];
        const known = new Set([
          ...handledInGemini,
          ...acceptedByApi,
          ...documentedForApi,
        ]);
        for (const schema of [
          GeminiConfigSchema,
          GeminiTtsConfigSchema,
          GeminiImageConfigSchema,
          GeminiVideoConfigSchema,
        ]) {
          for (const field of Object.keys(schema.shape)) {
            assert.ok(known.has(field), `unhandled config field: ${field}`);
          }
        }
      });

      it('sends any Gemma temperature to the server unchanged', async () => {
        // The plugin doesn't limit the range; the server validates it.
        assert.ok(GemmaConfigSchema.safeParse({ temperature: 1.5 }).success);

        const model = defineModel('gemma-4-31b-it', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run({ ...minimalRequest, config: { temperature: 1.5 } });

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.generationConfig?.temperature, 1.5);
      });

      it('still sends thinkingBudget to generateContent models', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run({
          ...minimalRequest,
          config: { thinkingConfig: { thinkingBudget: 1024 } },
        });

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generationConfig?.thinkingConfig, {
          thinkingBudget: 1024,
        });
      });

      it('keeps a location set on the googleMaps tool over retrievalConfig.latLng (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run({
          ...minimalRequest,
          config: {
            tools: [{ googleMaps: { latitude: 1, longitude: 2 } }],
            retrievalConfig: {
              latLng: { latitude: 43.0896, longitude: -79.0849 },
            },
          },
        });

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.tools, [
          { type: 'google_maps', latitude: 1, longitude: 2 },
        ]);
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
        const model = defineModel(
          'gemini-3.1-pro-preview',
          defaultPluginOptions
        );
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

      it('passes serviceTier to the API', async () => {
        const model = defineModel('gemini-3.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            serviceTier: 'flex',
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.serviceTier, 'flex');
      });

      it('passes serviceTier to the API (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions); // Uses Interactions API
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            serviceTier: 'flex',
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.service_tier, 'flex');
      });

      for (const name of [
        'gemini-4-argon',
        'gemini-3.8-flash',
        'gemini-3.8-flash-tts',
        'gemini-3.8-flash-lite-tts',
        'gemini-omni-1.1-flash',
      ]) {
        it(`routes ${name} to the Interactions API`, async () => {
          const model = defineModel(name, defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          // Only the request matters here; the mocked response isn't an
          // Interaction, so ignore how it's parsed.
          await model.run(minimalRequest).catch(() => undefined);

          sinon.assert.calledOnce(fetchStub);
          const url: string = fetchStub.lastCall.args[0];
          assert.ok(url.endsWith('/interactions'), url);
          const apiRequest: CreateInteractionRequest = JSON.parse(
            fetchStub.lastCall.args[1].body
          );
          assert.strictEqual(apiRequest.model, name);
        });
      }

      it('streams TTS as PCM chunks and returns them joined in one part', async () => {
        const model = defineModel(
          'gemini-3.8-flash-lite-tts',
          defaultPluginOptions
        );
        // The event shapes seen live: a model_output step whose audio arrives
        // as several `audio/l16` deltas.
        const pcm = [
          [1, 2],
          [3, 4],
          [5, 6],
        ];
        const events = [
          {
            event_type: 'interaction.created',
            interaction: { status: 'in_progress' },
          },
          {
            event_type: 'step.start',
            index: 0,
            step: { type: 'model_output' },
          },
          ...pcm.map((bytes) => ({
            event_type: 'step.delta',
            index: 0,
            delta: {
              type: 'audio',
              mime_type: 'audio/l16',
              data: Buffer.from(bytes).toString('base64'),
            },
          })),
          { event_type: 'step.stop', index: 0 },
          {
            event_type: 'interaction.completed',
            interaction: { status: 'completed' },
          },
        ];
        const encoder = new TextEncoder();
        fetchStub.resolves(
          new Response(
            new ReadableStream({
              start(controller) {
                for (const event of events) {
                  controller.enqueue(
                    encoder.encode(
                      `event: ${event.event_type}\ndata: ${JSON.stringify(event)}\n\n`
                    )
                  );
                }
                controller.close();
              },
            }),
            {
              status: 200,
              headers: { 'Content-Type': 'text/event-stream' },
            }
          )
        );

        const chunkTypes: (string | undefined)[] = [];
        const response = await model.run(minimalRequest, {
          onChunk: (chunk) =>
            chunkTypes.push(...chunk.content.map((p) => p.media?.contentType)),
        });

        assert.deepStrictEqual(chunkTypes, [
          'audio/l16',
          'audio/l16',
          'audio/l16',
        ]);
        // The final response has one part, in the format the server sent.
        const content = response.result.message?.content ?? [];
        assert.strictEqual(content.length, 1);
        assert.strictEqual(content[0].media?.contentType, 'audio/l16');
        const url = content[0].media?.url ?? '';
        assert.deepStrictEqual(
          Buffer.from(url.substring(url.indexOf(',') + 1), 'base64'),
          Buffer.from([1, 2, 3, 4, 5, 6])
        );
      });

      it('removes a C2PA ID3 tag from streamed audio/l16, in the chunks and the final WAV', async () => {
        const model = defineModel(
          'gemini-3.8-flash-lite-tts',
          defaultPluginOptions
        );
        // A small WAV like the server's first chunk: 44-byte header and two
        // bytes of audio.
        const wav = Buffer.alloc(46);
        wav.write('RIFF', 0, 'ascii');
        wav.writeUInt32LE(38, 4);
        wav.write('WAVE', 8, 'ascii');
        wav.write('fmt ', 12, 'ascii');
        wav.writeUInt32LE(16, 16);
        wav.writeUInt16LE(1, 20);
        wav.writeUInt16LE(1, 22);
        wav.writeUInt32LE(24000, 24);
        wav.writeUInt32LE(48000, 28);
        wav.writeUInt16LE(2, 32);
        wav.writeUInt16LE(16, 34);
        wav.write('data', 36, 'ascii');
        wav.writeUInt32LE(2, 40);
        wav.writeUInt16LE(0x0101, 44);
        // The tagged chunk seen live: a 6,125-byte ID3 tag, then PCM.
        const tag = Buffer.alloc(6125, 0x55);
        Buffer.from([
          0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00, 0x00, 0x2f, 0x63,
        ]).copy(tag);
        tag.write('GEOB', 10, 'ascii');
        const audio = (mime_type: string, bytes: Buffer) => ({
          event_type: 'step.delta',
          index: 0,
          delta: { type: 'audio', mime_type, data: bytes.toString('base64') },
        });
        const events = [
          {
            event_type: 'interaction.created',
            interaction: { status: 'in_progress' },
          },
          {
            event_type: 'step.start',
            index: 0,
            step: { type: 'model_output' },
          },
          audio('audio/wav', wav),
          audio('audio/l16', Buffer.from([2, 2])),
          audio('audio/l16', Buffer.concat([tag, Buffer.from([3, 3])])),
          { event_type: 'step.stop', index: 0 },
          {
            event_type: 'interaction.completed',
            interaction: { status: 'completed' },
          },
        ];
        const encoder = new TextEncoder();
        fetchStub.resolves(
          new Response(
            new ReadableStream({
              start(controller) {
                for (const event of events) {
                  controller.enqueue(
                    encoder.encode(
                      `event: ${event.event_type}\ndata: ${JSON.stringify(event)}\n\n`
                    )
                  );
                }
                controller.close();
              },
            }),
            { status: 200, headers: { 'Content-Type': 'text/event-stream' } }
          )
        );

        const chunkSizes: number[] = [];
        const response = await model.run(minimalRequest, {
          onChunk: (chunk) => {
            for (const p of chunk.content) {
              const url = p.media?.url ?? '';
              chunkSizes.push(
                Buffer.from(url.substring(url.indexOf(',') + 1), 'base64')
                  .length
              );
            }
          },
        });

        // The live chunks no longer contain the tag.
        assert.deepStrictEqual(chunkSizes, [46, 2, 2]);
        // The final WAV holds only the audio, with an even data size.
        const content = response.result.message?.content ?? [];
        assert.strictEqual(content.length, 1);
        assert.strictEqual(content[0].media?.contentType, 'audio/wav');
        const url = content[0].media?.url ?? '';
        const finalWav = Buffer.from(
          url.substring(url.indexOf(',') + 1),
          'base64'
        );
        assert.strictEqual(finalWav.readUInt32LE(40), 6);
        assert.deepStrictEqual(
          finalWav.subarray(44, 50),
          Buffer.from([1, 1, 2, 2, 3, 3])
        );
      });

      it('sends audioFormat as an audio response_format (Interactions API)', async () => {
        const model = defineModel('gemini-3.8-flash-tts', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model
          .run({
            ...minimalRequest,
            config: {
              audioFormat: { mimeType: 'audio/mulaw', sampleRate: 8000 },
            },
          })
          .catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.response_format, {
          type: 'audio',
          mime_type: 'audio/mulaw',
          sample_rate: 8000,
        });
        assert.ok(!('audio_format' in (apiRequest.generation_config ?? {})));
      });

      it('sends videoFormat as a video response_format (Interactions API)', async () => {
        const model = defineModel(
          'gemini-omni-1.1-flash',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model
          .run({
            ...minimalRequest,
            config: {
              store: true,
              videoFormat: {
                aspectRatio: '9:16',
                resolution: '1080p',
                delivery: 'uri',
              },
            },
          })
          .catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.response_format, {
          type: 'video',
          aspect_ratio: '9:16',
          resolution: '1080p',
          delivery: 'uri',
        });
        assert.ok(!('video_format' in (apiRequest.generation_config ?? {})));
      });

      it('sends videoConfig as generation_config.video_config (Interactions API)', async () => {
        const model = defineModel(
          'gemini-omni-1.1-flash',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model
          .run({
            ...minimalRequest,
            config: { videoConfig: { task: 'image_to_video' } },
          })
          .catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generation_config?.video_config, {
          task: 'image_to_video',
        });
      });

      it('sends no response_format for a video model without videoFormat (server defaults apply)', async () => {
        const model = defineModel(
          'gemini-omni-1.1-flash',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest).catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.response_format, undefined);
      });

      it('asks TTS models for audio without format settings when audioFormat is not set (server defaults apply)', async () => {
        const model = defineModel('gemini-3.8-flash-tts', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest).catch(() => undefined);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.response_format, { type: 'audio' });
      });

      it('drops permissive safetySettings on an Interactions model (default behavior)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            safetySettings: [
              {
                category: 'HARM_CATEGORY_HATE_SPEECH',
                threshold: 'BLOCK_NONE',
              },
              {
                category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold: 'BLOCK_NONE',
              },
              {
                category: 'HARM_CATEGORY_UNSPECIFIED',
                threshold: 'BLOCK_LOW_AND_ABOVE',
              },
            ],
          },
        };
        await model.run(request);

        const body = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.strictEqual(body.safety_settings, undefined);
        assert.strictEqual(body.generation_config?.safety_settings, undefined);
      });

      it('throws for blocking safetySettings on an Interactions model', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            safetySettings: [
              {
                category: 'HARM_CATEGORY_HATE_SPEECH',
                threshold: 'BLOCK_NONE',
              },
              {
                category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold: 'BLOCK_ONLY_HIGH',
              },
            ],
          },
        };
        await assert.rejects(
          () => model.run(request),
          (err: any) => {
            assert.strictEqual(err.status, 'INVALID_ARGUMENT');
            assert.ok(
              err.message.includes(
                "safetySettings with blocking thresholds are not supported for model 'gemini-flash-latest'"
              )
            );
            // Only the offending setting is reported.
            assert.deepStrictEqual(err.detail, {
              safetySettings: [
                {
                  category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
                  threshold: 'BLOCK_ONLY_HIGH',
                },
              ],
            });
            return true;
          }
        );
        sinon.assert.notCalled(fetchStub);
      });

      describe('TTS on an unlisted model (routed to Interactions)', () => {
        const ttsModel = 'gemini-3.8-flash-lite-tts';
        const multiSpeakerConfig = {
          speechConfig: {
            multiSpeakerVoiceConfig: {
              speakerVoiceConfigs: [
                {
                  speaker: 'Joe',
                  voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
                },
                {
                  speaker: 'Jane',
                  voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
                },
              ],
            },
          },
        };

        it('sends a single voice as speech_config and requests audio', async () => {
          const model = defineModel(ttsModel, defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          await model.run({
            ...minimalRequest,
            config: {
              speechConfig: {
                voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
              },
            },
          } as any);

          const [url, options] = fetchStub.lastCall.args;
          assert.ok(String(url).endsWith('/interactions'));
          const body: CreateInteractionRequest = JSON.parse(options.body);
          assert.deepStrictEqual(body.generation_config?.speech_config, [
            { voice: 'Puck' },
          ]);
          assert.deepStrictEqual(body.response_format, { type: 'audio' });
          assert.strictEqual(body.response_modalities, undefined);
        });

        it('sends multi-speaker turns with speech_metadata annotations', async () => {
          const model = defineModel(ttsModel, defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          await model.run({
            messages: [
              {
                role: 'user',
                content: [
                  {
                    text: "How's it going today Jane?",
                    metadata: { speechMetadata: { speaker: 'Joe' } },
                  },
                  {
                    text: 'Not too bad, how about you?',
                    metadata: {
                      speechMetadata: { speaker: 'Jane', style: 'calm' },
                    },
                  },
                ],
              },
            ],
            config: multiSpeakerConfig,
          } as any);

          const body: CreateInteractionRequest = JSON.parse(
            fetchStub.lastCall.args[1].body
          );
          assert.deepStrictEqual(body.generation_config?.speech_config, {
            speakers: [
              { voice: 'Puck', speaker: 'Joe' },
              { voice: 'Kore', speaker: 'Jane' },
            ],
          });
          const content = (body.input as any[])[0].content;
          assert.deepStrictEqual(
            content.map((c: any) => c.annotations),
            [
              [{ type: 'speech_metadata', speaker: 'Joe' }],
              [{ type: 'speech_metadata', speaker: 'Jane', style: 'calm' }],
            ]
          );
        });

        it('throws a clear error for multi-speaker without per-turn speakers', async () => {
          const model = defineModel(ttsModel, defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          await assert.rejects(
            () =>
              model.run({
                messages: [
                  {
                    role: 'user',
                    content: [{ text: 'Joe: Hi Jane!\nJane: Hi Joe!' }],
                  },
                ],
                config: multiSpeakerConfig,
              } as any),
            (err: any) => {
              assert.strictEqual(err.status, 'INVALID_ARGUMENT');
              assert.ok(err.message.includes('speechMetadata'));
              assert.ok(err.message.includes('Joe, Jane'));
              return true;
            }
          );
          sinon.assert.notCalled(fetchStub);
        });

        it('throws for a turn whose speaker is not configured', async () => {
          const model = defineModel(ttsModel, defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          await assert.rejects(
            () =>
              model.run({
                messages: [
                  {
                    role: 'user',
                    content: [
                      {
                        text: 'Hi Jane!',
                        metadata: { speechMetadata: { speaker: 'Joe' } },
                      },
                      {
                        text: 'Hi Joe!',
                        metadata: { speechMetadata: { speaker: 'Bob' } },
                      },
                    ],
                  },
                ],
                config: multiSpeakerConfig,
              } as any),
            (err: any) =>
              err.status === 'INVALID_ARGUMENT' &&
              err.message.includes('Joe, Jane')
          );
          sinon.assert.notCalled(fetchStub);
        });

        it('throws a clear error for speechMetadata without a voice', async () => {
          const model = defineModel(ttsModel, defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          await assert.rejects(
            () =>
              model.run({
                messages: [
                  {
                    role: 'user',
                    content: [
                      {
                        text: 'Have a wonderful day!',
                        metadata: { speechMetadata: { style: 'cheerful' } },
                      },
                    ],
                  },
                ],
              } as any),
            (err: any) =>
              err.status === 'INVALID_ARGUMENT' &&
              err.message.includes('requires a voice') &&
              err.message.includes('voiceName')
          );
          sinon.assert.notCalled(fetchStub);
        });

        it('allows plain text with no voice (default voice)', async () => {
          const model = defineModel(ttsModel, defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          await model.run({
            messages: [
              { role: 'user', content: [{ text: 'Have a wonderful day!' }] },
            ],
          } as any);
          sinon.assert.calledOnce(fetchStub);
        });

        it('ignores empty text parts when checking speakers', async () => {
          const model = defineModel(ttsModel, defaultPluginOptions);
          mockFetchResponse(defaultApiResponse);
          await model.run({
            messages: [
              {
                role: 'user',
                content: [
                  {
                    text: 'Hi Jane!',
                    metadata: { speechMetadata: { speaker: 'Joe' } },
                  },
                  { text: '  ' },
                ],
              },
            ],
            config: multiSpeakerConfig,
          } as any);
          sinon.assert.calledOnce(fetchStub);
        });
      });

      it('passes previousInteractionId to the API when store is true (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            previousInteractionId: 'interaction-123',
            store: true,
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.previous_interaction_id,
          'interaction-123'
        );
        assert.strictEqual(apiRequest.store, true);
      });

      it('throws when store: true is omitted but previousInteractionId is set (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            previousInteractionId: 'interaction-123',
          },
        };
        await assert.rejects(
          () => model.run(request),
          (err: any) => {
            assert.strictEqual(err.status, 'INVALID_ARGUMENT');
            assert.ok(
              err.message.includes(
                'store must be true when previousInteractionId is set'
              )
            );
            return true;
          }
        );
      });

      it('allows previousInteractionId without config store when pluginOptions has store: true (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', {
          ...defaultPluginOptions,
          store: true,
        });
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            previousInteractionId: 'interaction-123',
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.previous_interaction_id,
          'interaction-123'
        );
        assert.strictEqual(apiRequest.store, true);
      });

      it('extracts previousInteractionId from message metadata when not in config (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          messages: [
            { role: 'user', content: [{ text: 'Hello' }] },
            {
              role: 'model',
              content: [{ text: 'Hi' }],
              metadata: { interactionId: 'extracted-id-456' },
            },
            { role: 'user', content: [{ text: 'How are you?' }] },
          ],
          config: {
            store: true,
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.previous_interaction_id,
          'extracted-id-456'
        );
        assert.strictEqual(apiRequest.store, true);
        assert.deepStrictEqual(apiRequest.input, [
          {
            type: 'user_input',
            content: [{ type: 'text', text: 'How are you?' }],
          },
        ]);
      });

      it('config previousInteractionId overrides message metadata when both are present (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          messages: [
            { role: 'user', content: [{ text: 'Hello' }] },
            {
              role: 'model',
              content: [{ text: 'Hi' }],
              metadata: { interactionId: 'metadata-id-123' },
            },
            { role: 'user', content: [{ text: 'How are you?' }] },
          ],
          config: {
            previousInteractionId: 'config-override-id-789',
            store: true,
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.previous_interaction_id,
          'config-override-id-789'
        );
        assert.strictEqual(apiRequest.store, true);
        assert.deepStrictEqual(apiRequest.input, [
          {
            type: 'user_input',
            content: [{ type: 'text', text: 'How are you?' }],
          },
        ]);
      });

      it('does not extract previousInteractionId from message metadata when store is false (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          messages: [
            { role: 'user', content: [{ text: 'Hello' }] },
            {
              role: 'model',
              content: [{ text: 'Hi' }],
              metadata: { interactionId: 'metadata-id-123' },
            },
            { role: 'user', content: [{ text: 'How are you?' }] },
          ],
          config: {
            store: false,
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.previous_interaction_id, undefined);
        assert.strictEqual(apiRequest.store, false);
        assert.strictEqual((apiRequest.input as any[]).length, 3);
      });

      it('does not extract previousInteractionId from message metadata when store: true is omitted from config (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          messages: [
            { role: 'user', content: [{ text: 'Hello' }] },
            {
              role: 'model',
              content: [{ text: 'Hi' }],
              metadata: { interactionId: 'metadata-id-123' },
            },
            { role: 'user', content: [{ text: 'How are you?' }] },
          ],
          config: {},
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.previous_interaction_id, undefined);
        assert.strictEqual(apiRequest.store, false);
        assert.strictEqual((apiRequest.input as any[]).length, 3);
      });

      it('defaults store to false for Interactions API unless explicitly set', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.store, false);
      });

      it('passes store true when explicitly opted in (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            store: true,
          },
        };
        await model.run(request);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.store, true);
      });

      it('respects store from pluginOptions when set (Interactions API)', async () => {
        const model = defineModel('gemini-flash-latest', {
          ...defaultPluginOptions,
          store: true,
        });
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.store, true);
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
          responseModalities: ['TEXT', 'IMAGE'],
        });
      });

      it('defaults responseModalities to AUDIO for TTS models', async () => {
        const model = defineModel(
          'gemini-2.5-flash-preview-tts',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(
          apiRequest.generationConfig?.responseModalities,
          ['AUDIO']
        );
      });

      it('does not override responseModalities if specified for TTS models', async () => {
        const model = defineModel(
          'gemini-2.5-flash-preview-tts',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiTtsConfigSchema> = {
          ...minimalRequest,
          config: {
            responseModalities: ['TEXT'],
          },
        };
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(
          apiRequest.generationConfig?.responseModalities,
          ['TEXT']
        );
      });

      it('drops audioFormat and videoFormat on generateContent models', async () => {
        // The older preview TTS models use the TTS schema (which has
        // audioFormat) but are on generateContent, which rejects unknown
        // generation_config fields with a 400.
        const warnStub = sinon.stub(logger, 'warn');
        const model = defineModel(
          'gemini-2.5-flash-preview-tts',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model.run({
          ...minimalRequest,
          config: {
            audioFormat: { mimeType: 'audio/wav' },
            videoFormat: { resolution: '720p' },
          },
        });

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generationConfig, {
          responseModalities: ['AUDIO'],
        });
        const warnings = warnStub.getCalls().map((c) => String(c.args[0]));
        assert.ok(
          warnings.some((w) => w.startsWith('audioFormat is not supported')),
          warnings.join('\n')
        );
        assert.ok(
          warnings.some((w) => w.startsWith('videoFormat is not supported')),
          warnings.join('\n')
        );
      });

      it('does not default responseModalities to AUDIO for non-TTS models', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.generationConfig?.responseModalities,
          undefined
        );
      });

      it('sets a response schema for constrained JSON output', async () => {
        const model = defineModel('gemini-2.5-flash', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run(constrainedJsonRequest);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.generationConfig?.responseMimeType,
          'application/json'
        );
        assert.deepStrictEqual(
          apiRequest.generationConfig?.responseJsonSchema,
          jsonOutputSchema
        );
      });

      it('sets a legacy response schema for constrained JSON output', async () => {
        const model = defineModel('gemini-2.5-flash', {
          ...defaultPluginOptions,
          legacyResponseSchema: true,
        });
        mockFetchResponse(defaultApiResponse);
        await model.run(constrainedJsonRequest);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(
          apiRequest.generationConfig?.responseSchema,
          jsonOutputSchema
        );
        assert.strictEqual(
          apiRequest.generationConfig?.responseJsonSchema,
          undefined
        );
      });

      it('passes schema through untouched for Interactions API unless legacyResponseSchema is set', async () => {
        const model = defineModel('gemini-flash-latest', defaultPluginOptions);
        mockFetchResponse(defaultApiResponse);
        await model.run(constrainedJsonRequest);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(
          (apiRequest.response_format as any)?.schema,
          jsonOutputSchema
        );
      });

      it('cleans schema for Interactions API when legacyResponseSchema is set', async () => {
        const model = defineModel('gemini-flash-latest', {
          ...defaultPluginOptions,
          legacyResponseSchema: true,
        });
        mockFetchResponse(defaultApiResponse);
        await model.run(constrainedJsonRequest);

        const apiRequest: CreateInteractionRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(
          (apiRequest.response_format as any)?.schema,
          jsonOutputSchema
        );
      });

      it('simulates constrained generation for TTS models', async () => {
        const model = defineModel(
          'gemini-2.5-flash-preview-tts',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model.run(constrainedJsonRequest);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.generationConfig, {
          responseModalities: ['AUDIO'],
        });
        const lastMessage = apiRequest.contents[apiRequest.contents.length - 1];
        assert.ok(
          lastMessage.parts.some((part) =>
            part.text?.includes(
              'Output should be in JSON format and conform to the following schema'
            )
          )
        );
      });

      it('defaults responseModalities to TEXT, IMAGE for image models', async () => {
        const model = defineModel(
          'gemini-2.5-flash-image',
          defaultPluginOptions
        );
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(
          apiRequest.generationConfig?.responseModalities,
          ['TEXT', 'IMAGE']
        );
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
      assert.strictEqual(modelRef.info?.supports?.constrained, 'none');
      assert.deepStrictEqual(modelRef.info?.supports?.output, ['media']);
      assert.strictEqual(modelRef.configSchema, GeminiTtsConfigSchema);
    });

    it('returns a ModelReference for gemini-3.1-flash-tts-preview', () => {
      const name = 'gemini-3.1-flash-tts-preview';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.info?.supports?.multiturn, false);
      assert.strictEqual(modelRef.info?.supports?.constrained, 'none');
      assert.deepStrictEqual(modelRef.info?.supports?.output, ['media']);
      assert.strictEqual(modelRef.configSchema, GeminiTtsConfigSchema);
    });

    it('returns a ModelReference for an unknown tts model string', () => {
      const name = 'gemini-9.9-flash-tts';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.info?.supports?.multiturn, false);
      assert.strictEqual(modelRef.info?.supports?.constrained, 'none');
      assert.deepStrictEqual(modelRef.info?.supports?.output, ['media']);
      assert.strictEqual(modelRef.configSchema, GeminiTtsConfigSchema);
    });

    it('returns a ModelReference for an image type model string', () => {
      const name = 'gemini-2.5-flash-image';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.info?.supports?.multiturn, true);
      assert.strictEqual(modelRef.configSchema, GeminiImageConfigSchema);
    });

    it('returns a ModelReference for gemini-omni-1.1-flash (video)', () => {
      const name = 'gemini-omni-1.1-flash';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.configSchema, GeminiVideoConfigSchema);
      const supports = modelRef.info?.supports;
      assert.strictEqual(supports?.multiturn, true);
      assert.strictEqual(supports?.media, true);
      assert.strictEqual(supports?.tools, false);
      assert.strictEqual(supports?.systemRole, true);
      assert.strictEqual(supports?.constrained, 'none');
      assert.deepStrictEqual(supports?.output, ['media']);
    });

    it('returns a video ModelReference for an unknown gemini-omni model string', () => {
      const name = 'gemini-omni-9.9-flash';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.configSchema, GeminiVideoConfigSchema);
      assert.deepStrictEqual(
        modelRef.info?.supports,
        model('gemini-omni-1.1-flash').info?.supports
      );
    });

    it('treats gemini-omni-* names as video models, not Gemini models', () => {
      assert.strictEqual(isVideoModelName('gemini-omni-1.1-flash'), true);
      assert.strictEqual(isGeminiModelName('gemini-omni-1.1-flash'), false);
      assert.strictEqual(isVideoModelName('gemini-3.8-flash'), false);
    });

    it('treats gemini-nano-banana-* names as image models, not Gemini models', () => {
      assert.strictEqual(isImageModelName('gemini-nano-banana-2.1'), true);
      assert.strictEqual(isImageModelName('gemini-3.1-flash-lite-image'), true);
      assert.strictEqual(isImageModelName('gemini-3.8-flash'), false);
      assert.strictEqual(isGeminiModelName('gemini-nano-banana-2.1'), false);
      // An unlisted Nano Banana name gets the image config schema.
      assert.strictEqual(
        model('gemini-nano-banana-9').configSchema,
        GeminiImageConfigSchema
      );
    });

    it('returns a ModelReference for an unknown model string', () => {
      const name = 'gemini-42.0-flash';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `googleai/${name}`);
      assert.strictEqual(modelRef.info?.supports?.multiturn, true);
      assert.strictEqual(modelRef.configSchema, GeminiConfigSchema);
    });

    it('sets a stage on every hidden model', () => {
      for (const name of Object.keys(TEST_ONLY.HIDDEN_MODELS)) {
        const stage = model(name).info?.stage;
        assert.ok(
          stage === 'legacy' || stage === 'deprecated',
          `${name} has stage ${stage}`
        );
      }
    });

    it('keeps the default capabilities on hidden models', () => {
      // Passing `info` replaces the defaults, so a hidden model must copy them.
      assert.strictEqual(model('gemini-2.5-flash').info?.stage, 'legacy');
      assert.deepStrictEqual(
        model('gemini-2.5-flash').info?.supports,
        model('gemini-42.0-flash').info?.supports
      );

      assert.strictEqual(
        model('gemini-2.5-flash-preview-tts').info?.stage,
        'deprecated'
      );
      assert.deepStrictEqual(
        model('gemini-2.5-flash-preview-tts').info?.supports,
        model('gemini-9.9-flash-tts').info?.supports
      );

      assert.strictEqual(
        model('gemini-3.1-flash-image').info?.stage,
        'deprecated'
      );
      assert.deepStrictEqual(
        model('gemini-3.1-flash-image').info?.supports,
        model('gemini-9.9-flash-image').info?.supports
      );

      assert.strictEqual(model('gemini-3.5-flash').info?.stage, 'deprecated');
      assert.deepStrictEqual(
        model('gemini-3.5-flash').info?.supports,
        model('gemini-42.0-flash').info?.supports
      );
    });

    it('does not set a stage on known models', () => {
      for (const name of Object.keys(TEST_ONLY.KNOWN_MODELS)) {
        assert.strictEqual(model(name).info?.stage, undefined, name);
      }
    });

    it('describes gemini-nano-banana-2.1: tools, but no structured output', () => {
      const supports = model('gemini-nano-banana-2.1').info?.supports;
      assert.strictEqual(supports?.tools, true);
      assert.strictEqual(supports?.toolChoice, true);
      assert.strictEqual(supports?.constrained, 'none');
      assert.strictEqual(supports?.multiturn, true);
      assert.strictEqual(supports?.media, true);
    });
  });

  describe('image model capabilities', () => {
    // fetchStub and mockFetchResponse come from the outer describe.
    it('asks for JSON in the prompt instead of sending a schema for gemini-nano-banana-2.1', async () => {
      const imageModel = defineModel(
        'gemini-nano-banana-2.1',
        defaultPluginOptions
      );
      mockFetchResponse(defaultApiResponse);
      // Only the request matters here; the mocked response isn't an
      // Interaction, so ignore how it's parsed.
      await imageModel
        .run({
          messages: [{ role: 'user', content: [{ text: 'Name a landmark' }] }],
          output: {
            format: 'json',
            contentType: 'application/json',
            constrained: true,
            schema: {
              type: 'object',
              properties: { name: { type: 'string' } },
            },
          },
        })
        .catch(() => undefined);

      sinon.assert.calledOnce(fetchStub);
      const apiRequest: CreateInteractionRequest = JSON.parse(
        fetchStub.lastCall.args[1].body
      );
      // Only the image model's default outputs: no JSON text entry or schema.
      assert.deepStrictEqual(apiRequest.response_format, [
        { type: 'text' },
        { type: 'image' },
      ]);
      assert.ok(
        JSON.stringify(apiRequest.input).includes(
          'Output should be in JSON format and conform to the following schema'
        )
      );
    });
  });
});
