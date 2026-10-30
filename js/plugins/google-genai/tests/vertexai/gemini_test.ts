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
import { logger } from 'genkit/logging';
import { GenerateRequest, ModelReference } from 'genkit/model';
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { FinishReason } from '../../src/common/types.js';
import { getGenkitClientHeader } from '../../src/common/utils.js';
import { getVertexAIUrl } from '../../src/vertexai/client.js';
import {
  GeminiConfigSchema,
  GeminiImageConfigSchema,
  GeminiTtsConfigSchema,
  GeminiVideoConfigSchema,
  TEST_ONLY,
  defineModel,
  model,
} from '../../src/vertexai/gemini.js';
import {
  ClientOptions,
  GenerateContentRequest,
  GenerateContentResponse,
  HarmBlockThreshold,
  HarmCategory,
  isFunctionDeclarationsTool,
  isGoogleMapsTool,
  isGoogleSearchRetrievalTool,
  isRetrievalTool,
} from '../../src/vertexai/types.js';

describe('Vertex AI Gemini', () => {
  let fetchStub: sinon.SinonStub;
  let authMock: sinon.SinonStubbedInstance<GoogleAuth>;

  const defaultRegionalClientOptions: ClientOptions = {
    kind: 'regional',
    projectId: 'test-project',
    location: 'us-central1',
    authClient: {} as any,
  };

  const defaultGlobalClientOptions: ClientOptions = {
    kind: 'global',
    projectId: 'test-project',
    location: 'global',
    authClient: {} as any,
    apiKey: 'test-api-key',
  };

  const defaultExpressClientOptions: ClientOptions = {
    kind: 'express',
    apiKey: 'test-express-api-key',
  };

  const defaultMultiRegionalClientOptions: ClientOptions = {
    kind: 'multi-regional',
    projectId: 'test-project',
    location: 'eu',
    authClient: {} as any,
  };

  beforeEach(() => {
    authMock = sinon.createStubInstance(GoogleAuth);

    authMock.getAccessToken.resolves('test-token');
    defaultRegionalClientOptions.authClient = authMock as unknown as GoogleAuth;
    defaultGlobalClientOptions.authClient = authMock as unknown as GoogleAuth;
    defaultMultiRegionalClientOptions.authClient =
      authMock as unknown as GoogleAuth;

    fetchStub = sinon.stub(global, 'fetch');
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
    fetchStub.resolves(Promise.resolve(response));
  }

  const minimalRequest: GenerateRequest<typeof GeminiConfigSchema> = {
    messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
    config: {},
  };

  const mockCandidate = {
    index: 0,
    content: { role: 'model', parts: [{ text: 'Hi there' }] },
    finishReason: FinishReason.STOP,
  };

  const defaultApiResponse: GenerateContentResponse = {
    candidates: [mockCandidate],
  };

  describe('model() function', () => {
    it('returns a ModelReference for a known model string', () => {
      const name = 'gemini-2.5-flash';
      const modelRef: ModelReference<typeof GeminiConfigSchema> = model(name);
      assert.strictEqual(modelRef.name, `vertexai/${name}`);
      assert.ok(modelRef.info?.supports?.multiturn);
      assert.strictEqual(modelRef.configSchema, GeminiConfigSchema);
    });

    it('returns a ModelReference for an unknown model string', () => {
      const name = 'gemini-new-model';
      const modelRef: ModelReference<typeof GeminiConfigSchema> = model(name);
      assert.strictEqual(modelRef.name, `vertexai/${name}`);
      assert.ok(modelRef.info?.supports?.multiturn);
      assert.strictEqual(modelRef.configSchema, GeminiConfigSchema);
    });

    it('returns a ModelReference for a known image model string', () => {
      const name = 'gemini-3-pro-image-preview';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `vertexai/${name}`);
      assert.ok(modelRef.info?.supports?.multiturn);
      assert.strictEqual(modelRef.configSchema, GeminiImageConfigSchema);
    });

    it('returns a ModelReference for an unknown image model string', () => {
      const name = 'gemini-new-image-model';
      const modelRef = model(name);
      assert.strictEqual(modelRef.name, `vertexai/${name}`);
      assert.ok(modelRef.info?.supports?.multiturn);
      assert.strictEqual(modelRef.configSchema, GeminiImageConfigSchema);
    });

    it('treats nano-banana models as image models', () => {
      const modelRef = model('gemini-nano-banana-2.1');
      assert.strictEqual(modelRef.configSchema, GeminiImageConfigSchema);
    });

    it('keeps hidden models resolvable but out of the known lists', () => {
      const { HIDDEN_MODELS, ALL_MODELS, KNOWN_GEMINI_MODELS } = TEST_ONLY;
      for (const name of [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
      ]) {
        assert.ok(name in HIDDEN_MODELS, `${name} should be hidden`);
        assert.ok(name in ALL_MODELS);
        assert.ok(!(name in KNOWN_GEMINI_MODELS));
        assert.strictEqual(model(name).info?.stage, 'deprecated');
      }
    });

    it('returns a video ModelReference for Omni models', () => {
      for (const name of [
        'gemini-omni-1.1-flash-preview',
        'gemini-omni-9-flash',
      ]) {
        const modelRef = model(name);
        assert.strictEqual(modelRef.configSchema, GeminiVideoConfigSchema);
        assert.deepStrictEqual(modelRef.info?.supports?.output, ['media']);
      }
    });

    it('returns a TTS ModelReference for known and unknown TTS models', () => {
      for (const name of ['gemini-3.8-flash-tts', 'gemini-9-flash-tts']) {
        const modelRef = model(name);
        assert.strictEqual(modelRef.configSchema, GeminiTtsConfigSchema);
        assert.deepStrictEqual(modelRef.info?.supports?.output, ['media']);
      }
    });

    it('applies options to the ModelReference', () => {
      const options = { temperature: 0.9, topK: 20 };
      const modelRef: ModelReference<typeof GeminiConfigSchema> = model(
        'gemini-2.5-flash',
        options
      );
      assert.deepStrictEqual(modelRef.config, options);
    });

    it('applies image config options to the ModelReference', () => {
      const options = { imageConfig: { imageSize: '4K' } };
      const modelRef = model('gemini-3-pro-image-preview', options);
      assert.deepStrictEqual(modelRef.config, options);
    });
  });

  function runCommonTests(clientOptions: ClientOptions) {
    describe(`Model Action Callback ${clientOptions.kind}`, () => {
      function getExpectedHeaders(
        configApiKey?: string
      ): Record<string, string | undefined> {
        const headers: Record<string, string | undefined> = {
          'Content-Type': 'application/json',
          'X-Goog-Api-Client': getGenkitClientHeader(),
          'User-Agent': getGenkitClientHeader(),
        };
        if (clientOptions.kind !== 'express') {
          headers['Authorization'] = 'Bearer test-token';
          headers['x-goog-user-project'] = clientOptions.projectId;
        } else {
          headers['x-goog-api-key'] = clientOptions.apiKey as string;
        }
        if (configApiKey || clientOptions.apiKey) {
          headers['x-goog-api-key'] =
            configApiKey || clientOptions.apiKey || '';
        }
        return headers;
      }

      function getExpectedUrl(
        modelName: string,
        method: string,
        queryParams: string[] = []
      ): string {
        let baseUrl: string;
        let projectAndLocation = '';
        if (clientOptions.kind != 'express') {
          projectAndLocation = `projects/${clientOptions.projectId}/locations/${clientOptions.location}`;
        }

        if (clientOptions.kind === 'regional') {
          baseUrl = `https://${clientOptions.location}-aiplatform.googleapis.com/v1beta1/${projectAndLocation}`;
        } else if (clientOptions.kind === 'multi-regional') {
          baseUrl = `https://aiplatform.${clientOptions.location}.rep.googleapis.com/v1beta1/${projectAndLocation}`;
        } else if (clientOptions.kind === 'global') {
          baseUrl = `https://aiplatform.googleapis.com/v1beta1/${projectAndLocation}`;
        } else {
          // express
          baseUrl = `https://aiplatform.googleapis.com/v1beta1`;
        }

        let url = `${baseUrl}/publishers/google/models/${modelName}:${method}`;
        const params = [...queryParams];

        if (params.length > 0) {
          url += '?' + params.join('&');
        }
        return url;
      }

      it('throws if no messages are provided', async () => {
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await assert.rejects(
          model.run({ messages: [], config: {} }),
          /No messages provided/
        );
      });

      it('calls fetch for non-streaming requests', async () => {
        mockFetchResponse(defaultApiResponse);
        const model = defineModel('gemini-2.5-flash', clientOptions);
        const result = await model.run(minimalRequest);

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        const options = fetchArgs[1];

        const expectedUrl = getExpectedUrl(
          'gemini-2.5-flash',
          'generateContent'
        );
        assert.strictEqual(url, expectedUrl);
        assert.strictEqual(options.method, 'POST');
        const body = JSON.parse(options.body);
        assert.deepStrictEqual(body.contents, [
          { role: 'user', parts: [{ text: 'Hello' }] },
        ]);
        assert.strictEqual(body.labels, undefined);

        assert.deepStrictEqual(options.headers, getExpectedHeaders());
        assert.strictEqual(
          options.headers['X-Vertex-AI-LLM-Shared-Request-Type'],
          undefined
        );
        assert.strictEqual(
          options.headers['X-Vertex-AI-LLM-Request-Type'],
          undefined
        );

        assert.ok(result.result.candidates);
        assert.strictEqual(result.result.candidates.length, 1);
        assert.strictEqual(
          result.result.candidates[0].message.content[0].text,
          'Hi there'
        );
      });

      it('calls fetch for streaming requests', async () => {
        mockFetchStreamResponse([defaultApiResponse]);

        const sendChunkSpy = sinon.spy();
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(minimalRequest, { onChunk: sendChunkSpy });

        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];
        const expectedUrl = getExpectedUrl(
          'gemini-2.5-flash',
          'streamGenerateContent',
          ['alt=sse']
        );
        assert.strictEqual(url, expectedUrl);

        await new Promise((resolve) => setTimeout(resolve, 10));

        sinon.assert.calledOnce(sendChunkSpy);
        const chunkArg = sendChunkSpy.lastCall.args[0];
        assert.deepStrictEqual(chunkArg, {
          index: 0,
          content: [{ text: 'Hi there' }],
        });
      });

      it('passes AbortSignal to fetch', async () => {
        mockFetchResponse(defaultApiResponse);
        const controller = new AbortController();
        const abortSignal = controller.signal;
        const model = defineModel('gemini-2.5-flash', clientOptions);
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
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          messages: [
            { role: 'system', content: [{ text: 'Be concise' }] },
            { role: 'user', content: [{ text: 'Hello' }] },
          ],
          config: {},
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
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

      it('merges config from request', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { temperature: 0.1, topP: 0.8, maxOutputTokens: 100 },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(apiRequest.generationConfig?.temperature, 0.1);
        assert.strictEqual(apiRequest.generationConfig?.topP, 0.8);
        assert.strictEqual(apiRequest.generationConfig?.maxOutputTokens, 100);
      });

      it('passes thinkingLevel to the API', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            thinkingConfig: {
              thinkingLevel: 'HIGH',
            },
          },
        };
        const model = defineModel('gemini-3.1-pro-preview', clientOptions);
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

      it('handles imageConfig', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<any> = {
          ...minimalRequest,
          config: {
            imageConfig: {
              aspectRatio: '16:9',
              imageSize: '2K',
            },
          },
        };
        const model = defineModel('gemini-3-pro-image-preview', clientOptions);
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(
          (apiRequest.generationConfig as any)?.imageConfig,
          {
            aspectRatio: '16:9',
            imageSize: '2K',
          }
        );
      });

      it('sends labels when provided in config', async () => {
        mockFetchResponse(defaultApiResponse);
        const myLabels = { env: 'test', version: '1' };
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { labels: myLabels },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.labels, myLabels);
      });

      // The platform's generateContent has no `store` field and never stores
      // anything, so only `store: true` warns. The warning is logged once per
      // model and field, so each test uses its own unlisted image model name
      // (image models stay on generateContent) per client.
      it('warns once that store: true is ignored on generateContent models, and not for store: false', async () => {
        const warnStub = sinon.stub(logger, 'warn');
        const model = defineModel(
          `gemini-store-config-${clientOptions.kind}-image`,
          clientOptions
        );
        const storeWarnings = () =>
          warnStub
            .getCalls()
            .filter((c) =>
              String(c.args[0]).includes('store is not supported')
            );

        mockFetchResponse(defaultApiResponse);
        await model.run({ ...minimalRequest, config: { store: false } });
        assert.strictEqual(storeWarnings().length, 0);

        for (let i = 0; i < 2; i++) {
          mockFetchResponse(defaultApiResponse);
          await model.run({ ...minimalRequest, config: { store: true } });
          const apiRequest: GenerateContentRequest = JSON.parse(
            fetchStub.lastCall.args[1].body
          );
          assert.strictEqual('store' in apiRequest, false);
        }
        assert.strictEqual(storeWarnings().length, 1);
      });

      it('warns that the plugin store: true option is ignored on generateContent models', async () => {
        const warnStub = sinon.stub(logger, 'warn');
        const model = defineModel(
          `gemini-store-plugin-${clientOptions.kind}-image`,
          clientOptions,
          { store: true }
        );
        mockFetchResponse(defaultApiResponse);
        await model.run(minimalRequest);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual('store' in apiRequest, false);
        assert.ok(
          warnStub
            .getCalls()
            .some((c) => String(c.args[0]).includes('store is not supported')),
          'plugin store: true should warn'
        );
      });

      it('does not warn for a plugin store: false, or when the request overrides store: true with false', async () => {
        const warnStub = sinon.stub(logger, 'warn');
        const pluginFalse = defineModel(
          `gemini-store-plugin-false-${clientOptions.kind}-image`,
          clientOptions,
          { store: false }
        );
        mockFetchResponse(defaultApiResponse);
        await pluginFalse.run(minimalRequest);

        const overridden = defineModel(
          `gemini-store-overridden-${clientOptions.kind}-image`,
          clientOptions,
          { store: true }
        );
        mockFetchResponse(defaultApiResponse);
        await overridden.run({ ...minimalRequest, config: { store: false } });

        assert.ok(
          !warnStub
            .getCalls()
            .some((c) => String(c.args[0]).includes('store is not supported')),
          'store: false should not warn'
        );
      });

      it('warns that previousInteractionId is ignored on generateContent models', async () => {
        const warnStub = sinon.stub(logger, 'warn');
        const model = defineModel('gemini-2.5-flash', clientOptions);
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

      it('handles retrievalConfig', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            retrievalConfig: {
              latLng: {
                latitude: 37.7749,
                longitude: -122.4194,
              },
              languageCode: 'en-US',
            },
          },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.toolConfig?.retrievalConfig, {
          latLng: {
            latitude: 37.7749,
            longitude: -122.4194,
          },
          languageCode: 'en-US',
        });
      });

      it('constructs tools array with functionDeclarations', async () => {
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
          config: {},
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);

        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.ok(Array.isArray(apiRequest.tools));
        assert.strictEqual(apiRequest.tools?.length, 1);
        const tool = apiRequest.tools![0];
        assert.ok(
          isFunctionDeclarationsTool(tool),
          'Expected FunctionDeclarationsTool'
        );
        if (isFunctionDeclarationsTool(tool)) {
          assert.ok(tool.functionDeclarations);
          assert.strictEqual(tool.functionDeclarations?.length, 1);
          assert.strictEqual(tool.functionDeclarations?.[0].name, 'myFunc');
        }
      });

      it('handles googleSearchRetrieval tool (as googleSearch)', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            googleSearchRetrieval: {},
          },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        const searchTool = apiRequest.tools?.find(isGoogleSearchRetrievalTool);
        assert.ok(searchTool, 'Expected GoogleSearchRetrievalTool');
        if (searchTool) {
          assert.ok(searchTool.googleSearch, 'Expected googleSearch property');
          assert.deepStrictEqual(searchTool, { googleSearch: {} });
        }
      });

      it('handles googleMaps tool', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            tools: [{ googleMaps: { enableWidget: true } } as any],
          },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        const mapsTool = apiRequest.tools?.find(isGoogleMapsTool);
        assert.ok(mapsTool, 'Expected GoogleMapsTool');
        if (mapsTool) {
          assert.ok(mapsTool.googleMaps, 'Expected googleMaps property');
          assert.deepStrictEqual(mapsTool, {
            googleMaps: { enableWidget: true },
          });
        }
      });

      // Express has no project to fall back to.
      if (clientOptions.kind !== 'express') {
        it('handles vertexRetrieval tool', async () => {
          mockFetchResponse(defaultApiResponse);
          const request: GenerateRequest<typeof GeminiConfigSchema> = {
            ...minimalRequest,
            config: {
              vertexRetrieval: {
                datastore: { dataStoreId: 'my-store' },
                disableAttribution: true,
              },
            },
          };
          const model = defineModel('gemini-2.5-flash', clientOptions);
          await model.run(request);
          const apiRequest: GenerateContentRequest = JSON.parse(
            fetchStub.lastCall.args[1].body
          );
          const retrievalTool = apiRequest.tools?.find(isRetrievalTool);
          assert.ok(retrievalTool, 'Expected RetrievalTool');
          if (retrievalTool) {
            assert.ok(retrievalTool.retrieval, 'Expected retrieval property');
            assert.deepStrictEqual(retrievalTool, {
              retrieval: {
                vertexAiSearch: {
                  // The project and location come from the client options.
                  datastore: `projects/test-project/locations/${clientOptions.location}/collections/default_collection/dataStores/my-store`,
                },
                disableAttribution: true,
              },
            });
          }
        });
      }

      it('applies safetySettings', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: {
            safetySettings: [
              {
                category: 'HARM_CATEGORY_HATE_SPEECH',
                threshold: 'BLOCK_ONLY_HIGH',
              },
            ],
          },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.deepStrictEqual(apiRequest.safetySettings, [
          {
            category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
          },
        ]);
      });

      it('handles JSON output mode', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          output: { format: 'json' },
          config: {},
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        const apiRequest: GenerateContentRequest = JSON.parse(
          fetchStub.lastCall.args[1].body
        );
        assert.strictEqual(
          apiRequest.generationConfig?.responseMimeType,
          'application/json'
        );
      });

      it('throws if no candidates are returned', async () => {
        mockFetchResponse({ candidates: [] });
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await assert.rejects(
          model.run(minimalRequest),
          /No valid candidates returned/
        );
      });

      it('handles API call error', async () => {
        mockFetchResponse({ error: { message: 'API Error' } }, 400);
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await assert.rejects(
          model.run(minimalRequest),
          /Error fetching from .*?: \[400 Error\] API Error/
        );
      });

      it('handles config.apiKey override', async () => {
        mockFetchResponse(defaultApiResponse);
        const overrideKey = 'override-api-key';
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { apiKey: overrideKey },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];

        const expectedUrl = getExpectedUrl(
          'gemini-2.5-flash',
          'generateContent',
          []
        );
        assert.strictEqual(url, expectedUrl);
        assert.deepStrictEqual(
          fetchArgs[1].headers,
          getExpectedHeaders(overrideKey)
        );
      });

      it('handles config.location override', async () => {
        if (clientOptions.kind === 'express') {
          return; // location override not applicable to express
        }
        mockFetchResponse(defaultApiResponse);
        const overrideLocation = 'europe-west1';
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { location: overrideLocation },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];

        const newClientOptions = {
          ...clientOptions,
          location: overrideLocation,
          kind: 'regional' as const,
        };

        const expectedUrl = getVertexAIUrl({
          includeProjectAndLocation: true,
          resourcePath: 'publishers/google/models/gemini-2.5-flash',
          resourceMethod: 'generateContent',
          clientOptions: newClientOptions,
        });

        assert.strictEqual(url, expectedUrl);
        assert.ok(url.includes(overrideLocation));
      });

      it('handles config.apiVersion override', async () => {
        mockFetchResponse(defaultApiResponse);
        const overrideApiVersion = 'v1';
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { apiVersion: overrideApiVersion },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        sinon.assert.calledOnce(fetchStub);
        const fetchArgs = fetchStub.lastCall.args;
        const url = fetchArgs[0];

        const newClientOptions = {
          ...clientOptions,
          apiVersion: overrideApiVersion as 'v1' | 'v1beta1',
        };

        const expectedUrl = getVertexAIUrl({
          includeProjectAndLocation: true, // gets ignored inside getVertexAIUrl for express
          resourcePath: 'publishers/google/models/gemini-2.5-flash',
          resourceMethod: 'generateContent',
          clientOptions: newClientOptions,
        });

        assert.strictEqual(url, expectedUrl);
        assert.ok(url.includes(`/v1/`));
      });

      it('handles config.payGo priority spillover mode', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { payGo: 'priority' },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        sinon.assert.calledOnce(fetchStub);

        const fetchHeaders = fetchStub.lastCall.args[1].headers;
        assert.strictEqual(
          fetchHeaders['X-Vertex-AI-LLM-Shared-Request-Type'],
          'priority'
        );
        assert.strictEqual(
          fetchHeaders['X-Vertex-AI-LLM-Request-Type'],
          undefined
        );
      });

      it('handles config.payGo priority-only mode', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { payGo: 'priority-only' },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        sinon.assert.calledOnce(fetchStub);

        const fetchHeaders = fetchStub.lastCall.args[1].headers;
        assert.strictEqual(
          fetchHeaders['X-Vertex-AI-LLM-Shared-Request-Type'],
          'priority'
        );
        assert.strictEqual(
          fetchHeaders['X-Vertex-AI-LLM-Request-Type'],
          'shared'
        );
      });

      it('handles config.payGo flex spillover mode', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { payGo: 'flex' },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        sinon.assert.calledOnce(fetchStub);

        const fetchHeaders = fetchStub.lastCall.args[1].headers;
        assert.strictEqual(
          fetchHeaders['X-Vertex-AI-LLM-Shared-Request-Type'],
          'flex'
        );
        assert.strictEqual(
          fetchHeaders['X-Vertex-AI-LLM-Request-Type'],
          undefined
        );
      });

      it('handles config.payGo flex-only mode', async () => {
        mockFetchResponse(defaultApiResponse);
        const request: GenerateRequest<typeof GeminiConfigSchema> = {
          ...minimalRequest,
          config: { payGo: 'flex-only' },
        };
        const model = defineModel('gemini-2.5-flash', clientOptions);
        await model.run(request);
        sinon.assert.calledOnce(fetchStub);

        const fetchHeaders = fetchStub.lastCall.args[1].headers;
        assert.strictEqual(
          fetchHeaders['X-Vertex-AI-LLM-Shared-Request-Type'],
          'flex'
        );
        assert.strictEqual(
          fetchHeaders['X-Vertex-AI-LLM-Request-Type'],
          'shared'
        );
      });
    });
  }

  function runInteractionsTests(clientOptions: ClientOptions) {
    describe(`Interactions API ${clientOptions.kind}`, () => {
      // An unlisted model, so it uses the Interactions API.
      const interactionsModel = 'gemini-3.8-flash';

      // Shape seen live (Lyria 3 on the platform, and Gemini on the Gemini
      // API): no `id` or `usage` without `store`.
      const interactionResponse = {
        status: 'completed',
        role: 'model',
        object: 'interaction',
        model: interactionsModel,
        steps: [
          {
            type: 'model_output',
            content: [{ type: 'text', text: 'Hi there' }],
          },
        ],
      };

      function getExpectedInteractionsUrl(): string {
        if (clientOptions.kind === 'express') {
          return 'https://aiplatform.googleapis.com/v1beta1/locations/global/interactions';
        }
        const parent = `projects/${clientOptions.projectId}/locations/${clientOptions.location}`;
        if (clientOptions.kind === 'regional') {
          return `https://${clientOptions.location}-aiplatform.googleapis.com/v1beta1/${parent}/interactions`;
        }
        if (clientOptions.kind === 'multi-regional') {
          return `https://aiplatform.${clientOptions.location}.rep.googleapis.com/v1beta1/${parent}/interactions`;
        }
        return `https://aiplatform.googleapis.com/v1beta1/${parent}/interactions`;
      }

      function lastRequestBody() {
        return JSON.parse(fetchStub.lastCall.args[1].body);
      }

      it('sends unlisted models to the Interactions API', async () => {
        mockFetchResponse(interactionResponse);
        const model = defineModel(interactionsModel, clientOptions);
        const result = await model.run(minimalRequest);

        sinon.assert.calledOnce(fetchStub);
        const [url, options] = fetchStub.lastCall.args;
        assert.strictEqual(url, getExpectedInteractionsUrl());
        assert.strictEqual(options.headers['Api-Revision'], '2026-05-20');
        assert.deepStrictEqual(lastRequestBody(), {
          model: interactionsModel,
          generation_config: {},
          stream: false,
          input: [
            { type: 'user_input', content: [{ type: 'text', text: 'Hello' }] },
          ],
          store: false,
        });
        assert.strictEqual(result.result.message?.content[0].text, 'Hi there');
      });

      for (const knownModel of [
        'gemini-3.1-pro-preview',
        'gemini-3.1-pro-preview-customtools',
        'gemini-3.8-flash-cyber',
        // Hidden models keep their routing.
        'gemini-2.5-flash',
      ]) {
        it(`keeps ${knownModel} on generateContent`, async () => {
          mockFetchResponse(defaultApiResponse);
          const model = defineModel(knownModel, clientOptions);
          await model.run(minimalRequest);
          assert.match(fetchStub.lastCall.args[0], /:generateContent$/);
        });
      }

      it('sends Omni video format and config to the Interactions API', async () => {
        mockFetchResponse(interactionResponse);
        const model = defineModel(
          'gemini-omni-1.1-flash-preview',
          clientOptions
        );
        await model.run({
          ...minimalRequest,
          config: {
            videoFormat: { aspectRatio: '9:16', resolution: '1080p' },
            videoConfig: { task: 'text_to_video' },
          },
        });
        assert.strictEqual(
          fetchStub.lastCall.args[0],
          getExpectedInteractionsUrl()
        );
        const body = lastRequestBody();
        assert.deepStrictEqual(body.response_format, {
          type: 'video',
          aspect_ratio: '9:16',
          resolution: '1080p',
        });
        assert.deepStrictEqual(body.generation_config, {
          video_config: { task: 'text_to_video' },
        });
      });

      for (const imageModel of [
        'gemini-new-image-model',
        'gemini-nano-banana-2.1',
      ]) {
        it(`keeps unlisted image model ${imageModel} on generateContent`, async () => {
          mockFetchResponse(defaultApiResponse);
          const model = defineModel(imageModel, clientOptions);
          await model.run(minimalRequest);
          assert.match(fetchStub.lastCall.args[0], /:generateContent$/);
          assert.deepStrictEqual(
            JSON.parse(fetchStub.lastCall.args[1].body).generationConfig
              .responseModalities,
            ['TEXT', 'IMAGE']
          );
        });
      }

      for (const ttsModel of ['gemini-3.8-flash-tts', 'gemini-9-flash-tts']) {
        it(`keeps TTS model ${ttsModel} on generateContent`, async () => {
          mockFetchResponse(defaultApiResponse);
          const model = defineModel(ttsModel, clientOptions);
          await model.run({
            messages: [
              {
                role: 'user',
                content: [
                  {
                    text: 'Hi Jane!',
                    metadata: { speechMetadata: { speaker: 'Joe' } },
                  },
                ],
              },
            ],
            config: {
              speechConfig: {
                multiSpeakerVoiceConfig: {
                  speakerVoiceConfigs: [
                    { speaker: 'Joe', voiceConfig: { voice: 'Puck' } },
                    { speaker: 'Jane', voiceConfig: { voice: 'Kore' } },
                  ],
                },
              },
              audioFormat: { mimeType: 'audio/mulaw' },
            },
          });
          assert.match(fetchStub.lastCall.args[0], /:generateContent$/);
          const body = JSON.parse(fetchStub.lastCall.args[1].body);
          assert.deepStrictEqual(body.contents[0].parts, [
            { text: 'Hi Jane!', speechMetadata: { speaker: 'Joe' } },
          ]);
          assert.deepStrictEqual(body.generationConfig, {
            speechConfig: {
              multiSpeakerVoiceConfig: {
                speakerVoiceConfigs: [
                  { speaker: 'Joe', voiceConfig: { voice: 'Puck' } },
                  { speaker: 'Jane', voiceConfig: { voice: 'Kore' } },
                ],
              },
            },
            responseModalities: ['AUDIO'],
            responseFormat: [{ audio: { mimeType: 'AUDIO_MULAW' } }],
          });
        });
      }

      it('maps tools, tool choice and generation config', async () => {
        mockFetchResponse(interactionResponse);
        const model = defineModel(interactionsModel, clientOptions);
        await model.run({
          ...minimalRequest,
          messages: [
            { role: 'system', content: [{ text: 'Be brief.' }] },
            ...minimalRequest.messages,
          ],
          tools: [
            {
              name: 'getWeather',
              description: 'Gets the weather',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
          toolChoice: 'required',
          config: {
            temperature: 0.5,
            maxOutputTokens: 100,
            thinkingConfig: { thinkingLevel: 'LOW', includeThoughts: true },
            googleSearchRetrieval: {},
          },
        });

        const body = lastRequestBody();
        assert.strictEqual(body.system_instruction, 'Be brief.');
        assert.deepStrictEqual(body.tools, [
          {
            type: 'function',
            name: 'getWeather',
            description: 'Gets the weather',
            parameters: { type: 'object', properties: {} },
          },
          { type: 'google_search' },
        ]);
        assert.deepStrictEqual(body.generation_config, {
          temperature: 0.5,
          max_output_tokens: 100,
          thinking_level: 'low',
          thinking_summaries: 'auto',
          tool_choice: { allowed_tools: { mode: 'any' } },
        });
      });

      it('sends safety settings and labels', async () => {
        mockFetchResponse(interactionResponse);
        const model = defineModel(interactionsModel, clientOptions);
        await model.run({
          ...minimalRequest,
          config: {
            safetySettings: [
              {
                category: 'HARM_CATEGORY_HATE_SPEECH',
                threshold: 'BLOCK_LOW_AND_ABOVE',
              },
            ],
            labels: { team: 'genkit' },
          },
        });
        const body = lastRequestBody();
        assert.deepStrictEqual(body.safety_settings, [
          { type: 'hate_speech', threshold: 'block_low_and_above' },
        ]);
        assert.deepStrictEqual(body.labels, { team: 'genkit' });
      });

      it('does not send topK (the platform rejects top_k with a 400)', async () => {
        mockFetchResponse(interactionResponse);
        const model = defineModel(interactionsModel, clientOptions);
        await model.run({
          ...minimalRequest,
          config: { topK: 40, temperature: 0.5 },
        });
        assert.deepStrictEqual(lastRequestBody().generation_config, {
          temperature: 0.5,
        });
      });

      it('maps vertexRetrieval to a retrieval tool', async () => {
        mockFetchResponse(interactionResponse);
        const model = defineModel(interactionsModel, clientOptions);
        await model.run({
          ...minimalRequest,
          config: {
            vertexRetrieval: {
              datastore: {
                projectId: 'p',
                location: 'us-central1',
                dataStoreId: 'ds',
              },
            },
          },
        });
        assert.deepStrictEqual(lastRequestBody().tools, [
          {
            type: 'retrieval',
            retrieval_types: ['vertex_ai_search'],
            vertex_ai_search_config: {
              datastores: [
                'projects/p/locations/us-central1/collections/default_collection/dataStores/ds',
              ],
            },
          },
        ]);
      });

      it('resends only code execution steps of the server-side tools, and warns once', async () => {
        // The platform rejects resent Google Search, URL context and Google
        // Maps steps with a 501 (seen live, Oct 2026); file search and MCP
        // steps weren't seen working, so they're not resent either. Code
        // execution steps are accepted, so they're kept.
        TEST_ONLY.resetDroppedResentStepsWarning();
        const warnStub = sinon.stub(logger, 'warn');
        mockFetchResponse(interactionResponse);
        const model = defineModel(interactionsModel, clientOptions);
        const request = {
          messages: [
            { role: 'user' as const, content: [{ text: 'Research this.' }] },
            {
              role: 'model' as const,
              content: [
                {
                  custom: { googleSearchCall: { id: 's1' } },
                },
                { custom: { googleSearchResult: { callId: 's1' } } },
                { custom: { urlContextCall: { id: 'u1' } } },
                { custom: { urlContextResult: { callId: 'u1' } } },
                { custom: { googleMapsCall: { id: 'm1' } } },
                { custom: { googleMapsResult: { callId: 'm1' } } },
                { custom: { fileSearchCall: { id: 'f1' } } },
                { custom: { fileSearchResult: { callId: 'f1' } } },
                { custom: { mcpServerToolCall: { id: 'p1' } } },
                { custom: { mcpServerToolResult: { callId: 'p1' } } },
                {
                  custom: {
                    executableCode: { code: 'print(1)', language: 'PYTHON' },
                  },
                  metadata: { callId: 'c1' },
                },
                {
                  custom: {
                    codeExecutionResult: {
                      output: '1\n',
                      outcome: 'OUTCOME_OK',
                    },
                  },
                  metadata: { callId: 'c1' },
                },
                { text: 'Done.' },
              ],
            },
            { role: 'user' as const, content: [{ text: 'And then?' }] },
          ],
        };
        await model.run(request);

        const sentTypes = lastRequestBody().input.map(
          (step: { type: string }) => step.type
        );
        assert.deepStrictEqual(sentTypes, [
          'user_input',
          'code_execution_call',
          'code_execution_result',
          'model_output',
          'user_input',
        ]);
        const droppedWarnings = () =>
          warnStub
            .getCalls()
            .filter((c) => String(c.args[0]).includes('store: true'));
        assert.strictEqual(droppedWarnings().length, 1);

        // Warned once, not on every request.
        mockFetchResponse(interactionResponse);
        await model.run(request);
        assert.strictEqual(droppedWarnings().length, 1);
      });

      it('continues a stored interaction', async () => {
        mockFetchResponse({ ...interactionResponse, id: 'v1_def456' });
        const model = defineModel(interactionsModel, clientOptions);
        await model.run({
          messages: [
            { role: 'user', content: [{ text: 'Hi' }] },
            {
              role: 'model',
              content: [{ text: 'Hello' }],
              metadata: { interactionId: 'v1_abc123' },
            },
            { role: 'user', content: [{ text: 'Again' }] },
          ],
          config: { store: true },
        });

        const body = lastRequestBody();
        assert.strictEqual(body.store, true);
        assert.strictEqual(body.previous_interaction_id, 'v1_abc123');
        assert.deepStrictEqual(body.input, [
          { type: 'user_input', content: [{ type: 'text', text: 'Again' }] },
        ]);
      });

      it('uses the plugin-level store option', async () => {
        mockFetchResponse(interactionResponse);
        const model = defineModel(interactionsModel, clientOptions, {
          store: true,
        });
        await model.run({
          messages: [
            { role: 'user', content: [{ text: 'Hi' }] },
            {
              role: 'model',
              content: [{ text: 'Hello' }],
              // Interaction IDs on the platform look like this (seen live).
              metadata: { interactionId: 'ChAabc123' },
            },
            { role: 'user', content: [{ text: 'Again' }] },
          ],
          config: { previousInteractionId: 'ChAdef456' },
        });

        const body = lastRequestBody();
        assert.strictEqual(body.store, true);
        // An explicit previousInteractionId wins over the message metadata.
        assert.strictEqual(body.previous_interaction_id, 'ChAdef456');
        assert.deepStrictEqual(body.input, [
          { type: 'user_input', content: [{ type: 'text', text: 'Again' }] },
        ]);
      });

      it('rejects previousInteractionId without store', async () => {
        const model = defineModel(interactionsModel, clientOptions);
        await assert.rejects(
          model.run({
            ...minimalRequest,
            config: { previousInteractionId: 'v1_abc123' },
          }),
          /store must be true when previousInteractionId is set/
        );
        sinon.assert.notCalled(fetchStub);
      });

      it('streams text deltas', async () => {
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
          {
            event_type: 'step.delta',
            index: 0,
            delta: { type: 'text', text: 'Hi ' },
          },
          {
            event_type: 'step.delta',
            index: 0,
            delta: { type: 'text', text: 'there' },
          },
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

        const chunks: string[] = [];
        const model = defineModel(interactionsModel, clientOptions);
        const result = await model.run(minimalRequest, {
          onChunk: (chunk) =>
            chunks.push(...chunk.content.map((p) => p.text ?? '')),
        });

        assert.strictEqual(
          fetchStub.lastCall.args[0],
          getExpectedInteractionsUrl()
        );
        assert.strictEqual(lastRequestBody().stream, true);
        assert.deepStrictEqual(chunks, ['Hi ', 'there']);
        assert.strictEqual(
          result.result.message?.content.map((p) => p.text).join(''),
          'Hi there'
        );
      });

      it('streams a function call as a tool request chunk', async () => {
        // The event shapes seen live (gemini-3.8-flash): the call's
        // arguments arrive in one arguments_delta.
        const events = [
          {
            event_type: 'step.start',
            index: 0,
            step: {
              id: 'call_7227796',
              type: 'function_call',
              name: 'getWeather',
              arguments: {},
            },
          },
          {
            event_type: 'step.delta',
            index: 0,
            delta: {
              type: 'arguments_delta',
              arguments: '{"location":"Paris, France"}',
            },
          },
          {
            event_type: 'step.stop',
            index: 0,
            usage: { total_tokens: 210 },
          },
          {
            event_type: 'interaction.completed',
            interaction: { id: '', status: 'requires_action' },
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

        const chunkParts: unknown[] = [];
        const model = defineModel(interactionsModel, clientOptions);
        const result = await model.run(minimalRequest, {
          onChunk: (chunk) => chunkParts.push(...chunk.content),
        });

        const toolRequest = {
          toolRequest: {
            name: 'getWeather',
            ref: 'call_7227796',
            input: { location: 'Paris, France' },
          },
        };
        assert.deepStrictEqual(chunkParts, [toolRequest]);
        assert.deepStrictEqual(result.result.message?.content, [toolRequest]);
      });
    });
  }

  describe('defineModel - Regional Client', () => {
    runCommonTests(defaultRegionalClientOptions);
  });

  describe('defineModel - Global Client', () => {
    runCommonTests(defaultGlobalClientOptions);
  });

  describe('defineModel - Express Client', () => {
    runCommonTests(defaultExpressClientOptions);
  });

  describe('defineModel - Multi-Regional Client', () => {
    runCommonTests(defaultMultiRegionalClientOptions);
  });

  describe('defineModel - Interactions API', () => {
    runInteractionsTests(defaultRegionalClientOptions);
    runInteractionsTests(defaultGlobalClientOptions);
    runInteractionsTests(defaultExpressClientOptions);
    runInteractionsTests(defaultMultiRegionalClientOptions);

    it('handles every config schema field explicitly (Interactions API)', () => {
      // The Interactions API rejects unknown `generation_config` fields with
      // a 400, so every field in the plugin's config schemas must be either
      // handled in gemini.ts / toInteractionGenerationConfig, or known to be
      // accepted as a snake_case `generation_config` field. If this fails
      // after adding a schema field, map it explicitly (translate it, or
      // drop it with a warning) and add it to the right list below.
      const handledInGemini = [
        'apiKey',
        'apiVersion',
        'location',
        'payGo',
        'version',
        'labels',
        'safetySettings',
        'store',
        'previousInteractionId',
        'functionCallingConfig',
        'retrievalConfig',
        'googleSearchRetrieval',
        'vertexRetrieval',
        'responseModalities',
        'thinkingConfig',
        'tools',
        // TTS models stay on generateContent; listed so the TTS schema can
        // be checked too. Dropped with a warning on the Interactions path.
        'audioFormat',
        'speechConfig',
        // Sent as `response_format` entries.
        'videoFormat',
        'imageConfig',
        // Dropped with a warning: the platform rejects `top_k` with a 400.
        'topK',
      ];
      // Accepted in `generation_config` (checked live on the Gemini API).
      const acceptedByApi = [
        'temperature',
        'topP',
        'maxOutputTokens',
        'stopSequences',
      ];
      // A `generation_config` field in the Omni guide (video models); not
      // yet verified live on the platform.
      const documentedForApi = ['videoConfig'];
      const known = new Set([
        ...handledInGemini,
        ...acceptedByApi,
        ...documentedForApi,
      ]);
      for (const schema of [
        GeminiConfigSchema,
        GeminiImageConfigSchema,
        GeminiTtsConfigSchema,
        GeminiVideoConfigSchema,
      ]) {
        for (const field of Object.keys(schema.shape)) {
          assert.ok(known.has(field), `unhandled config field: ${field}`);
        }
      }
    });
  });
});
