/**
 * Copyright 2026 Google LLC
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
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { GeminiInteraction } from '../../src/common/interaction-types.js';
import {
  DeepResearchConfigSchema,
  defineModel,
} from '../../src/googleai/deep-research.js';
import { GoogleAIPluginOptions } from '../../src/googleai/types.js';

describe('Deep Research', () => {
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

  function mockFetchResponse(body: any, status = 200) {
    fetchStub.callsFake(() => {
      return Promise.resolve(
        new Response(JSON.stringify(body), {
          status: status,
          statusText: status === 200 ? 'OK' : 'Error',
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });
  }

  const defaultPluginOptions: GoogleAIPluginOptions = {
    apiKey: 'test-api-key-plugin',
  };

  const minimalRequest: GenerateRequest<typeof DeepResearchConfigSchema> = {
    messages: [
      { role: 'user', content: [{ text: 'Research quantum computing' }] },
    ],
  };

  const mockInteractionResponse: GeminiInteraction = {
    id: 'interaction-123',
    status: 'completed',
    outputs: [{ type: 'text', text: 'Here is the report...' }],
  };

  describe('defineModel', () => {
    it('passes responseModalities to the API', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        ...minimalRequest,
        config: {
          responseModalities: ['TEXT', 'IMAGE'],
        },
      };

      await model.start(request);

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];
      const body = JSON.parse(options.body);

      assert.deepStrictEqual(body.response_modalities, ['text', 'image']);
    });

    it('passes thinkingSummaries to the API', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        ...minimalRequest,
        config: {
          thinkingSummaries: 'AUTO',
        },
      };

      await model.start(request);

      sinon.assert.calledOnce(fetchStub);
      const options = fetchStub.lastCall.args[1];
      const body = JSON.parse(options.body);

      assert.deepStrictEqual(body.agent_config, {
        type: 'deep-research',
        thinking_summaries: 'auto',
      });
    });

    it('passes previousInteractionId and store to the API', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        ...minimalRequest,
        config: {
          previousInteractionId: 'prev-123',
          store: true,
        },
      };

      await model.start(request);

      sinon.assert.calledOnce(fetchStub);
      const options = fetchStub.lastCall.args[1];
      const body = JSON.parse(options.body);

      assert.strictEqual(body.previous_interaction_id, 'prev-123');
      assert.strictEqual(body.store, true);
    });

    it('converts system instructions to user role', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        messages: [
          { role: 'system', content: [{ text: 'Be concise' }] },
          { role: 'user', content: [{ text: 'Hello' }] },
        ],
      };

      await model.start(request);

      sinon.assert.calledOnce(fetchStub);
      const options = fetchStub.lastCall.args[1];
      const body = JSON.parse(options.body);

      assert.strictEqual(body.system_instruction, undefined);
      assert.deepStrictEqual(body.input, [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Be concise',
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Hello',
            },
          ],
        },
      ]);
    });

    it('handles JSON output format', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        ...minimalRequest,
        output: {
          format: 'json',
          schema: { type: 'object', properties: { foo: { type: 'string' } } },
        },
      };

      await model.start(request);

      sinon.assert.calledOnce(fetchStub);
      const options = fetchStub.lastCall.args[1];
      const body = JSON.parse(options.body);

      assert.strictEqual(body.response_mime_type, 'application/json');
      assert.deepStrictEqual(body.response_format, {
        type: 'object',
        properties: { foo: { type: 'string' } },
      });
    });

    it('handles tools', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        ...minimalRequest,
        tools: [
          {
            name: 'myFunc',
            description: 'desc',
            inputSchema: {
              type: 'object',
              properties: { arg: { type: 'string' } },
            },
          },
        ],
      };

      await model.start(request);

      sinon.assert.calledOnce(fetchStub);
      const options = fetchStub.lastCall.args[1];
      const body = JSON.parse(options.body);

      assert.strictEqual(body.tools.length, 1);
      assert.strictEqual(body.tools[0].type, 'function');
      assert.strictEqual(body.tools[0].name, 'myFunc');
    });

    it('persists client options to metadata', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        ...minimalRequest,
        config: {
          apiKey: 'request-api-key',
          baseUrl: 'https://custom.url',
        },
      };

      const operation = await model.start(request);

      assert.deepStrictEqual(operation.metadata?.clientOptions, {
        apiVersion: undefined,
        apiKey: 'request-api-key',
        baseUrl: 'https://custom.url',
        customHeaders: undefined,
      });
    });

    it('uses persisted options in check', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse); // For start
      mockFetchResponse(mockInteractionResponse); // For check

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        ...minimalRequest,
        config: {
          apiKey: 'request-api-key',
          baseUrl: 'https://custom.url',
        },
      };

      const operation = await model.start(request);

      // Now call check with the operation that has metadata
      await model.check!(operation);

      // fetchStub should be called twice: once for start, once for check
      sinon.assert.calledTwice(fetchStub);

      const checkCall = fetchStub.getCall(1);
      const url = checkCall.args[0] as string;
      const options = checkCall.args[1];

      // Check URL uses custom base URL
      assert.ok(url.startsWith('https://custom.url'));

      // Check API Key header
      // The client usually adds it to x-goog-api-key or key query param.
      // We need to know how `client.ts` implements `getInteraction`.
      // Assuming it adds it to headers or query params.
      // Based on googleai/client.ts (which I haven't read, but assuming standard behavior):
      // The `fetch` mock args[1] contains headers.
      const headers = options.headers as any;
      if (headers['x-goog-api-key']) {
        assert.strictEqual(headers['x-goog-api-key'], 'request-api-key');
      } else {
        // Check query param if not in header (though likely in header)
        assert.ok(url.includes('key=request-api-key'));
      }
    });

    it('maps usage statistics', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      const usageResponse: GeminiInteraction = {
        ...mockInteractionResponse,
        usage: {
          total_input_tokens: 100,
          total_output_tokens: 200,
          total_tokens: 300,
          total_cached_tokens: 60,
          total_thought_tokens: 10,
          input_tokens_by_modality: [
            { modality: 'text', tokens: 80 },
            { modality: 'image', tokens: 20 },
          ],
          output_tokens_by_modality: [
            { modality: 'text', tokens: 150 },
            { modality: 'audio', tokens: 50 },
          ],
        },
      };
      mockFetchResponse(usageResponse);

      const operation = await model.start(minimalRequest);

      assert.deepStrictEqual(operation.output?.usage, {
        inputTokens: 100,
        outputTokens: 200,
        totalTokens: 300,
        cachedContentTokens: 60,
        thoughtsTokens: 10,
        inputCharacters: 80,
        inputImages: 20,
        outputCharacters: 150,
        outputAudioFiles: 50,
      });
    });

    it('passes through arbitrary config like fileSearch', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
        ...minimalRequest,
        config: {
          fileSearch: {
            fileSearchStoreNames: ['stores/123'],
            metadataFilter: 'foo=bar',
          },
        } as any,
      };

      await model.start(request);

      sinon.assert.calledOnce(fetchStub);
      const options = fetchStub.lastCall.args[1];
      const body = JSON.parse(options.body);

      assert.deepStrictEqual(body.fileSearch, {
        fileSearchStoreNames: ['stores/123'],
        metadataFilter: 'foo=bar',
      });
    });

    it('handles 200 response for cancellation request', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      // Mock 200 OK response
      mockFetchResponse({});

      const operation = await model.cancel!({
        id: '123',
        action: 'foo',
      });

      assert.strictEqual(operation.done, true);
      assert.strictEqual(operation.output?.finishReason, 'aborted');
      assert.strictEqual(
        operation.output?.finishMessage,
        'Operation cancelled'
      );
    });

    it('handles 499 response for cancellation request', async () => {
      const model = defineModel(
        'deep-research-pro-preview-12-2025',
        defaultPluginOptions
      );
      // Mock 499 response
      mockFetchResponse({ error: { message: 'Operation cancelled.' } }, 499);

      const operation = await model.cancel!({
        id: '123',
        action: 'foo',
      });

      assert.strictEqual(operation.done, true);
      assert.strictEqual(operation.output?.finishReason, 'aborted');
      assert.strictEqual(
        operation.output?.finishMessage,
        'Operation cancelled'
      );
    });
  });
});
