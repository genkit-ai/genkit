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
import { LyriaConfigSchema, defineModel } from '../../src/googleai/lyria.js';
import { GoogleAIPluginOptions } from '../../src/googleai/types.js';

describe('Lyria 3', () => {
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

  const minimalRequest: GenerateRequest<typeof LyriaConfigSchema> = {
    messages: [
      { role: 'user', content: [{ text: 'Create an acoustic folk song' }] },
    ],
  };

  const mockInteractionResponse: GeminiInteraction = {
    id: 'interaction-123',
    status: 'completed',
    outputs: [
      { type: 'text', text: 'Here are the lyrics...' },
      { type: 'audio', mime_type: 'audio/mpeg', data: 'audio-data' },
    ],
  };

  describe('defineModel', () => {
    it('passes default responseModalities to the API', async () => {
      const model = defineModel('lyria-3-clip-preview', defaultPluginOptions);
      mockFetchResponse(mockInteractionResponse);

      await model(minimalRequest, {} as any);

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];
      const body = JSON.parse(options.body);

      assert.deepStrictEqual(body.response_modalities, ['audio', 'text']);
      assert.strictEqual(body.model, 'lyria-3-clip-preview');
    });

    it('passes explicit responseModalities to the API', async () => {
      const model = defineModel('lyria-3-clip-preview', defaultPluginOptions);
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof LyriaConfigSchema> = {
        ...minimalRequest,
        config: {
          responseModalities: ['AUDIO'],
        },
      };

      await model(request, {} as any);

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];
      const body = JSON.parse(options.body);

      assert.deepStrictEqual(body.response_modalities, ['audio']);
    });

    it('maps usage statistics correctly', async () => {
      const model = defineModel('lyria-3-pro-preview', defaultPluginOptions);
      const usageResponse: GeminiInteraction = {
        ...mockInteractionResponse,
        usage: {
          total_input_tokens: 15,
          total_output_tokens: 500,
          total_tokens: 515,
          total_cached_tokens: 0,
          total_thought_tokens: 0,
          input_tokens_by_modality: [{ modality: 'text', tokens: 15 }],
          output_tokens_by_modality: [
            { modality: 'text', tokens: 100 },
            { modality: 'audio', tokens: 400 },
          ],
        },
      };
      mockFetchResponse(usageResponse);

      const response = await model(minimalRequest, {} as any);

      assert.deepStrictEqual(response.usage, {
        inputTokens: 15,
        outputTokens: 500,
        totalTokens: 515,
        cachedContentTokens: 0,
        thoughtsTokens: 0,
        inputCharacters: 15,
        outputCharacters: 100,
        outputAudioFiles: 400,
      });
    });

    it('returns structured audio and text response', async () => {
      const model = defineModel('lyria-3-pro-preview', defaultPluginOptions);
      mockFetchResponse(mockInteractionResponse);

      const response = await model(minimalRequest, {} as any);

      assert.strictEqual(response.message?.content.length, 2);
      assert.strictEqual(
        response.message.content[0].text,
        'Here are the lyrics...'
      );
      assert.strictEqual(
        response.message.content[1].media?.contentType,
        'audio/mpeg'
      );
      assert.strictEqual(
        response.message.content[1].media?.url,
        'data:audio/mpeg;base64,audio-data'
      );
    });

    it('throws error for failed interaction status', async () => {
      const model = defineModel('lyria-3-pro-preview', defaultPluginOptions);
      const failedResponse: GeminiInteraction = {
        id: 'interaction-123',
        status: 'failed',
      };
      mockFetchResponse(failedResponse);

      await assert.rejects(
        model(minimalRequest, {} as any),
        /Interaction failed/
      );
    });
  });
});
