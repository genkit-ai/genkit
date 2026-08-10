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
import {
  AntigravityConfigSchema,
  defineModel,
} from '../../src/googleai/antigravity.js';
import { GeminiInteraction } from '../../src/googleai/interaction-types.js';
import { GoogleAIPluginOptions } from '../../src/googleai/types.js';

describe('Antigravity', () => {
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

  const minimalRequest: GenerateRequest<typeof AntigravityConfigSchema> = {
    messages: [{ role: 'user', content: [{ text: 'Hello antigravity' }] }],
  };

  const mockInteractionResponse: GeminiInteraction = {
    id: 'interaction-123',
    status: 'completed',
    environment_id: 'env-123',
    steps: [
      {
        type: 'model_output',
        content: [{ type: 'text', text: 'Hello human' }],
      },
    ],
  };

  describe('defineModel', () => {
    it('sets agent field and correctly maps empty config', async () => {
      const model = defineModel(
        'antigravity-preview-05-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      await model(minimalRequest, {} as any);

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];
      const body = JSON.parse(options.body);

      assert.strictEqual(body.agent, 'antigravity-preview-05-2026');
      assert.strictEqual(body.store, undefined);
      assert.strictEqual(body.tools, undefined); // undefined explicitly enables default tools
    });

    it('maps system messages to user messages', async () => {
      const model = defineModel(
        'antigravity-preview-05-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof AntigravityConfigSchema> = {
        messages: [
          { role: 'system', content: [{ text: 'You are an agent' }] },
          { role: 'user', content: [{ text: 'Hello' }] },
        ],
      };

      await model(request, {} as any);

      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];
      const body = JSON.parse(options.body);

      // System role should be mapped to user
      assert.strictEqual(body.input[0].type, 'user_input');
      assert.strictEqual(body.input[0].content[0].text, 'You are an agent');
      assert.strictEqual(body.input[1].type, 'user_input');
      assert.strictEqual(body.input[1].content[0].text, 'Hello');
    });

    it('maps explicitly provided config options correctly', async () => {
      const model = defineModel(
        'antigravity-preview-05-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof AntigravityConfigSchema> = {
        ...minimalRequest,
        config: {
          environment: 'remote',
          previousInteractionId: 'interaction-abc',
          tools: [{ type: 'google_search' }, { type: 'code_execution' }],
          store: true,
        },
      };

      await model(request, {} as any);

      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];
      const body = JSON.parse(options.body);

      assert.strictEqual(body.environment, 'remote');
      assert.strictEqual(body.store, true);
      assert.strictEqual(body.previous_interaction_id, 'interaction-abc');
      assert.deepStrictEqual(body.tools, [
        { type: 'google_search' },
        { type: 'code_execution' },
      ]);
    });

    it('adds interaction.id and environment_id to message metadata', async () => {
      const model = defineModel(
        'antigravity-preview-05-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const response = await model(minimalRequest, {} as any);

      assert.strictEqual(
        response.message?.metadata?.interactionId,
        'interaction-123'
      );
      assert.strictEqual(response.message?.metadata?.environmentId, 'env-123');
    });

    it('passes unmapped config directly to API via rest', async () => {
      const model = defineModel(
        'antigravity-preview-05-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const request: GenerateRequest<typeof AntigravityConfigSchema> = {
        ...minimalRequest,
        config: {
          temperature: 0.8,
          topP: 0.9,
        } as any,
      };

      await model(request, {} as any);

      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];
      const body = JSON.parse(options.body);

      assert.strictEqual(body.temperature, 0.8);
      assert.strictEqual(body.topP, 0.9);
    });

    it('routes abortSignal to fetch correctly', async () => {
      const model = defineModel(
        'antigravity-preview-05-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      const abortController = new AbortController();
      const abortSignal = abortController.signal;

      await model(minimalRequest, { abortSignal });

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];

      assert.ok(options.signal instanceof AbortSignal);
    });
  });
});
