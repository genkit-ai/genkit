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
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { GeminiInteraction } from '../../src/common/interaction-types.js';
import {
  DeepResearchConfigSchema,
  TEST_ONLY,
  defineModel,
  model,
} from '../../src/vertexai/deep-research.js';
import {
  ExpressClientOptions,
  GlobalClientOptions,
  RegionalClientOptions,
} from '../../src/vertexai/types.js';

const MODEL_NAME = 'deep-research-preview-04-2026';

describe('Vertex AI Deep Research', () => {
  let fetchStub: sinon.SinonStub;
  let globalOptions: GlobalClientOptions;
  let defaultedOptions: RegionalClientOptions;

  beforeEach(() => {
    fetchStub = sinon.stub(global, 'fetch');
    const authMock = sinon.createStubInstance(GoogleAuth);
    authMock.getAccessToken.resolves('test-token');
    globalOptions = {
      kind: 'global',
      location: 'global',
      projectId: 'test-project',
      authClient: authMock,
    };
    defaultedOptions = {
      kind: 'regional',
      location: 'us-central1',
      locationDefaulted: true,
      projectId: 'test-project',
      authClient: authMock,
    };
  });

  afterEach(() => {
    sinon.restore();
  });

  function mockFetchResponse(body: unknown, status = 200) {
    fetchStub.callsFake(() =>
      Promise.resolve(
        new Response(JSON.stringify(body), {
          status,
          statusText: status === 200 ? 'OK' : 'Error',
          headers: { 'Content-Type': 'application/json' },
        })
      )
    );
  }

  function lastBody() {
    return JSON.parse(fetchStub.lastCall.args[1].body);
  }

  const request: GenerateRequest<typeof DeepResearchConfigSchema> = {
    messages: [
      { role: 'user', content: [{ text: 'Research quantum computing' }] },
    ],
  };

  const inProgress: GeminiInteraction = {
    id: 'v1_abc123',
    status: 'in_progress',
  };

  const completed: GeminiInteraction = {
    id: 'v1_abc123',
    status: 'completed',
    steps: [
      {
        type: 'model_output',
        content: [{ type: 'text', text: 'Here is the report...' }],
      },
    ],
  };

  describe('model()', () => {
    it('returns a known model with long-running support', () => {
      const ref = model(MODEL_NAME);
      assert.strictEqual(ref.name, `vertexai/${MODEL_NAME}`);
      // `longRunning` isn't in the ModelInfo type, so check it with `in`.
      const supports = ref.info?.supports;
      assert.ok(supports && 'longRunning' in supports);
      assert.strictEqual(supports.longRunning, true);
      assert.ok(Object.keys(TEST_ONLY.KNOWN_MODELS).includes(MODEL_NAME));
    });

    it('resolves unlisted deep-research names with the generic info', () => {
      const ref = model('deep-research-preview-12-2026');
      assert.strictEqual(ref.name, 'vertexai/deep-research-preview-12-2026');
      assert.deepStrictEqual(ref.info, TEST_ONLY.GENERIC_MODEL.info);
    });
  });

  describe('start', () => {
    it('sends a background agent request to the global location', async () => {
      mockFetchResponse(inProgress);
      const op = await defineModel(MODEL_NAME, defaultedOptions).start(request);

      assert.strictEqual(
        fetchStub.lastCall.args[0],
        'https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/interactions'
      );
      assert.strictEqual(
        fetchStub.lastCall.args[1].headers['Api-Revision'],
        '2026-05-20'
      );
      assert.deepStrictEqual(lastBody(), {
        agent: MODEL_NAME,
        input: [
          {
            type: 'user_input',
            content: [{ type: 'text', text: 'Research quantum computing' }],
          },
        ],
        background: true,
        agent_config: { type: 'deep-research' },
      });
      assert.strictEqual(op.id, 'v1_abc123');
      assert.strictEqual(op.done, false);
    });

    it('keeps a chosen regional location', async () => {
      mockFetchResponse(inProgress);
      const { locationDefaulted: _, ...chosen } = defaultedOptions;
      await defineModel(MODEL_NAME, chosen).start(request);
      assert.ok(
        fetchStub.lastCall.args[0].startsWith(
          'https://us-central1-aiplatform.googleapis.com/'
        )
      );
    });

    it('maps the agent config fields', async () => {
      mockFetchResponse(inProgress);
      await defineModel(MODEL_NAME, globalOptions).start({
        ...request,
        config: {
          thinkingSummaries: 'AUTO',
          visualization: 'OFF',
          responseModalities: ['TEXT', 'IMAGE'],
        },
      });
      const body = lastBody();
      assert.deepStrictEqual(body.agent_config, {
        type: 'deep-research',
        thinking_summaries: 'auto',
        visualization: 'off',
      });
      assert.deepStrictEqual(body.response_format, [
        { type: 'text' },
        { type: 'image' },
      ]);
      assert.strictEqual(body.response_modalities, undefined);
    });

    it('converts built-in tools from config.tools', async () => {
      mockFetchResponse(inProgress);
      await defineModel(MODEL_NAME, globalOptions).start({
        ...request,
        config: {
          tools: [
            { googleSearch: { searchTypes: ['enterpriseWebSearch'] } },
            { mcpServer: { name: 'docs', url: 'https://example.com/mcp' } },
          ],
        },
      });
      assert.deepStrictEqual(lastBody().tools, [
        { type: 'google_search', search_types: ['enterprise_web_search'] },
        { type: 'mcp_server', name: 'docs', url: 'https://example.com/mcp' },
      ]);
    });

    it('maps system messages to user messages', async () => {
      mockFetchResponse(inProgress);
      await defineModel(MODEL_NAME, globalOptions).start({
        messages: [
          { role: 'system', content: [{ text: 'Be brief' }] },
          ...request.messages,
        ],
      });
      const input = lastBody().input;
      assert.strictEqual(input[0].type, 'user_input');
      assert.strictEqual(input[0].content[0].text, 'Be brief');
    });

    it('sends JSON output as response_format', async () => {
      mockFetchResponse(inProgress);
      await defineModel(MODEL_NAME, globalOptions).start({
        ...request,
        output: {
          format: 'json',
          schema: { type: 'object', additionalProperties: false },
        },
      });
      assert.deepStrictEqual(lastBody().response_format, {
        type: 'text',
        mime_type: 'application/json',
        schema: { type: 'object' },
      });
    });
  });

  describe('store', () => {
    it('is not sent when nothing is set (server default)', async () => {
      mockFetchResponse(inProgress);
      await defineModel(MODEL_NAME, globalOptions).start(request);
      assert.strictEqual(lastBody().store, undefined);
    });

    it('sends the plugin store option, even false', async () => {
      mockFetchResponse(inProgress);
      await defineModel(MODEL_NAME, globalOptions, { store: false }).start(
        request
      );
      assert.strictEqual(lastBody().store, false);
    });

    it('prefers config.store over the plugin option', async () => {
      mockFetchResponse(inProgress);
      await defineModel(MODEL_NAME, globalOptions, { store: false }).start({
        ...request,
        config: { store: true },
      });
      assert.strictEqual(lastBody().store, true);
    });
  });

  describe('check and cancel', () => {
    it('polls the interaction on the global location', async () => {
      mockFetchResponse(completed);
      const op = await defineModel(MODEL_NAME, defaultedOptions).check({
        id: 'v1_abc123',
      });

      assert.strictEqual(
        fetchStub.lastCall.args[0],
        'https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/interactions/v1_abc123'
      );
      assert.strictEqual(fetchStub.lastCall.args[1].method, 'GET');
      assert.strictEqual(op.done, true);
      assert.strictEqual(
        op.output?.message?.content[0].text,
        'Here is the report...'
      );
    });

    it('cancels the interaction', async () => {
      mockFetchResponse({ id: 'v1_abc123', status: 'cancelled' });
      const op = await defineModel(MODEL_NAME, globalOptions).cancel?.({
        id: 'v1_abc123',
      });

      assert.strictEqual(
        fetchStub.lastCall.args[0],
        'https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/interactions/v1_abc123/cancel'
      );
      assert.strictEqual(fetchStub.lastCall.args[1].method, 'POST');
      assert.strictEqual(op?.done, true);
      assert.strictEqual(op?.output?.finishReason, 'aborted');
    });

    it('uses the Express Interactions URLs', async () => {
      const express: ExpressClientOptions = {
        kind: 'express',
        apiKey: 'test-api-key',
      };
      const action = defineModel(MODEL_NAME, express);

      mockFetchResponse(inProgress);
      await action.start(request);
      assert.strictEqual(
        fetchStub.lastCall.args[0],
        'https://aiplatform.googleapis.com/v1beta1/locations/global/interactions'
      );

      mockFetchResponse(completed);
      await action.check({ id: 'v1_abc123' });
      assert.strictEqual(
        fetchStub.lastCall.args[0],
        'https://aiplatform.googleapis.com/v1beta1/locations/global/interactions/v1_abc123'
      );
    });
  });
});
