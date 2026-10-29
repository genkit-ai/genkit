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
  AntigravityConfigSchema,
  defineModel,
} from '../../src/googleai/antigravity.js';
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
        'antigravity-preview-09-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      await model(minimalRequest, {} as any);

      sinon.assert.calledOnce(fetchStub);
      const fetchArgs = fetchStub.lastCall.args;
      const options = fetchArgs[1];
      const body = JSON.parse(options.body);

      assert.strictEqual(body.agent, 'antigravity-preview-09-2026');
      assert.strictEqual(body.store, undefined);
      assert.strictEqual(body.tools, undefined); // undefined explicitly enables default tools
    });

    it('maps system messages to user messages', async () => {
      const model = defineModel(
        'antigravity-preview-09-2026',
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
        'antigravity-preview-09-2026',
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

    it('sends config.model as agent_config.model', async () => {
      const model = defineModel(
        'antigravity-preview-09-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      await model(
        { ...minimalRequest, config: { model: 'gemini-3.5-flash-lite' } },
        {} as any
      );

      const body = JSON.parse(fetchStub.lastCall.args[1].body);
      assert.deepStrictEqual(body.agent_config, {
        type: 'antigravity',
        model: 'gemini-3.5-flash-lite',
      });
      assert.strictEqual(body.model, undefined);
      assert.strictEqual(body.agent, 'antigravity-preview-09-2026');
    });

    it('accepts model names that are not in the list', async () => {
      const model = defineModel(
        'antigravity-preview-09-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      await model(
        { ...minimalRequest, config: { model: 'gemini-9.9-flash' } },
        {} as any
      );

      const body = JSON.parse(fetchStub.lastCall.args[1].body);
      assert.deepStrictEqual(body.agent_config, {
        type: 'antigravity',
        model: 'gemini-9.9-flash',
      });
      assert.ok(
        AntigravityConfigSchema.safeParse({ model: 'gemini-9.9-flash' }).success
      );
    });

    it('sends function tools after the built-in tools from config', async () => {
      const model = defineModel(
        'antigravity-preview-09-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      await model(
        {
          ...minimalRequest,
          config: { tools: [{ type: 'code_execution' }] },
          tools: [
            {
              name: 'getWeather',
              description: 'Gets the weather',
              inputSchema: {
                type: 'object',
                properties: { location: { type: 'string' } },
                required: ['location'],
              },
            },
          ],
        },
        {} as any
      );

      const body = JSON.parse(fetchStub.lastCall.args[1].body);
      assert.deepStrictEqual(body.tools, [
        { type: 'code_execution' },
        {
          type: 'function',
          name: 'getWeather',
          description: 'Gets the weather',
          parameters: {
            type: 'object',
            properties: { location: { type: 'string' } },
            required: ['location'],
          },
        },
      ]);
    });

    it('returns a pending function call as a toolRequest', async () => {
      const model = defineModel(
        'antigravity-preview-09-2026',
        defaultPluginOptions
      );
      mockFetchResponse({
        id: 'v1_abc123',
        status: 'requires_action',
        environment_id: 'env-123',
        steps: [
          {
            type: 'function_call',
            id: 'call-1',
            name: 'getWeather',
            arguments: { location: 'Tokyo' },
          },
        ],
      });

      const response = await model(minimalRequest, {} as any);

      assert.deepStrictEqual(response.message?.content, [
        {
          toolRequest: {
            name: 'getWeather',
            ref: 'call-1',
            input: { location: 'Tokyo' },
          },
        },
      ]);
    });

    describe('tool loop', () => {
      const getWeatherTool = {
        name: 'getWeather',
        description: 'Gets the weather',
        inputSchema: {
          type: 'object',
          properties: { location: { type: 'string' } },
        },
      };

      // The history after the first turn of a tool loop, as Genkit sends it:
      // the prompt, the model's pending call (with the IDs from the
      // response metadata) and the tool's result.
      const toolLoopMessages: GenerateRequest<
        typeof AntigravityConfigSchema
      >['messages'] = [
        { role: 'user', content: [{ text: 'Weather in Tokyo?' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'getWeather',
                ref: 'call_58754',
                input: { location: 'Tokyo' },
              },
            },
          ],
          metadata: { interactionId: 'v1_abc123', environmentId: 'env-123' },
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'getWeather',
                ref: 'call_58754',
                output: { conditions: 'cloudy' },
              },
            },
          ],
        },
      ];

      it('continues the stored interaction by default', async () => {
        const model = defineModel(
          'antigravity-preview-09-2026',
          defaultPluginOptions
        );
        mockFetchResponse(mockInteractionResponse);

        await model(
          {
            messages: toolLoopMessages,
            tools: [getWeatherTool],
            config: { environment: 'remote' },
          },
          {} as any
        );

        const body = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.strictEqual(body.previous_interaction_id, 'v1_abc123');
        // A new 'remote' sandbox wouldn't have the previous turn's files.
        assert.strictEqual(body.environment, 'env-123');
        // Unset, so the server default (stored) applies.
        assert.strictEqual(body.store, undefined);
        assert.deepStrictEqual(body.input, [
          {
            type: 'function_result',
            name: 'getWeather',
            result: { conditions: 'cloudy' },
            call_id: 'call_58754',
          },
        ]);
      });

      it('continues a conversation without tool calls the same way', async () => {
        const model = defineModel(
          'antigravity-preview-09-2026',
          defaultPluginOptions
        );
        mockFetchResponse(mockInteractionResponse);

        await model(
          {
            messages: [
              { role: 'user', content: [{ text: 'Write hello.py' }] },
              {
                role: 'model',
                content: [{ text: 'Done.' }],
                metadata: {
                  interactionId: 'v1_abc123',
                  environmentId: 'env-123',
                },
              },
              { role: 'user', content: [{ text: 'Now run it.' }] },
            ],
          },
          {} as any
        );

        const body = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.strictEqual(body.previous_interaction_id, 'v1_abc123');
        assert.strictEqual(body.environment, 'env-123');
        assert.deepStrictEqual(body.input, [
          {
            type: 'user_input',
            content: [{ type: 'text', text: 'Now run it.' }],
          },
        ]);
      });

      it('keeps an explicitly configured environment ID', async () => {
        const model = defineModel(
          'antigravity-preview-09-2026',
          defaultPluginOptions
        );
        mockFetchResponse(mockInteractionResponse);

        await model(
          {
            messages: toolLoopMessages,
            tools: [getWeatherTool],
            config: { environment: 'env-other' },
          },
          {} as any
        );

        const body = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.strictEqual(body.environment, 'env-other');
      });

      it('sends only the newer messages when previousInteractionId is set', async () => {
        const model = defineModel(
          'antigravity-preview-09-2026',
          defaultPluginOptions
        );
        mockFetchResponse(mockInteractionResponse);

        await model(
          {
            messages: toolLoopMessages,
            tools: [getWeatherTool],
            config: { previousInteractionId: 'v1_explicit' },
          },
          {} as any
        );

        const body = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.strictEqual(body.previous_interaction_id, 'v1_explicit');
        assert.deepStrictEqual(
          body.input.map((s: { type: string }) => s.type),
          ['function_result']
        );
      });

      // The agent rejects store: false; the plugin passes it through so the
      // server's error explains it (and so it keeps working if the server
      // ever accepts it).
      it('passes store: false in config through to the server', async () => {
        const model = defineModel(
          'antigravity-preview-09-2026',
          defaultPluginOptions
        );
        mockFetchResponse(mockInteractionResponse);

        await model({ ...minimalRequest, config: { store: false } }, {} as any);

        const body = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.strictEqual(body.store, false);
      });

      it('passes store: false in the plugin options through to the server', async () => {
        const model = defineModel('antigravity-preview-09-2026', {
          ...defaultPluginOptions,
          store: false,
        });
        mockFetchResponse(mockInteractionResponse);

        await model(minimalRequest, {} as any);

        const body = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.strictEqual(body.store, false);
      });

      it('lets config.store override the plugin options', async () => {
        const model = defineModel('antigravity-preview-09-2026', {
          ...defaultPluginOptions,
          store: false,
        });
        mockFetchResponse(mockInteractionResponse);

        await model(
          {
            messages: toolLoopMessages,
            tools: [getWeatherTool],
            config: { store: true },
          },
          {} as any
        );

        const body = JSON.parse(fetchStub.lastCall.args[1].body);
        assert.strictEqual(body.store, true);
        assert.strictEqual(body.previous_interaction_id, 'v1_abc123');
      });
    });

    it('omits agent_config when no model is set', async () => {
      const model = defineModel(
        'antigravity-preview-09-2026',
        defaultPluginOptions
      );
      mockFetchResponse(mockInteractionResponse);

      await model(minimalRequest, {} as any);

      const body = JSON.parse(fetchStub.lastCall.args[1].body);
      assert.strictEqual(body.agent_config, undefined);
    });

    it('adds interaction.id and environment_id to message metadata', async () => {
      const model = defineModel(
        'antigravity-preview-09-2026',
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
        'antigravity-preview-09-2026',
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
        'antigravity-preview-09-2026',
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
