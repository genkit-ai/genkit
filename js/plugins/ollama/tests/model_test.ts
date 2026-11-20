/**
 * Copyright 2024 Google LLC
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
import { genkit, z, type Genkit } from 'genkit';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { ollama } from '../src/index.js';
import type { OllamaPluginParams } from '../src/types.js';

const MOCK_TOOL_CALL_RESPONSE = {
  model: 'llama3.2',
  created_at: '2024-07-22T20:33:28.123648Z',
  message: {
    role: 'assistant',
    content: '',
    tool_calls: [
      {
        function: {
          name: 'get_current_weather',
          arguments: {
            format: 'celsius',
            location: 'Paris, FR',
          },
        },
      },
    ],
  },
  done_reason: 'stop',
  done: true,
};

const MOCK_END_RESPONSE = {
  model: 'llama3.2',
  created_at: '2024-07-22T20:33:28.123648Z',
  message: {
    role: 'assistant',
    content: 'The weather is sunny',
  },
  done_reason: 'stop',
  done: true,
};

const MAGIC_WORD = 'sunnnnnnny';

// Mock fetch to simulate API responses
global.fetch = async (input: RequestInfo | URL, options?: RequestInit) => {
  const url = typeof input === 'string' ? input : input.toString();
  if (url.includes('/api/chat')) {
    if (options?.body && JSON.stringify(options.body).includes(MAGIC_WORD)) {
      return new Response(JSON.stringify(MOCK_END_RESPONSE));
    }
    return new Response(JSON.stringify(MOCK_TOOL_CALL_RESPONSE));
  }
  throw new Error('Unknown API endpoint');
};

describe('ollama models', () => {
  const options: OllamaPluginParams = {
    models: [{ name: 'test-model', supports: { tools: true } }],
    serverAddress: 'http://localhost:3000',
  };

  let ai: Genkit;
  beforeEach(() => {
    ai = genkit({
      plugins: [ollama(options)],
    });
  });

  it('should successfully return tool call response', async () => {
    const get_current_weather = ai.defineTool(
      {
        name: 'get_current_weather',
        description: 'gets weather',
        inputSchema: z.object({ format: z.string(), location: z.string() }),
      },
      async () => {
        return MAGIC_WORD;
      }
    );

    const result = await ai.generate({
      model: 'ollama/test-model',
      prompt: 'Hello',
      tools: [get_current_weather],
    });
    assert.ok(result.text === 'The weather is sunny');
  });

  it('should throw for primitive tools', async () => {
    const get_current_weather = ai.defineTool(
      {
        name: 'get_current_weather',
        description: 'gets weather',
        inputSchema: z.object({ format: z.string(), location: z.string() }),
      },
      async () => {
        return MAGIC_WORD;
      }
    );
    const fooz = ai.defineTool(
      {
        name: 'fooz',
        description: 'gets fooz',
        inputSchema: z.string(),
      },
      async () => {
        return 1;
      }
    );

    await assert.rejects(async () => {
      await ai.generate({
        model: 'ollama/test-model',
        prompt: 'Hello',
        tools: [get_current_weather, fooz],
      });
    });
  });

  describe('assistant text with tool_calls', () => {
    const originalFetch = global.fetch;

    afterEach(() => {
      global.fetch = originalFetch;
    });

    function defineWeatherTool() {
      return ai.defineTool(
        {
          name: 'weather',
          description: 'w',
          inputSchema: z.object({ city: z.string() }),
          outputSchema: z.string(),
        },
        async () => 'sunny'
      );
    }

    function chatResponse(message: Record<string, unknown>) {
      return new Response(
        JSON.stringify({ model: 'test-model', message, done: true })
      );
    }

    it('keeps assistant text next to toolRequest when parsing a response', async () => {
      global.fetch = async () =>
        chatResponse({
          role: 'assistant',
          content: 'Let me check the weather.',
          tool_calls: [
            { function: { name: 'weather', arguments: { city: 'Paris' } } },
          ],
        });

      const result = await ai.generate({
        model: 'ollama/test-model',
        prompt: 'weather?',
        tools: [defineWeatherTool()],
        returnToolRequests: true,
      });

      const textPart = result.message.content.find((p) => p.text);
      const toolPart = result.message.content.find((p) => p.toolRequest);
      assert.strictEqual(textPart?.text, 'Let me check the weather.');
      assert.strictEqual(toolPart?.toolRequest?.name, 'weather');
      assert.deepStrictEqual(toolPart?.toolRequest?.input, { city: 'Paris' });
    });

    it('replays assistant text together with tool_calls on the next request', async () => {
      const sent: Array<Record<string, any>> = [];
      global.fetch = async (_input, init) => {
        sent.push(JSON.parse(init!.body as string));
        return chatResponse({ role: 'assistant', content: 'Sunny.' });
      };

      await ai.generate({
        model: 'ollama/test-model',
        messages: [
          { role: 'user', content: [{ text: 'weather?' }] },
          {
            role: 'model',
            content: [
              { text: 'Let me check the weather.' },
              { toolRequest: { name: 'weather', input: { city: 'Paris' } } },
            ],
          },
        ],
      });

      const assistant = sent[0].messages.find(
        (m: { role: string }) => m.role === 'assistant'
      );
      assert.strictEqual(assistant.content, 'Let me check the weather.');
      assert.deepStrictEqual(assistant.tool_calls, [
        { function: { name: 'weather', arguments: { city: 'Paris' } } },
      ]);
    });

    it('keeps assistant text next to toolRequest in streaming chunks', async () => {
      global.fetch = async () =>
        chatResponse({
          role: 'assistant',
          content: 'Let me check the weather.',
          tool_calls: [
            { function: { name: 'weather', arguments: { city: 'Paris' } } },
          ],
        });

      const chunks: Array<{ text?: string; toolRequest?: { name?: string } }> =
        [];
      const { stream } = await ai.generateStream({
        model: 'ollama/test-model',
        prompt: 'weather?',
        tools: [defineWeatherTool()],
        returnToolRequests: true,
      });
      for await (const chunk of stream) {
        chunks.push(...chunk.content);
      }

      assert.ok(
        chunks.some((p) => p.text === 'Let me check the weather.'),
        'stream chunks should include the assistant text'
      );
      assert.ok(
        chunks.some((p) => p.toolRequest?.name === 'weather'),
        'stream chunks should include the tool request'
      );
    });
  });
});
