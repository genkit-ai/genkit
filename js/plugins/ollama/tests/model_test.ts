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
import { beforeEach, describe, it } from 'node:test';
import { ollama } from '../src/index.js';
import type {
  OllamaPluginParams,
  RequestHeaderFunction,
} from '../src/types.js';

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
});

describe('ollama request headers', () => {
  it('passes the model request to the headers function', async () => {
    let seen: Parameters<RequestHeaderFunction>[0] | undefined;
    const ai = genkit({
      plugins: [
        ollama({
          models: [{ name: 'test-model' }],
          serverAddress: 'http://localhost:3000',
          requestHeaders: async (params) => {
            seen = params;
            return { Authorization: 'Bearer token' };
          },
        }),
      ],
    });

    await ai.generate({ model: 'ollama/test-model', prompt: MAGIC_WORD });

    assert.strictEqual(seen?.serverAddress, 'http://localhost:3000');
    assert.strictEqual(seen?.model?.name, 'test-model');
    assert.deepStrictEqual(seen?.modelRequest?.messages[0].content, [
      { text: MAGIC_WORD },
    ]);
  });
});
