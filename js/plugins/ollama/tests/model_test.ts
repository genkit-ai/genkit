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
import type { OllamaPluginParams } from '../src/types.js';

const BASE_TIME = new Date('2024-07-22T20:33:28.123648Z').getTime();

const MOCK_TOOL_CALL_RESPONSE = {
  model: 'llama3.2',
  created_at: new Date(BASE_TIME).toISOString(),
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
  created_at: new Date(BASE_TIME).toISOString(),
  message: {
    role: 'assistant',
    content: 'The weather is sunny',
  },
  done_reason: 'stop',
  done: true,
};

const MOCK_NO_TOOLS_END_RESPONSE = {
  model: 'llama3.2',
  created_at: new Date(BASE_TIME).toISOString(),
  message: {
    role: 'assistant',
    content: 'I have no way of knowing that',
  },
  done_reason: 'stop',
  done: true,
};

// MockModel class to simulate the tool calling flow more clearly
class MockModel {
  private callCount = 0;
  private hasTools = false;

  // for non-streaming requests
  async chat(request: any): Promise<any> {
    this.callCount++;

    // First call: initial request with tools → return tool call
    if (this.callCount === 1 && request.tools && request.tools.length > 0) {
      this.hasTools = true;
      return MOCK_TOOL_CALL_RESPONSE;
    }

    // Second call: follow-up with tool results → return final answer
    if (
      this.callCount === 2 &&
      this.hasTools &&
      request.messages?.some((m: any) => m.role === 'tool')
    ) {
      return MOCK_END_RESPONSE;
    }

    // Basic request without tools → return end response
    return MOCK_NO_TOOLS_END_RESPONSE;
  }

  // Create a streaming response for testing using a ReadableStream
  createStreamingResponse(): ReadableStream<Uint8Array> {
    const words = ['this', 'is', 'a', 'streaming', 'response'];

    return new ReadableStream({
      start(controller) {
        let wordIndex = 0;

        const sendNextChunk = () => {
          if (wordIndex >= words.length) {
            controller.close();
            return;
          }

          // Stream individual words (not cumulative)
          const currentWord = words[wordIndex];
          const isLastChunk = wordIndex === words.length - 1;

          // Increment timestamp for each chunk
          const chunkTime = new Date(BASE_TIME + wordIndex * 100).toISOString();

          const response = {
            model: 'llama3.2',
            created_at: chunkTime,
            message: {
              role: 'assistant',
              content: currentWord + (isLastChunk ? '' : ' '), // Add space except for last word
            },
            done_reason: isLastChunk ? 'stop' : undefined,
            done: isLastChunk,
          };

          controller.enqueue(
            new TextEncoder().encode(JSON.stringify(response) + '\n')
          );

          wordIndex++;
          setTimeout(sendNextChunk, 10); // Small delay to simulate streaming
        };

        sendNextChunk();
      },
    });
  }

  reset(): void {
    this.callCount = 0;
    this.hasTools = false;
  }
}

// Create a mock model instance to simulate the tool calling flow
const mockModel = new MockModel();

// Mock fetch to simulate the multi-turn tool calling flow using MockModel
global.fetch = async (input: RequestInfo | URL, options?: RequestInit) => {
  const url = typeof input === 'string' ? input : input.toString();
  if (url.includes('/api/chat')) {
    const body = JSON.parse((options?.body as string) || '{}');

    // Check if this is a streaming request
    if (body.stream) {
      const stream = mockModel.createStreamingResponse();
      return new Response(stream, {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Non-streaming request
    const response = await mockModel.chat(body);
    return new Response(JSON.stringify(response));
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
    mockModel.reset(); // Reset mock state between tests
    ai = genkit({
      plugins: [ollama(options)],
    });
  });

  it('should successfully return basic response', async () => {
    const result = await ai.generate({
      model: 'ollama/test-model',
      prompt: 'Hello',
    });
    assert.ok(result.text === 'I have no way of knowing that');
  });

  it('should successfully return tool call response', async () => {
    const get_current_weather = ai.defineTool(
      {
        name: 'get_current_weather',
        description: 'gets weather',
        inputSchema: z.object({ format: z.string(), location: z.string() }),
      },
      async () => {
        return 'sunny';
      }
    );

    const result = await ai.generate({
      model: 'ollama/test-model',
      prompt: 'Hello',
      tools: [get_current_weather],
    });
    assert.ok(result.text === 'The weather is sunny');
  });

  it('should throw for tools with primitive (non-object) input schema.', async () => {
    // This tool will throw an error because it has a primitive (non-object) input schema.
    const toolWithNonObjectInput = ai.defineTool(
      {
        name: 'toolWithNonObjectInput',
        description: 'tool with non-object input schema',
        inputSchema: z.string(),
      },
      async () => {
        return 'anything';
      }
    );

    try {
      await ai.generate({
        model: 'ollama/test-model',
        prompt: 'Hello',
        tools: [toolWithNonObjectInput],
      });
    } catch (error) {
      assert.ok(error instanceof Error);

      assert.ok(
        error.message.includes('Ollama only supports tools with object inputs')
      );
    }
  });

  it('should successfully return streaming response', async () => {
    const streamingResult = ai.generateStream({
      model: 'ollama/test-model',
      prompt: 'Hello',
    });

    let fullText = '';
    let chunkCount = 0;
    for await (const chunk of streamingResult.stream) {
      fullText += chunk.text; // Each chunk contains individual words
      chunkCount++;
    }

    // Should have received multiple chunks (one per word)
    assert.ok(chunkCount > 1);
    // Final text should be complete
    assert.ok(fullText === 'this is a streaming response');
  });
});
