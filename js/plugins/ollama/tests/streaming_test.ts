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
import { genkit, z, type Genkit } from 'genkit';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { ollama } from '../src/index.js';

const encoder = new TextEncoder();

/** One line of the NDJSON stream returned by the Ollama chat API. */
function chatLine(
  message: { content: string; tool_calls?: unknown[] },
  done = false
): string {
  return (
    JSON.stringify({
      model: 'test-model',
      created_at: '2024-07-22T20:33:28.123648Z',
      message: { role: 'assistant', ...message },
      done,
      ...(done ? { done_reason: 'stop' } : {}),
    }) + '\n'
  );
}

/** One line of the NDJSON stream returned by the Ollama generate API. */
function generateLine(response: string, done = false): string {
  return (
    JSON.stringify({
      model: 'test-model',
      created_at: '2024-07-22T20:33:28.123648Z',
      response,
      done,
    }) + '\n'
  );
}

/**
 * Replaces `fetch` with a function that streams the given network chunks as
 * the body of the response, exactly as they are split (a chunk can contain
 * several NDJSON lines, or only part of one).
 */
function mockStreamingFetch(chunks: Array<string | Uint8Array>) {
  globalThis.fetch = async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(
            typeof chunk === 'string' ? encoder.encode(chunk) : chunk
          );
        }
        controller.close();
      },
    });
    return new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'application/x-ndjson' },
    });
  };
}

describe('ollama streaming', () => {
  let originalFetch: typeof globalThis.fetch;
  let ai: Genkit;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    ai = genkit({
      plugins: [
        ollama({
          models: [
            { name: 'test-model', supports: { tools: true } },
            { name: 'generate-model', type: 'generate' },
          ],
          serverAddress: 'http://localhost:3000',
        }),
      ],
    });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  async function streamText(model = 'ollama/test-model') {
    const chunks: string[] = [];
    const response = await ai.generate({
      model,
      prompt: 'Hello',
      onChunk: (chunk) => chunks.push(chunk.text),
    });
    return { chunks, text: response.text };
  }

  it('handles one line per network chunk', async () => {
    mockStreamingFetch([
      chatLine({ content: 'Hello' }),
      chatLine({ content: ' world' }),
      chatLine({ content: '' }, true),
    ]);

    const { chunks, text } = await streamText();

    assert.deepStrictEqual(chunks, ['Hello', ' world', '']);
    assert.strictEqual(text, 'Hello world');
  });

  it('handles several lines in one network chunk', async () => {
    mockStreamingFetch([
      chatLine({ content: 'Hello' }) + chatLine({ content: ' world' }),
      chatLine({ content: '' }, true),
    ]);

    const { chunks, text } = await streamText();

    assert.deepStrictEqual(chunks, ['Hello', ' world', '']);
    assert.strictEqual(text, 'Hello world');
  });

  it('handles all lines in a single network chunk', async () => {
    mockStreamingFetch([
      chatLine({ content: 'Hello' }) +
        chatLine({ content: ' world' }) +
        chatLine({ content: '' }, true),
    ]);

    const { text } = await streamText();

    assert.strictEqual(text, 'Hello world');
  });

  it('handles a line split across network chunks', async () => {
    const line = chatLine({ content: 'Hello world' });
    mockStreamingFetch([
      line.slice(0, 40),
      line.slice(40),
      chatLine({ content: '' }, true),
    ]);

    const { chunks, text } = await streamText();

    assert.deepStrictEqual(chunks, ['Hello world', '']);
    assert.strictEqual(text, 'Hello world');
  });

  it('handles a multi-byte character split across network chunks', async () => {
    const bytes = encoder.encode(chatLine({ content: 'café 世界 😀' }));
    // Cut in the middle of every multi-byte character in turn.
    const cuts: number[] = [];
    for (let i = 1; i < bytes.length; i++) {
      if ((bytes[i] & 0xc0) === 0x80) {
        cuts.push(i);
      }
    }
    assert.ok(cuts.length > 0);
    for (const cut of cuts) {
      mockStreamingFetch([
        bytes.slice(0, cut),
        bytes.slice(cut),
        chatLine({ content: '' }, true),
      ]);

      const { chunks, text } = await streamText();

      assert.deepStrictEqual(chunks, ['café 世界 😀', ''], `cut at ${cut}`);
      assert.strictEqual(text, 'café 世界 😀', `cut at ${cut}`);
    }
  });

  it('handles every line split into single bytes', async () => {
    const bytes = encoder.encode(
      chatLine({ content: 'café' }) + chatLine({ content: '' }, true)
    );
    mockStreamingFetch(Array.from(bytes, (byte) => Uint8Array.of(byte)));

    const { text } = await streamText();

    assert.strictEqual(text, 'café');
  });

  it('handles a last line without a trailing newline', async () => {
    mockStreamingFetch([
      chatLine({ content: 'Hello' }),
      chatLine({ content: ' world' }, true).trimEnd(),
    ]);

    const { text } = await streamText();

    assert.strictEqual(text, 'Hello world');
  });

  it('handles the generate API', async () => {
    mockStreamingFetch([
      generateLine('Hello') + generateLine(' wor'),
      generateLine('ld').slice(0, 30),
      generateLine('ld').slice(30) + generateLine('', true),
    ]);

    const { chunks, text } = await streamText('ollama/generate-model');

    assert.deepStrictEqual(chunks, ['Hello', ' wor', 'ld', '']);
    assert.strictEqual(text, 'Hello world');
  });

  it('returns streamed tool calls in the final message', async () => {
    mockStreamingFetch([
      chatLine({
        content: '',
        tool_calls: [
          {
            function: {
              name: 'get_weather',
              arguments: { city: 'Paris' },
            },
          },
        ],
      }),
      chatLine({ content: '' }, true),
    ]);
    const getWeather = ai.defineTool(
      {
        name: 'get_weather',
        description: 'gets the weather',
        inputSchema: z.object({ city: z.string() }),
      },
      async () => 'sunny'
    );

    const response = await ai.generate({
      model: 'ollama/test-model',
      prompt: 'Weather in Paris?',
      tools: [getWeather],
      returnToolRequests: true,
      onChunk: () => {},
    });

    assert.strictEqual(response.text, '');
    assert.deepStrictEqual(
      response.toolRequests.map(({ toolRequest: { name, input } }) => ({
        name,
        input,
      })),
      [{ name: 'get_weather', input: { city: 'Paris' } }]
    );
  });

  it('keeps the text that precedes a streamed tool call', async () => {
    mockStreamingFetch([
      chatLine({ content: 'Let me check.' }),
      chatLine({
        content: '',
        tool_calls: [
          {
            function: {
              name: 'get_weather',
              arguments: { city: 'Paris' },
            },
          },
        ],
      }),
      chatLine({ content: '' }, true),
    ]);
    const getWeather = ai.defineTool(
      {
        name: 'get_weather',
        description: 'gets the weather',
        inputSchema: z.object({ city: z.string() }),
      },
      async () => 'sunny'
    );

    const response = await ai.generate({
      model: 'ollama/test-model',
      prompt: 'Weather in Paris?',
      tools: [getWeather],
      returnToolRequests: true,
      onChunk: () => {},
    });

    assert.strictEqual(response.text, 'Let me check.');
    assert.deepStrictEqual(
      response.toolRequests.map(({ toolRequest: { name, input } }) => ({
        name,
        input,
      })),
      [{ name: 'get_weather', input: { city: 'Paris' } }]
    );
  });

  it('reports an error line streamed by the server', async () => {
    mockStreamingFetch([
      chatLine({ content: 'Hello' }),
      JSON.stringify({ error: 'model runner crashed' }) + '\n',
    ]);

    await assert.rejects(streamText(), /model runner crashed/);
  });
});
