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
import { genkit, type Genkit } from 'genkit';
import { beforeEach, describe, it } from 'node:test';
import { defineTwelveLabsModel } from '../src/models.js';

let lastBody: any;

// When set, the next /analyze call returns this string as the SSE response
// body instead of a JSON (stream: false) response.
let nextSseBody: string | undefined;

function sseStream(text: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(text));
      controller.close();
    },
  });
}

// Mock fetch to simulate the TwelveLabs /analyze endpoint.
global.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === 'string' ? input : input.toString();
  if (url.includes('/analyze')) {
    lastBody = JSON.parse(init!.body as string);
    if (nextSseBody !== undefined) {
      const body = sseStream(nextSseBody);
      nextSseBody = undefined;
      return { ok: true, body } as unknown as Response;
    }
    return {
      ok: true,
      body: {},
      json: async () => ({
        id: 'abc',
        data: 'A short clip of a tree.',
        finish_reason: 'stop',
        usage: { output_tokens: 7 },
      }),
    } as unknown as Response;
  }
  throw new Error('Unknown API endpoint');
}) as typeof fetch;

describe('defineTwelveLabsModel', () => {
  let ai: Genkit;
  beforeEach(() => {
    ai = genkit({ plugins: [] });
  });

  it('sends the video URL + prompt and returns the generated text', async () => {
    const model = defineTwelveLabsModel(ai, {
      apiKey: 'test-key',
      baseUrl: 'https://api.twelvelabs.io/v1.3',
      model: { name: 'pegasus1.5' },
    });
    const { text } = await ai.generate({
      model,
      messages: [
        {
          role: 'user',
          content: [
            { text: 'Describe this video.' },
            {
              media: {
                url: 'https://example.com/v.mp4',
                contentType: 'video/mp4',
              },
            },
          ],
        },
      ],
    });
    assert.strictEqual(text, 'A short clip of a tree.');
    assert.deepStrictEqual(lastBody.video, {
      type: 'url',
      url: 'https://example.com/v.mp4',
    });
    assert.strictEqual(lastBody.prompt, 'Describe this video.');
    assert.strictEqual(lastBody.model_name, 'pegasus1.5');
  });

  it('parses SSE-framed streaming events and recomputes totalTokens', async () => {
    // Real SSE framing: `data:` prefix, blank separator lines, a keepalive
    // comment, and a terminal [DONE] sentinel.
    nextSseBody = [
      ': keepalive',
      'data: {"event_type":"text_generation","text":"A short "}',
      '',
      'data: {"event_type":"text_generation","text":"clip."}',
      '',
      'data: {"event_type":"stream_end","metadata":{"usage":{"output_tokens":7}}}',
      '',
      'data: [DONE]',
      '',
    ].join('\n');

    const model = defineTwelveLabsModel(ai, {
      apiKey: 'test-key',
      baseUrl: 'https://api.twelvelabs.io/v1.3',
      model: { name: 'pegasus1.5' },
    });

    const chunks: string[] = [];
    const { text, usage } = await ai.generate({
      model,
      messages: [
        {
          role: 'user',
          content: [
            { text: 'Describe this video.' },
            {
              media: {
                url: 'https://example.com/v.mp4',
                contentType: 'video/mp4',
              },
            },
          ],
        },
      ],
      onChunk: (c) => chunks.push(c.text),
    });

    assert.strictEqual(lastBody.stream, true);
    assert.strictEqual(text, 'A short clip.');
    assert.deepStrictEqual(chunks, ['A short ', 'clip.']);
    assert.strictEqual(usage?.outputTokens, 7);
    assert.strictEqual(
      usage?.totalTokens,
      (usage?.inputTokens ?? 0) + 7
    );
  });

  it('errors when no video is provided', async () => {
    const model = defineTwelveLabsModel(ai, {
      apiKey: 'test-key',
      baseUrl: 'https://api.twelvelabs.io/v1.3',
      model: { name: 'pegasus1.5' },
    });
    await assert.rejects(
      () => ai.generate({ model, prompt: 'Describe this video.' }),
      /requires a video/
    );
  });
});
