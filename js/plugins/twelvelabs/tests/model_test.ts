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

// Mock fetch to simulate the TwelveLabs /analyze endpoint (stream: false).
global.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === 'string' ? input : input.toString();
  if (url.includes('/analyze')) {
    lastBody = JSON.parse(init!.body as string);
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
