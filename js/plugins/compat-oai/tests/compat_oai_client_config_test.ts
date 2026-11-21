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

import { describe, expect, it, jest } from '@jest/globals';
import OpenAI from 'openai';
import { defineCompatOpenAIEmbedder } from '../src/embedder';
import { defineCompatOpenAIImageModel } from '../src/image';

describe('request-scoped client config', () => {
  it('does not include apiKey in image request bodies', async () => {
    const fetch = jest.fn(async (_url: any, init?: RequestInit) => {
      return new globalThis.Response(
        JSON.stringify({ data: [{ b64_json: 'AA==' }] }),
        { headers: { 'content-type': 'application/json' } }
      );
    });
    const client = new OpenAI({ apiKey: 'default-key', fetch: fetch as any });
    const imageModel = defineCompatOpenAIImageModel({
      name: 'dall-e-3',
      client,
      pluginOptions: { name: 'openai', fetch: fetch as any },
    });

    await imageModel.run({
      messages: [{ role: 'user', content: [{ text: 'a cat' }] }],
      config: { apiKey: 'request-key' },
    });

    const body = JSON.parse(String(fetch.mock.calls[0][1]?.body));
    expect(body).not.toHaveProperty('apiKey');
  });

  it('does not include apiKey in embedding request bodies', async () => {
    const fetch = jest.fn(async (_url: any, init?: RequestInit) => {
      return new globalThis.Response(
        JSON.stringify({ data: [{ embedding: [0], index: 0 }] }),
        { headers: { 'content-type': 'application/json' } }
      );
    });
    const client = new OpenAI({ apiKey: 'default-key', fetch: fetch as any });
    const embedder = defineCompatOpenAIEmbedder({
      name: 'text-embedding-3-small',
      client,
      pluginOptions: { name: 'openai', fetch: fetch as any },
    });

    await embedder.run({
      input: [{ content: [{ text: 'hello' }] }],
      options: { apiKey: 'request-key' },
    });

    const body = JSON.parse(String(fetch.mock.calls[0][1]?.body));
    expect(body).not.toHaveProperty('apiKey');
  });
});
