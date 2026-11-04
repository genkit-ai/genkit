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

import { describe, expect, it } from '@jest/globals';
import { genkit } from 'genkit';
import type { PluginOptions } from '../src';
import { DEEPSEEK_BASE_URL, deepSeek } from '../src/deepseek';
import { XAI_BASE_URL, xAI } from '../src/xai';

type PluginFetch = NonNullable<PluginOptions['fetch']>;
type PluginFetchResponse = Awaited<ReturnType<PluginFetch>>;

const asPluginFetchResponse = (res: Response): PluginFetchResponse =>
  res as unknown as PluginFetchResponse;

describe('request-scoped clients keep the provider base URL', () => {
  function recordingFetch(seen: string[]): PluginFetch {
    return async (url) => {
      seen.push(String(url));
      return asPluginFetchResponse(
        new Response(
          JSON.stringify({
            id: 'test',
            object: 'chat.completion',
            created: 0,
            model: 'test-model',
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: { role: 'assistant', content: 'ok' },
              },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          }),
          { status: 200, headers: { 'content-type': 'application/json' } }
        )
      );
    };
  }

  const cases = [
    {
      provider: 'deepseek',
      baseURL: DEEPSEEK_BASE_URL,
      model: 'deepseek/deepseek-chat',
      plugin: (fetch: PluginFetch) => deepSeek({ apiKey: 'plugin-key', fetch }),
    },
    {
      provider: 'xai',
      baseURL: XAI_BASE_URL,
      model: 'xai/grok-3',
      plugin: (fetch: PluginFetch) => xAI({ apiKey: 'plugin-key', fetch }),
    },
  ];

  for (const { provider, baseURL, model, plugin } of cases) {
    it(`${provider}: a per-call apiKey does not redirect the request to OpenAI`, async () => {
      const seen: string[] = [];
      const ai = genkit({ plugins: [plugin(recordingFetch(seen))] });

      await ai.generate({
        model,
        prompt: 'hi',
        config: { apiKey: 'scoped-key' },
      });

      expect(seen).toHaveLength(1);
      expect(seen[0]).toContain(new URL(baseURL).host);
      expect(seen[0]).not.toContain('api.openai.com');
    });

    it(`${provider}: without a per-call apiKey the default client is used`, async () => {
      const seen: string[] = [];
      const ai = genkit({ plugins: [plugin(recordingFetch(seen))] });

      await ai.generate({ model, prompt: 'hi' });

      expect(seen).toHaveLength(1);
      expect(seen[0]).toContain(new URL(baseURL).host);
    });
  }
});
