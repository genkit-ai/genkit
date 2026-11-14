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
import { genkit } from 'genkit';
import { describe, it } from 'node:test';
import { ollama } from '../src/index.js';

describe('Ollama stop sequences', () => {
  for (const type of ['chat', 'generate'] as const) {
    for (const stopSequences of [
      undefined,
      [],
      ['<END>'],
      ['<END>', '<USER>'],
    ]) {
      it(`preserves ${JSON.stringify(stopSequences)} for ${type}`, async (t) => {
        let request: { options: { stop?: string[] } } | undefined;
        t.mock.method(globalThis, 'fetch', async (url, init) => {
          assert.strictEqual(url, `http://stub/api/${type}`);
          assert.ok(init?.body, 'Ollama request should include a body');
          request = JSON.parse(init.body);
          return new Response(
            JSON.stringify({
              model: 'test-model',
              message: { role: 'assistant', content: 'done' },
              response: 'done',
              done: true,
            })
          );
        });
        const ai = genkit({
          plugins: [
            ollama({
              serverAddress: 'http://stub',
              models: [{ name: 'test-model', type }],
            }),
          ],
        });

        const result = await ai.generate({
          model: 'ollama/test-model',
          prompt: 'hello',
          config: { stopSequences },
        });

        assert.strictEqual(result.text, 'done');
        assert.ok(request);
        assert.deepStrictEqual(request.options.stop, stopSequences);
        assert.strictEqual(
          'stop' in request.options,
          stopSequences !== undefined
        );
      });
    }
  }
});
