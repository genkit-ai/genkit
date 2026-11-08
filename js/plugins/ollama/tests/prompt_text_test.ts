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

describe('Ollama message text', () => {
  const cases = [
    {
      name: 'single text part',
      parts: [{ text: 'Hello, world!' }],
      expected: 'Hello, world!',
    },
    {
      name: 'multiple text parts',
      parts: [{ text: 'Hello' }, { text: ', ' }, { text: 'world!' }],
      expected: 'Hello, world!',
    },
    {
      name: 'empty parts and caller-provided commas',
      parts: [
        { text: '' },
        { text: ',Hello' },
        { text: '' },
        { text: ', world!' },
      ],
      expected: ',Hello, world!',
    },
  ];

  for (const type of ['chat', 'generate'] as const) {
    for (const { name, parts, expected } of cases) {
      it(`preserves ${name} for ${type}`, async (t) => {
        let request:
          | {
              prompt?: string;
              system?: string;
              messages?: { role: string; content: string }[];
            }
          | undefined;
        t.mock.method(globalThis, 'fetch', async (url, init) => {
          assert.strictEqual(url, `http://stub/api/${type}`);
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

        await ai.generate({
          model: 'ollama/test-model',
          system: parts,
          prompt: parts,
        });

        assert.ok(request);
        if (type === 'generate') {
          assert.deepStrictEqual(
            { prompt: request.prompt, system: request.system },
            { prompt: expected, system: expected }
          );
        } else {
          assert.deepStrictEqual(request.messages, [
            { role: 'system', content: expected },
            { role: 'user', content: expected },
          ]);
        }
      });
    }
  }
});
