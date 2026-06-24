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
import { genkit } from 'genkit';
import { describe, it } from 'node:test';
import { twelvelabs } from '../src/index.js';

// Live test: requires a real TWELVELABS_API_KEY. Skipped when unset.
// Run with: TWELVELABS_API_KEY=... pnpm test:live
const hasKey = !!process.env.TWELVELABS_API_KEY;

describe('twelvelabs live', { skip: !hasKey }, () => {
  it('embeds text via Marengo and returns a 512-dim vector', async () => {
    const ai = genkit({
      plugins: [
        twelvelabs({ embedders: [{ name: 'marengo3.0', dimensions: 512 }] }),
      ],
    });
    const result = await ai.embed({
      embedder: 'twelvelabs/marengo3.0',
      content: 'a cat',
    });
    assert.strictEqual(result.length, 1);
    assert.strictEqual(result[0].embedding.length, 512);
  });
});
