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

import { echoModel, mockModel } from 'genkit/testing';
import { describe, expect, it } from 'vitest';
import { ai, recommendDish } from '../src/menu.js';

// The same patterns as menu_test.ts, written with vitest instead of node:test —
// `genkit/testing` is runner-agnostic. Register the mock under the app's default
// model name ('menuModel') so the flow runs unchanged.
describe('recommendDish flow (vitest)', () => {
  it('returns the model recommendation', async () => {
    const model = mockModel(ai, {
      name: 'menuModel',
      respond: () => ({ text: 'Try the risotto.' }),
    });

    const out = await recommendDish({ restaurant: 'Lumen', mood: 'cozy' });

    expect(out).toBe('Try the risotto.');
    expect(model.lastMessage!.text).toMatch(
      /Recommend a dish at Lumen for someone feeling cozy/
    );
  });

  it('runs the tool the model calls, then returns the final text', async () => {
    const model = mockModel(ai, {
      name: 'menuModel',
      info: { supports: { tools: true } },
      respond: (req) => {
        const toolAnswered = req.messages.some((m) =>
          m.content.some((c) => c.toolResponse)
        );
        return toolAnswered
          ? { text: 'Go for the mushroom risotto.' }
          : {
              toolRequests: [
                { name: 'dailySpecial', input: { restaurant: 'Lumen' } },
              ],
            };
      },
    });

    const out = await recommendDish({ restaurant: 'Lumen', mood: 'curious' });

    expect(out).toBe('Go for the mushroom risotto.');
    expect(model.requestCount).toBe(2);
  });
});

describe('streaming (vitest)', () => {
  it('streams chunks through generateStream', async () => {
    mockModel(ai, {
      name: 'menuModel',
      respond: (_req, { sendChunk }) => {
        sendChunk('Try ');
        sendChunk('the ');
        sendChunk('risotto.');
        return { text: 'Try the risotto.' };
      },
    });

    const { response, stream } = ai.generateStream({ prompt: 'recommend' });
    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk.text);
    }

    expect(chunks).toEqual(['Try ', 'the ', 'risotto.']);
    expect((await response).text).toBe('Try the risotto.');
  });
});

describe('prompt assembly with echoModel (vitest)', () => {
  it('shows what the model would have seen', async () => {
    echoModel(ai, { name: 'menuModel' });

    const out = await recommendDish({ restaurant: 'Lumen', mood: 'tired' });

    expect(out).toMatch(/Recommend a dish at Lumen for someone feeling tired/);
  });
});
