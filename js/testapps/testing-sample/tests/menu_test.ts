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

import { echoModel, mockModel } from 'genkit/testing';
import assert from 'node:assert/strict';
import { beforeEach, describe, it } from 'node:test';
import { createMenuApp } from '../src/menu.js';

// Build a fresh app (its own Genkit registry) for each test, then register the
// mock under the app's default model name ('menuModel') so the flow's
// `ai.generate({ prompt, tools })` resolves to it with no code change. The
// fresh instance keeps tests isolated and avoids re-registering the same model
// name on a shared registry.
let ai: ReturnType<typeof createMenuApp>['ai'];
let recommendDish: ReturnType<typeof createMenuApp>['recommendDish'];
beforeEach(() => {
  ({ ai, recommendDish } = createMenuApp());
});

describe('recommendDish flow', () => {
  it('returns the model recommendation', async () => {
    const model = mockModel(ai, {
      name: 'menuModel',
      respond: () => ({ text: 'Try the risotto.' }),
    });

    const out = await recommendDish({ restaurant: 'Lumen', mood: 'cozy' });

    assert.equal(out, 'Try the risotto.');
    // The flow rendered our inputs into the prompt the model saw.
    // `lastRequestMessage` is a genkit Message, so `.text` works just like it
    // does on a response.
    assert.match(
      model.lastRequestMessage!.text,
      /Recommend a dish at Lumen for someone feeling cozy/
    );
  });

  it('exposes the dailySpecial tool to the model', async () => {
    const model = mockModel(ai, {
      name: 'menuModel',
      info: { supports: { tools: true } },
      respond: () => ({ text: 'ok' }),
    });

    await recommendDish({ restaurant: 'Lumen', mood: 'hungry' });

    const toolNames = (model.lastRequest!.tools ?? []).map((t) => t.name);
    assert.ok(toolNames.includes('dailySpecial'));
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

    assert.equal(out, 'Go for the mushroom risotto.');
    // Two turns: the tool request, then the follow-up with the tool result.
    assert.equal(model.requestCount, 2);
    const toolResult = model.lastRequest!.messages
      .flatMap((m) => m.content)
      .find((c) => c.toolResponse);
    assert.match(String(toolResult?.toolResponse?.output), /mushroom risotto/);
  });
});

describe('prompt assembly with echoModel', () => {
  it('shows what the model would have seen', async () => {
    const model = echoModel(ai, { name: 'menuModel' });

    const out = await recommendDish({ restaurant: 'Lumen', mood: 'tired' });

    assert.match(out, /Recommend a dish at Lumen for someone feeling tired/);
    assert.ok(model.requestCount >= 1);
  });
});

describe('streaming', () => {
  it('streams chunks through generateStream', async () => {
    const model = mockModel(ai, {
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

    assert.deepEqual(chunks, ['Try ', 'the ', 'risotto.']);
    assert.equal((await response).text, 'Try the risotto.');
    assert.equal(model.requestCount, 1);
  });
});
