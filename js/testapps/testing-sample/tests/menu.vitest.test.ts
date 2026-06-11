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
import { beforeEach, describe, expect, it } from 'vitest';
import { createMenuApp } from '../src/menu.js';

// The same patterns as menu_test.ts, written with vitest instead of node:test —
// `genkit/testing` is runner-agnostic. Build a fresh app per test, then register
// the mock under the app's default model name ('menuModel') so the app runs
// unchanged.
type App = ReturnType<typeof createMenuApp>;
let ai: App['ai'];
let recommendDish: App['recommendDish'];
let recommendPrompt: App['recommendPrompt'];
let streamRecommendation: App['streamRecommendation'];
beforeEach(() => {
  ({ ai, recommendDish, recommendPrompt, streamRecommendation } =
    createMenuApp());
});

describe('recommendDish flow — structured output + business logic (vitest)', () => {
  const respondWithRecommendation = () => ({
    text: JSON.stringify({
      dish: 'Mushroom risotto',
      reason: 'Comforting and in season.',
      priceUSD: 18,
    }),
  });

  it('parses the structured recommendation and applies the budget logic', async () => {
    mockModel(ai, { name: 'menuModel', respond: respondWithRecommendation });

    // Same model output, two budgets — only our logic decides `withinBudget`.
    const ok = await recommendDish({
      restaurant: 'Lumen',
      mood: 'cozy',
      budgetUSD: 30,
    });
    expect(ok.dish).toBe('Mushroom risotto');
    expect(ok.withinBudget).toBe(true);
  });

  it('rejects a recommendation the flow considers invalid', async () => {
    mockModel(ai, {
      name: 'menuModel',
      respond: () => ({
        text: JSON.stringify({
          dish: 'Free water',
          reason: 'Out of stock on everything else.',
          priceUSD: 0,
        }),
      }),
    });

    await expect(
      recommendDish({ restaurant: 'Lumen', mood: 'broke', budgetUSD: 30 })
    ).rejects.toThrow(/non-positive price/);
  });
});

describe('recommendDish flow — tool round-trip (vitest)', () => {
  it('runs dailySpecial, then returns the structured recommendation', async () => {
    const model = mockModel(ai, {
      name: 'menuModel',
      info: { supports: { tools: true } },
      respond: (req) => {
        const toolAnswered = req.messages.some((m) =>
          m.content.some((c) => c.toolResponse)
        );
        return toolAnswered
          ? {
              text: JSON.stringify({
                dish: 'Mushroom risotto',
                reason: "It's the daily special.",
                priceUSD: 22,
              }),
            }
          : {
              toolRequests: [
                { name: 'dailySpecial', input: { restaurant: 'Lumen' } },
              ],
            };
      },
    });

    const out = await recommendDish({
      restaurant: 'Lumen',
      mood: 'curious',
      budgetUSD: 40,
    });

    expect(out.dish).toBe('Mushroom risotto');
    expect(model.requestCount).toBe(2);
  });
});

describe('prompt assembly with echoModel (vitest)', () => {
  it('shows the full rendered request — system + interpolated template', async () => {
    echoModel(ai, { name: 'menuModel', info: { supports: { tools: true } } });

    const res = await recommendPrompt({
      restaurant: 'Lumen',
      mood: 'tired',
      budgetUSD: 40,
    });

    expect(res.text).toMatch(/system: You are a concise restaurant concierge/);
    expect(res.text).toMatch(
      /Recommend a dish at Lumen for someone feeling tired\. Their budget is 40 USD/
    );
  });
});

describe('streamRecommendation flow — streaming through the flow (vitest)', () => {
  it('forwards model chunks out through the flow stream', async () => {
    mockModel(ai, {
      name: 'menuModel',
      respond: (_req, { sendChunk }) => {
        sendChunk('Try ');
        sendChunk('the ');
        sendChunk('risotto.');
        return { text: 'Try the risotto.' };
      },
    });

    const { stream, output } = streamRecommendation.stream({
      restaurant: 'Lumen',
      mood: 'cozy',
    });

    const chunks: string[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks).toEqual(['Try ', 'the ', 'risotto.']);
    expect(await output).toBe('Try the risotto.');
  });
});
