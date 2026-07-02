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

import { mockModel } from 'genkit/testing';
import { beforeEach, describe, expect, it } from 'vitest';
import {
  ai,
  confirmBooking,
  recommendDish,
  streamRecommendation,
} from '../src/menu.js';

// The same patterns as menu_test.ts, written with vitest instead of node:test —
// `genkit/testing` is runner-agnostic. One mock is registered under the app's
// default model name for the whole file (vitest isolates each test file in its
// own worker, so this can't collide with other files); each test scripts its
// own behavior with `respondWith(...)` after `reset()` clears shared state.
const model = mockModel(ai, {
  name: 'menuModel',
  info: { supports: { tools: true } },
});

beforeEach(() => model.reset());

describe('recommendDish flow — structured output + business logic (vitest)', () => {
  const recommendation = {
    text: JSON.stringify({
      dish: 'Mushroom risotto',
      reason: 'Comforting and in season.',
      priceUSD: 18,
    }),
  };

  it('parses the structured recommendation and applies the budget logic', async () => {
    model.respondWith(recommendation);

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
    model.respondWith({
      text: JSON.stringify({
        dish: 'Free water',
        reason: 'Out of stock on everything else.',
        priceUSD: 0,
      }),
    });

    await expect(
      recommendDish({ restaurant: 'Lumen', mood: 'broke', budgetUSD: 30 })
    ).rejects.toThrow(/non-positive price/);
  });
});

describe('recommendDish flow — tool round-trip (vitest)', () => {
  it('runs dailySpecial, then returns the structured recommendation', async () => {
    model.respondWith((req) => {
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
    });

    const out = await recommendDish({
      restaurant: 'Lumen',
      mood: 'curious',
      budgetUSD: 40,
    });

    expect(out.dish).toBe('Mushroom risotto');
    expect(model.requestCount).toBe(2);
    expect(model.toolResponses[0]?.name).toBe('dailySpecial');
  });
});

describe('human-in-the-loop with interrupts (vitest)', () => {
  it('pauses on confirmBooking, then resumes to complete the booking', async () => {
    // Queued responses: request the interrupting tool, then answer.
    model.respondWith([
      {
        toolRequests: [
          { name: 'confirmBooking', input: { dish: 'Mushroom risotto' } },
        ],
      },
      { text: 'Enjoy your meal!' },
    ]);

    const paused = await ai.generate({
      prompt: 'Book the risotto.',
      tools: [confirmBooking],
    });
    expect(paused.interrupts.length).toBe(1);
    expect(paused.interrupts[0].toolRequest.name).toBe('confirmBooking');

    const done = await ai.generate({
      messages: paused.messages,
      tools: [confirmBooking],
      resume: {
        restart: confirmBooking.restart(paused.interrupts[0], {
          confirmed: true,
        }),
      },
    });

    expect(done.text).toBe('Enjoy your meal!');
    expect(model.requestCount).toBe(2);
    expect(String(model.toolResponses[0]?.output)).toMatch(
      /Booked: Mushroom risotto/
    );
  });
});

describe('streamRecommendation flow — streaming through the flow (vitest)', () => {
  it('forwards model chunks out through the flow stream', async () => {
    model.respondWith((_req, { sendChunk }) => {
      sendChunk('Try ');
      sendChunk('the ');
      sendChunk('risotto.');
      return { text: 'Try the risotto.' };
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
