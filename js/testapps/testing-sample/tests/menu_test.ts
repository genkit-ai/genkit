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
import assert from 'node:assert/strict';
import { beforeEach, describe, it } from 'node:test';
import {
  ai,
  confirmBooking,
  recommendDish,
  streamRecommendation,
} from '../src/menu.js';

// Register ONE mock under the app's default model name ('menuModel'), once for
// this whole file — the app's `ai.generate` / prompt resolves to it with no
// code change. Test runners (`node --test`, Jest, Vitest) run each test file
// in its own process/module graph, so this registration can't collide with
// other test files. Within the file, `reset()` in beforeEach clears recorded
// history and re-arms the construction respond, and each test scripts its own
// behavior with `respondWith(...)`.
const model = mockModel(ai, {
  name: 'menuModel',
  info: { supports: { tools: true } },
});

beforeEach(() => model.reset());

// Structured output is the highest-value case: the model returns JSON, and the
// flow's *own* logic (validation, deriving `withinBudget`) is what we pin down.
describe('recommendDish flow — structured output + business logic', () => {
  // A fixed, deterministic structured response from the "model".
  const recommendation = {
    text: JSON.stringify({
      dish: 'Mushroom risotto',
      reason: 'Comforting and in season.',
      priceUSD: 18,
    }),
  };

  it('parses the structured recommendation and marks it within budget', async () => {
    model.respondWith(recommendation);

    const out = await recommendDish({
      restaurant: 'Lumen',
      mood: 'cozy',
      budgetUSD: 30,
    });

    assert.equal(out.dish, 'Mushroom risotto');
    assert.equal(out.withinBudget, true);
    assert.equal(model.requestCount, 1);
  });

  it('marks the SAME model output over budget when the budget is lower', async () => {
    // Identical model response as above — only the flow's input changes. This
    // proves the test exercises *our* budget logic, not the model.
    model.respondWith(recommendation);

    const out = await recommendDish({
      restaurant: 'Lumen',
      mood: 'cozy',
      budgetUSD: 15,
    });

    assert.equal(out.withinBudget, false);
  });

  it('inspects prompt assembly on the structured path via lastRequestText', async () => {
    // echoModel can't be used here — the prompt requests structured output, and
    // echo returns text. Inspect the recorded request instead: `lastRequestText`
    // flattens the whole assembled conversation (system + rendered template).
    model.respondWith(recommendation);

    await recommendDish({ restaurant: 'Lumen', mood: 'cozy', budgetUSD: 30 });

    assert.match(
      model.lastRequestText!,
      /You are a concise restaurant concierge/
    );
    assert.match(
      model.lastRequestText!,
      /Recommend a dish at Lumen for someone feeling cozy/
    );
  });

  it('rejects a recommendation the flow considers invalid', async () => {
    // The model returns a structurally-valid but business-invalid price; the
    // flow's guard, not the framework, is what throws.
    model.respondWith({
      text: JSON.stringify({
        dish: 'Free water',
        reason: 'Out of stock on everything else.',
        priceUSD: 0,
      }),
    });

    await assert.rejects(
      recommendDish({ restaurant: 'Lumen', mood: 'broke', budgetUSD: 30 }),
      /non-positive price/
    );
  });
});

describe('recommendDish flow — tool round-trip', () => {
  it('runs dailySpecial, then returns the structured recommendation', async () => {
    model.respondWith((req) => {
      const toolAnswered = req.messages.some((m) =>
        m.content.some((c) => c.toolResponse)
      );
      // First turn: ask for the special. Second turn (after the tool ran):
      // return the structured recommendation.
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

    assert.equal(out.dish, 'Mushroom risotto');
    assert.equal(out.withinBudget, true);
    // Two turns: the tool request, then the follow-up with the tool result.
    assert.equal(model.requestCount, 2);
    // The tool's output was fed back to the model.
    assert.equal(model.toolResponses[0]?.name, 'dailySpecial');
    assert.match(String(model.toolResponses[0]?.output), /mushroom risotto/);
  });
});

describe('streamRecommendation flow — streaming through the flow', () => {
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

    assert.deepEqual(chunks, ['Try ', 'the ', 'risotto.']);
    assert.equal(await output, 'Try the risotto.');
  });
});

describe('scripting responses with a queue', () => {
  it('scripts a two-turn tool interaction with an array of responses', async () => {
    // Turn 1: ask for the special. Turn 2: return the recommendation. The
    // array is consumed one item per call, so no branching callback is needed.
    model.respondWith([
      {
        toolRequests: [
          { name: 'dailySpecial', input: { restaurant: 'Lumen' } },
        ],
      },
      {
        text: JSON.stringify({
          dish: 'Mushroom risotto',
          reason: "It's the daily special.",
          priceUSD: 22,
        }),
      },
    ]);

    const out = await recommendDish({
      restaurant: 'Lumen',
      mood: 'curious',
      budgetUSD: 40,
    });

    assert.equal(out.dish, 'Mushroom risotto');
    assert.equal(model.requestCount, 2);
    assert.equal(model.toolResponses[0]?.name, 'dailySpecial');
  });
});

describe('model failure handling', () => {
  it('surfaces a model error injected via a queued Error', async () => {
    // A queued Error is thrown when reached — here on the very first call — so
    // you can test how a flow behaves when the model fails.
    model.respondWith([new Error('model overloaded')]);

    await assert.rejects(
      recommendDish({ restaurant: 'Lumen', mood: 'cozy', budgetUSD: 30 }),
      /model overloaded/
    );
  });
});

describe('human-in-the-loop with interrupts', () => {
  it('pauses on confirmBooking, then resumes to complete the booking', async () => {
    model.respondWith([
      {
        toolRequests: [
          { name: 'confirmBooking', input: { dish: 'Mushroom risotto' } },
        ],
      },
      { text: 'Enjoy your meal!' },
    ]);

    // First pass: the tool interrupts, so generation pauses awaiting the human.
    const paused = await ai.generate({
      prompt: 'Book the risotto.',
      tools: [confirmBooking],
    });
    assert.equal(paused.interrupts.length, 1);
    assert.equal(paused.interrupts[0].toolRequest.name, 'confirmBooking');

    // The human confirms; `restart` re-runs the tool with the decision (so its
    // `resumed` branch books the dish), then the model finishes.
    const done = await ai.generate({
      messages: paused.messages,
      tools: [confirmBooking],
      resume: {
        restart: confirmBooking.restart(paused.interrupts[0], {
          confirmed: true,
        }),
      },
    });

    assert.equal(done.text, 'Enjoy your meal!');
    assert.equal(model.requestCount, 2);
    assert.match(
      String(model.toolResponses[0]?.output),
      /Booked: Mushroom risotto/
    );
  });
});
