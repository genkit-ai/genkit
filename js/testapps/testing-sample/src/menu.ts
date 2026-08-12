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

// `genkit/beta` is a superset of `genkit` — used here because the
// human-in-the-loop `confirmBooking` tool relies on interrupts, a beta feature.
// Everything else in this app works identically on the stable `genkit` entry.
import { z } from 'genkit';
import { genkit } from 'genkit/beta';

/** What we ask the model to produce: a structured recommendation. */
export const RecommendationSchema = z.object({
  dish: z.string(),
  reason: z.string(),
  priceUSD: z.number(),
});

/** What the flow returns to its caller, after applying our own logic. */
export const DishResultSchema = z.object({
  dish: z.string(),
  reason: z.string(),
  /** Derived by the flow, not by the model. */
  withinBudget: z.boolean(),
});

/** Flow input. */
export const RecommendInputSchema = z.object({
  restaurant: z.string(),
  mood: z.string(),
  budgetUSD: z.number(),
});

/**
 * The app's Genkit instance, a plain module-level singleton — the standard
 * Genkit app structure. The default model is referenced by name only: in
 * production you'd register it from a provider plugin (e.g. googleAI), and in
 * tests you register a mock under the same name. Either way the app code below
 * is unchanged. Test runners (`node --test`, Jest, Vitest) isolate each test
 * file in its own process/module graph, so every test file gets a fresh
 * registry for free; within a file, tests share one mock and use
 * `reset()` / `respondWith()` (see the tests).
 */
export const ai = genkit({ model: 'menuModel' });

/**
 * A tool the model can call to look up today's special for a restaurant.
 * In a real app this would hit a database; here it is deterministic.
 */
export const dailySpecial = ai.defineTool(
  {
    name: 'dailySpecial',
    description: "Get today's special dish for a restaurant.",
    inputSchema: z.object({ restaurant: z.string() }),
    outputSchema: z.string(),
  },
  async ({ restaurant }) => `${restaurant}'s special today is mushroom risotto.`
);

/**
 * A dotprompt: a Handlebars template with a system instruction and the
 * `dailySpecial` tool. Tests use `echoModel` to assert this renders correctly
 * (system + interpolated variables). The structured output schema is supplied
 * by the flow at call time (see below) so the prompt itself stays text-
 * renderable — `echoModel` produces text, which a strict output schema would
 * reject.
 */
export const recommendPrompt = ai.definePrompt({
  name: 'recommendPrompt',
  input: { schema: RecommendInputSchema },
  system: 'You are a concise restaurant concierge.',
  prompt:
    'Recommend a dish at {{restaurant}} for someone feeling {{mood}}. ' +
    'Their budget is {{budgetUSD}} USD.',
  tools: [dailySpecial],
});

/**
 * A flow with real logic worth testing: it asks the model (via the prompt)
 * for a structured recommendation, validates it, and derives `withinBudget`
 * itself — that derivation, not the model, is what the tests pin down.
 */
export const recommendDish = ai.defineFlow(
  {
    name: 'recommendDish',
    inputSchema: RecommendInputSchema,
    outputSchema: DishResultSchema,
  },
  async (input) => {
    const { output } = await recommendPrompt(input, {
      output: { schema: RecommendationSchema },
    });
    if (!output) {
      throw new Error('Model did not return a structured recommendation.');
    }
    if (output.priceUSD <= 0) {
      throw new Error('Recommendation has a non-positive price.');
    }
    return {
      dish: output.dish,
      reason: output.reason,
      withinBudget: output.priceUSD <= input.budgetUSD,
    };
  }
);

/**
 * A streaming flow: it forwards the model's tokens out through the flow's own
 * stream (and returns the final text). Tests drive it with `flow.stream(...)`
 * to assert chunks arrive through the flow, not just from the model directly.
 */
export const streamRecommendation = ai.defineFlow(
  {
    name: 'streamRecommendation',
    inputSchema: z.object({ restaurant: z.string(), mood: z.string() }),
    outputSchema: z.string(),
    streamSchema: z.string(),
  },
  async ({ restaurant, mood }, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      prompt: `Recommend a dish at ${restaurant} for someone feeling ${mood}.`,
    });
    for await (const chunk of stream) {
      sendChunk(chunk.text);
    }
    return (await response).text;
  }
);

/**
 * A human-in-the-loop tool: booking a dish needs explicit confirmation, so the
 * tool `interrupt()`s on first run to hand control back to the caller, and
 * completes once generation is resumed with the human's answer. Tests drive
 * the pause -> resume round-trip with `mockModel` (see menu_test.ts).
 */
export const confirmBooking = ai.defineTool(
  {
    name: 'confirmBooking',
    description: 'Confirm with the user before booking a dish.',
    inputSchema: z.object({ dish: z.string() }),
    outputSchema: z.string(),
  },
  async ({ dish }, { interrupt, resumed }) => {
    if (resumed) return `Booked: ${dish}.`;
    return interrupt({ dish });
  }
);
