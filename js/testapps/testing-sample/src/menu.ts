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

import { genkit, z } from 'genkit';

/**
 * Builds the app's Genkit instance, tool, and flow. Exposed as a factory so
 * each test can build a fresh, isolated app (its own registry) and register a
 * mock under the default model name without colliding with other tests. The
 * default model is referenced by name only — in production you'd register it
 * from a provider plugin (e.g. googleAI), and in tests you register a mock
 * under the same name. Either way the flow code below is unchanged.
 */
export function createMenuApp() {
  const ai = genkit({ model: 'menuModel' });

  /**
   * A tool the model can call to look up today's specials for a restaurant.
   * In a real app this would hit a database; here it is deterministic.
   */
  const dailySpecial = ai.defineTool(
    {
      name: 'dailySpecial',
      description: "Get today's special dish for a restaurant.",
      inputSchema: z.object({ restaurant: z.string() }),
      outputSchema: z.string(),
    },
    async ({ restaurant }) =>
      `${restaurant}'s special today is mushroom risotto.`
  );

  /**
   * A flow that asks a model to recommend a dish, with the `dailySpecial` tool
   * available. Returns the model's text recommendation.
   */
  const recommendDish = ai.defineFlow(
    {
      name: 'recommendDish',
      inputSchema: z.object({ restaurant: z.string(), mood: z.string() }),
      outputSchema: z.string(),
    },
    async ({ restaurant, mood }) => {
      const { text } = await ai.generate({
        prompt: `Recommend a dish at ${restaurant} for someone feeling ${mood}.`,
        tools: [dailySpecial],
      });
      return text;
    }
  );

  return { ai, dailySpecial, recommendDish };
}

/**
 * The default app singleton, used by the dev server (`pnpm genkit:dev`). Tests
 * call {@link createMenuApp} directly for isolation.
 */
export const { ai, dailySpecial, recommendDish } = createMenuApp();
