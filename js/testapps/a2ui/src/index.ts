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

/**
 * A2UI testapp backend.
 *
 * Defines a "generative UI" agent with `ai.defineAgent` — the whole A2UI
 * integration is the `a2ui()` middleware in `use`. The agent is served over
 * HTTP with `expressHandler`, and the browser talks to it with `remoteAgent`
 * from `genkit/beta/client`. Session state (history) is server-managed by the
 * agent's `InMemorySessionStore`.
 */

import { a2ui } from '@genkit-ai/a2ui';
import { expressHandler } from '@genkit-ai/express';
import { googleAI } from '@genkit-ai/google-genai';
import cors from 'cors';
import express from 'express';
import { existsSync } from 'fs';
import { z } from 'genkit';
import { InMemorySessionStore, genkit } from 'genkit/beta';
import { logger } from 'genkit/logging';
import { join } from 'path';

logger.setLogLevel('debug');

const ai = genkit({
  plugins: [googleAI(), a2ui.plugin()],
});

/** A demo tool the model can call to fetch (fake) weather data. */
const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'Gets the current weather for a given city.',
    inputSchema: z.object({ city: z.string() }),
    outputSchema: z.object({
      city: z.string(),
      tempC: z.number(),
      condition: z.string(),
      humidity: z.number(),
    }),
  },
  async ({ city }) => {
    // Deterministic pseudo-random values so the demo is stable per-city.
    const seed = [...city].reduce((a, c) => a + c.charCodeAt(0), 0);
    const conditions = ['Sunny', 'Partly cloudy', 'Rainy', 'Windy', 'Foggy'];
    return {
      city,
      tempC: 10 + (seed % 20) + (Math.random() - 0.5) * 4,
      condition: conditions[seed % conditions.length],
      humidity: 40 + (seed % 50),
    };
  }
);

/**
 * The A2UI-enabled agent. The whole integration is `a2ui()` in `use`. An
 * `InMemorySessionStore` makes state server-managed, so the browser only needs
 * to pass a session id (handled for it by `remoteAgent`).
 */
export const uiAgent = ai.defineAgent({
  name: 'uiAgent',
  model: googleAI.model('gemini-flash-latest'),
  system: `You are a helpful assistant that can render rich UI.
Prefer rendering an A2UI surface whenever a result is clearer shown than told —
for example weather, comparisons, lists, forms, or anything interactive. Keep any
prose brief; put the substance in the UI. When asked about weather, call the
getWeather tool, then render a nice Card/Column summarizing it (temperature,
condition, humidity). Feel free to add a Button (e.g. "Refresh") when useful.`,
  tools: [getWeather],
  use: [a2ui()], // defaults to the bundled 'basic' catalog
  store: new InMemorySessionStore(),
});

// --- HTTP server: expose the agent + serve the built web frontend. ----------

const PORT = Number(process.env.PORT ?? 8080);
const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// The agent is a bidi action; `expressHandler` serves it directly. `remoteAgent`
// in the browser hits this endpoint (proxied at /api/uiAgent by Vite in dev).
app.post('/api/uiAgent', expressHandler(uiAgent));

// Serve the Vite build output if present (production / `pnpm start`).
// Resolved from the process CWD (the testapp dir when run via pnpm scripts).
const webDist = join(process.cwd(), 'web', 'dist');
if (existsSync(webDist)) {
  app.use(express.static(webDist));
  app.get('*', (_req, res) => res.sendFile(join(webDist, 'index.html')));
  logger.info(`Serving web UI from ${webDist}`);
} else {
  logger.info(
    'No web build found. Run `pnpm --filter a2ui-testapp web:dev` for the Vite dev server (http://localhost:5173).'
  );
}

app.listen(PORT, () => {
  logger.info(`A2UI testapp server listening on http://localhost:${PORT}`);
});
