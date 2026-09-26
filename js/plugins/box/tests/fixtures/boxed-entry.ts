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

// A minimal real Genkit box. Spawned by the exec-runner integration test with
// GENKIT_REFLECTION_V2_SERVER and the host's secret set, so its runtime dials
// back into the test's reflection host. It just defines a couple of actions.

import { z } from 'genkit';
import { genkit, InMemorySessionStore } from 'genkit/beta';

const ai = genkit({});

// A tiny echo model so the agent needs no API key.
const echoModel = ai.defineModel({ name: 'echo' }, async (req) => {
  const last = req.messages.at(-1);
  const text = last?.content.map((p) => p.text ?? '').join('') ?? '';
  return {
    finishReason: 'stop',
    message: { role: 'model', content: [{ text: `echo:${text}` }] },
  };
});

ai.defineAgent({
  name: 'echoAgent',
  system: 'You echo.',
  model: echoModel,
  store: new InMemorySessionStore(),
});

ai.defineTool(
  {
    name: 'shout',
    description: 'Uppercases the input text.',
    inputSchema: z.object({ text: z.string() }),
    outputSchema: z.object({ out: z.string() }),
  },
  async ({ text }) => ({ out: text.toUpperCase() })
);

ai.defineFlow(
  {
    name: 'countTo',
    inputSchema: z.object({ n: z.number() }),
    outputSchema: z.object({ total: z.number() }),
    streamSchema: z.object({ i: z.number() }),
  },
  async ({ n }, { sendChunk }) => {
    for (let i = 1; i <= n; i++) sendChunk({ i });
    return { total: n };
  }
);

// Keep the process alive; the runtime holds the ws connection open.
setInterval(() => {}, 1 << 30);
