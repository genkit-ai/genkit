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

import {
  fetchHandler,
  fetchHandlers,
  withActionOptions,
} from '@genkit-ai/fetch';
import { googleAI } from '@genkit-ai/google-genai';
import { serve } from '@hono/node-server';
import { UserFacingError, genkit, z } from 'genkit';
import type { ContextProvider } from 'genkit/context';
import { Hono } from 'hono';

const ai = genkit({
  plugins: [googleAI()],
});

// Example: context provider for auth (require header: Authorization: Bearer open-sesame)
const authContextProvider: ContextProvider<{ userId: string }> = (req) => {
  const auth = req.headers['authorization'];
  if (auth !== 'Bearer open-sesame') {
    throw new UserFacingError(
      'PERMISSION_DENIED',
      'Invalid or missing Authorization header. Use: Bearer open-sesame'
    );
  }
  return { userId: 'authenticated-user' };
};

// Actions (flows) served over HTTP via @genkit-ai/fetch
const helloFlow = ai.defineFlow('hello', async (input: string) => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.0-flash'),
    prompt: `Say hello to: ${input}`,
  });
  return text;
});

const greetingFlow = ai.defineFlow(
  {
    name: 'greeting',
    inputSchema: z.object({ name: z.string() }),
  },
  async (input) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-2.0-flash'),
      prompt: `Write a short greeting for someone named ${input.name}.`,
    });
    return text;
  }
);

const streamingFlow = ai.defineFlow(
  {
    name: 'streaming',
    inputSchema: z.object({ prompt: z.string() }),
  },
  async (input, { sendChunk }) => {
    const { stream } = ai.generateStream({
      model: googleAI.model('gemini-2.0-flash'),
      prompt: input.prompt,
    });
    let full = '';
    for await (const chunk of stream) {
      full += chunk.text ?? '';
      sendChunk(chunk.text ?? '');
    }
    return full;
  }
);

// Example: action that uses context (auth), secured with withActionOptions
const secureGreetingFlow = ai.defineFlow(
  {
    name: 'secureGreeting',
    inputSchema: z.object({ name: z.string() }),
  },
  async (input, { context }) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-2.0-flash'),
      prompt: `Say a short greeting to ${input.name} (user ${context?.userId}).`,
    });
    return text?.trim() ?? '';
  }
);

const actions = [
  helloFlow,
  greetingFlow,
  streamingFlow,
  withActionOptions(secureGreetingFlow, {
    contextProvider: authContextProvider,
  }),
];

const app = new Hono();

app.get('/', (c) =>
  c.json({
    message: 'Genkit + Hono + @genkit-ai/fetch',
    actions: ['hello', 'greeting', 'streaming', 'secureGreeting'],
    usage:
      'POST /api/<actionName> with body { "data": <input> }. secureGreeting requires header: Authorization: Bearer open-sesame',
  })
);

app.post('/models/gemini-flash', async (c) => {
  const model = await googleAI().model('gemini-2.0-flash');
  return fetchHandler(model)(c.req.raw);
});

// Mount Genkit actions under /api - fetchHandlers routes by path
app.all('/api/*', (c) => fetchHandlers(actions, '/api')(c.req.raw));

const port = Number(process.env.PORT) || 3780;
serve({ fetch: app.fetch, port }, (info) => {
  console.log(`Listening on http://localhost:${info.port}`);
  console.log(
    `Genkit actions: http://localhost:${info.port}/api/hello, /api/greeting, /api/streaming, /api/secureGreeting`
  );
});
