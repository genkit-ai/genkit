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

import { expressHandler } from '@genkit-ai/express';
import { googleAI, vertexAI } from '@genkit-ai/google-genai';
import express, { type Request, type Response } from 'express';
import { UserFacingError, genkit, z } from 'genkit';
import type { ContextProvider, RequestData } from 'genkit/context';
import { logger } from 'genkit/logging';
import { ollama } from 'genkitx-ollama';

logger.setLogLevel('debug');

const ai = genkit({
  plugins: [
    googleAI(),
    vertexAI(),
    ollama({
      models: [
        { name: 'llama2', type: 'generate' },
        { name: 'gemma', type: 'chat' },
      ],
      serverAddress: 'http://127.0.0.1:11434', // default local address
    }),
  ],
});

export const jokeFlow = ai.defineFlow(
  { name: 'jokeFlow', inputSchema: z.string(), outputSchema: z.string() },
  async (subject, { context, sendChunk }) => {
    if (context!.auth!.username != 'Ali Baba') {
      throw new UserFacingError('PERMISSION_DENIED', context!.auth!.username!);
    }
    return await ai.run('call-llm', async () => {
      const llmResponse = await ai.generate({
        prompt: `tell me long joke about ${subject}`,
        model: googleAI.model('gemini-flash-latest'),
        config: {
          temperature: 1,
        },
        onChunk: (c) => sendChunk(c.text),
      });

      return llmResponse.text;
    });
  }
);

interface AuthContext {
  auth: {
    username: string;
  };
}

// ContextProviders often follow this pattern, where a factory function
// creates a ContextProvider that also validates the Context.
// This is often done either declaratively or with a callback.
// This type, once defined, can be used in other web frameworks such
// as Next.js as well.
function auth(requiredUser?: string): ContextProvider<AuthContext> {
  return (req: RequestData): AuthContext => {
    // Parsing:
    const token = req.headers['authorization'];
    const context: AuthContext = {
      auth: {
        // pretend we check auth token
        username: token === 'open sesame' ? 'Ali Baba' : '40 thieves',
      },
    };
    // validating
    if (requiredUser && context.auth.username != requiredUser) {
      throw new UserFacingError('PERMISSION_DENIED', context.auth.username);
    }
    return context;
  };
}

const app = express();
app.use(express.json());

const acls: Record<string, string> = {
  jokeFlow: 'Ali Baba',
};

// curl http://localhost:5000/jokeFlow?stream=true -d '{"data": "banana"}' -H "content-type: application/json" -H "authorization: open sesame"
ai.flows.forEach((f) => {
  app.post(
    `/${f.name}`,
    expressHandler(f, { contextProvider: auth(acls[f.name]) })
  );
});

// curl http://localhost:5000/jokeHandler?stream=true -d '{"data": "banana"}' -H "content-type: application/json"
app.post('/jokeHandler', expressHandler(jokeFlow));

// curl http://localhost:5000/jokeWithFlow?subject=banana
app.get('/jokeWithFlow', async (req: Request, res: Response) => {
  const subject = req.query['subject']?.toString();
  if (!subject) {
    res.status(400).send('provide subject query param');
    return;
  }
  res.send(await jokeFlow(subject));
});

// curl http://localhost:5000/jokeStream?subject=banana
app.get('/jokeStream', async (req: Request, res: Response) => {
  const subject = req.query['subject']?.toString();
  if (!subject) {
    res.status(400).send('provide subject query param');
    return;
  }

  res.writeHead(200, {
    'Content-Type': 'text/plain',
    'Transfer-Encoding': 'chunked',
  });
  await ai.generate({
    prompt: `Tell me a long joke about ${subject}`,
    model: googleAI.model('gemini-flash-latest'),
    config: {
      temperature: 1,
    },
    onChunk: (c) => {
      res.write(c.content[0].text);
    },
  });

  res.end();
});

const port = process.env.PORT || 5000;
app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
