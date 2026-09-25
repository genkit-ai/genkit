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

import openAI from '@genkit-ai/compat-oai/openai';
import express from 'express';
import { genkit } from 'genkit';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';
import { toGenkitContent } from './message-adapter.js';
import { openuiSystemPrompt } from './openui-prompt.js';

const app = express();
const port = Number(process.env.PORT ?? 3400);
const modelName = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';

const ai = genkit({
  plugins: [openAI()],
});

const chatRequestSchema = z.object({
  threadId: z.string().min(1).max(200),
  runId: z.string().min(1).max(200),
  messages: z
    .array(
      z.object({
        role: z.enum(['user', 'assistant']),
        content: z.string().min(1).max(50_000),
      })
    )
    .min(1)
    .max(32),
});

function writeCompletionChunk(
  response: express.Response,
  requestId: string,
  delta: { role?: 'assistant'; content?: string },
  finishReason: 'stop' | null = null
) {
  response.write(
    `data: ${JSON.stringify({
      id: requestId,
      object: 'chat.completion.chunk',
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    })}\n\n`
  );
}

app.use(express.json({ limit: '256kb' }));

app.get('/api/health', (_request, response) => {
  response.json({
    ready: Boolean(process.env.OPENAI_API_KEY),
    provider: 'openai-via-genkit',
    model: modelName,
  });
});

app.post('/api/chat', async (request, response) => {
  if (!process.env.OPENAI_API_KEY) {
    response.status(503).json({
      error: 'Set OPENAI_API_KEY in the server environment before chatting.',
    });
    return;
  }

  const parsed = chatRequestSchema.safeParse(request.body);
  if (!parsed.success) {
    response.status(400).json({ error: 'Invalid chat request.' });
    return;
  }

  const latestMessage = parsed.data.messages.at(-1);
  if (!latestMessage || latestMessage.role !== 'user') {
    response
      .status(400)
      .json({ error: 'The last message must be a user turn.' });
    return;
  }

  const requestId = crypto.randomUUID();
  console.log(`Genkit request ${requestId} started`);
  const abortController = new AbortController();
  request.on('aborted', () => abortController.abort());
  response.on('close', () => {
    if (!response.writableEnded) {
      abortController.abort();
    }
  });

  try {
    const history = parsed.data.messages.slice(0, -1).map((message) => ({
      role:
        message.role === 'assistant' ? ('model' as const) : ('user' as const),
      content: [{ text: toGenkitContent(message) }],
    }));

    const { stream } = ai.generateStream({
      model: openAI.model(modelName),
      system: openuiSystemPrompt,
      messages: history,
      prompt: toGenkitContent(latestMessage),
      abortSignal: abortController.signal,
    });

    response.status(200);
    response.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    response.setHeader('Cache-Control', 'no-cache, no-transform');
    response.setHeader('X-Accel-Buffering', 'no');
    response.setHeader('X-Genkit-Request-Id', requestId);
    response.flushHeaders();

    writeCompletionChunk(response, requestId, { role: 'assistant' });

    for await (const chunk of stream) {
      if (chunk.text) {
        writeCompletionChunk(response, requestId, { content: chunk.text });
      }
    }
    writeCompletionChunk(response, requestId, {}, 'stop');
    response.write('data: [DONE]\n\n');
    response.end();
  } catch (error) {
    if (abortController.signal.aborted) {
      return;
    }

    console.error(`Genkit request ${requestId} failed`, error);
    if (!response.headersSent) {
      response.status(502).json({ error: 'The model request failed.' });
      return;
    }
    response.destroy();
  }
});

const currentDirectory = path.dirname(fileURLToPath(import.meta.url));
const staticDirectory = path.resolve(currentDirectory, '../dist');
app.use('/api', (_request, response) => {
  response.status(404).json({ error: 'Unknown API route.' });
});
app.use(express.static(staticDirectory));
app.use((_request, response) => {
  response.sendFile(path.join(staticDirectory, 'index.html'));
});

app.listen(port, () => {
  console.log(`Genkit OpenUI server listening on http://localhost:${port}`);
});
