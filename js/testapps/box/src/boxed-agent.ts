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

// The boxed side of the agent demo: the WHOLE agent (model calls, tools,
// conversation state) lives here, started by `index-agent.ts`, one process
// per chat session. Nothing but reflection calls cross the boundary.

import { googleAI } from '@genkit-ai/google-genai';
import { z } from 'genkit';
import { genkit, InMemorySessionStore } from 'genkit/beta';

const ai = genkit({ plugins: [googleAI()] });

// State the agent keeps across turns; the host shows it in the Dev UI.
export const NotesStateSchema = z.object({ notes: z.array(z.string()) });

const takeNote = ai.defineTool(
  {
    name: 'takeNote',
    description: 'Remember a short note for later in this conversation.',
    inputSchema: z.object({ note: z.string() }),
    outputSchema: z.object({ pid: z.number(), count: z.number() }),
  },
  async ({ note }) => {
    let count = 0;
    ai.currentSession<z.infer<typeof NotesStateSchema>>().updateCustom((s) => {
      const notes = [...(s?.notes ?? []), note];
      count = notes.length;
      return { notes };
    });
    // The pid proves each session gets its own box process.
    return { pid: process.pid, count };
  }
);

ai.defineAgent({
  name: 'notesAgent',
  system:
    'You are a note taker. When the user shares something worth remembering, ' +
    'call takeNote. Keep answers short, and mention the pid takeNote returns.',
  model: googleAI.model('gemini-flash-latest'),
  tools: [takeNote],
  store: new InMemorySessionStore(),
});

// Keep the process alive serving reflection requests.
setInterval(() => {}, 1 << 30);
