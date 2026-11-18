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

// Genkit in a Box, boxed-agent edition. The whole agent runs in the box
// (`boxed-agent.ts`), one process per chat session. This process holds no
// model and no tools; it only routes. `defineAgent` registers the boxed agent
// here, so it shows up and is chattable in the Dev UI like a local one.
//
// Run it:
//   GEMINI_API_KEY=... pnpm genkit:dev:agent
// then chat with `notesAgent` in the Dev UI. Start a second session and note
// that takeNote reports a different pid.

import { box, execRunner, sessionRoute } from '@genkit-ai/box';
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { NotesStateSchema } from './boxed-agent.js';

const ai = genkit({});

const agentBox = box(ai, {
  runner: execRunner({ cmd: 'tsx src/boxed-agent.ts' }),
  // One box per chat session, keyed by the session the call names (in the
  // agent's init: the Dev UI and `chat({ sessionId })` send it). Calls that
  // name none, like a later Dev UI turn resuming by snapshot alone, go to a
  // shared box; the file store lets that box pick the conversation up.
  route: sessionRoute,
  // Reclaim a session's box after 10 idle minutes (default: 5).
  retention: { idle: 10 * 60_000 },
});

// Declared from a spec: the agent lives only in the box. `stateManagement`
// must match it (it has a session store, so 'server').
export const notesAgent = agentBox.defineAgent<
  z.infer<typeof NotesStateSchema>
>({
  name: 'notesAgent',
  description: 'Takes notes, one box process per chat session.',
  stateManagement: 'server',
  stateSchema: NotesStateSchema,
});

// Drives the boxed agent in-process, the way an app backend would.
export const chatWithNotes = ai.defineFlow(
  {
    name: 'chatWithNotes',
    inputSchema: z.object({ sessionId: z.string(), message: z.string() }),
    outputSchema: z.object({ text: z.string(), notes: z.array(z.string()) }),
  },
  async ({ sessionId, message }) => {
    // A fresh chat per request: `sessionId` (picked by the caller) resumes the
    // session's latest turn, or starts it under that id. It is in every
    // request's init, so the route always sends it to the session's box.
    const res = await notesAgent.chat({ sessionId }).send(message);
    return { text: res.text, notes: res.state?.notes ?? [] };
  }
);
