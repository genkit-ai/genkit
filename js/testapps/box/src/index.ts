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

// Genkit in a Box: one entry file defines a `runShell` tool AND boxes it (self
// mode). The tool body runs in a separate process; the agent and the model API
// key stay in this one. Only `runShell` calls cross the boundary.
//
// Run it:
//   GEMINI_API_KEY=... pnpm genkit:dev
// then call the `runInBox` flow or chat with `codingAgent` in the Dev UI.

import { box, execRunner } from '@genkit-ai/box';
import { googleAI } from '@genkit-ai/google-genai';
import { retry } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const run = promisify(exec);

const ai = genkit({ plugins: [googleAI(), retry.plugin()] });

// Run the boxed side as *this same program*, in a child process.
const myBox = box(ai, { runner: execRunner({ self: true }) });

// The real tool implementation. This body executes inside the box.
const runShell = ai.defineTool(
  {
    name: 'runShell',
    description: 'Run a shell command and return its stdout.',
    inputSchema: z.object({ cmd: z.string() }),
    outputSchema: z.object({
      stdout: z.string().optional(),
      error: z.string().optional(),
    }),
  },
  async ({ cmd }) => {
    try {
      const { stdout } = await run(cmd, { timeout: 10_000 });
      return { stdout };
    } catch (e) {
      return { error: `${e}` };
    }
  }
);

// A boxed handle to that tool. Hand over the real action (no schema
// restatement); types are inferred from `runShell`. Every call is routed into
// the box.
const boxedRunShell = myBox.fromTool(runShell);

// The agent (and the Gemini API key) live OUTSIDE the box. Only runShell
// crosses the boundary.
export const codingAgent = ai.defineAgent({
  name: 'codingAgent',
  system:
    'You are a helpful coding assistant. Use runShell to inspect the ' +
    'system when needed. Keep answers concise.',
  model: googleAI.model('gemini-flash-latest'),
  tools: [boxedRunShell],
  maxTurns: 20,
  use: [retry()],
});

// A flow so the boxed tool is easy to exercise directly from the Dev UI without
// going through the model.
export const runInBox = ai.defineFlow(
  {
    name: 'runInBox',
    inputSchema: z.object({ cmd: z.string() }),
    outputSchema: z.object({
      stdout: z.string().optional(),
      error: z.string().optional(),
    }),
  },
  async ({ cmd }) => boxedRunShell({ cmd })
);
