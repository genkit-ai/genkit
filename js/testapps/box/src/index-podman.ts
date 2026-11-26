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

// Genkit in a Box, container edition. Same shape as `index.ts`, but the boxed
// side runs in a podman container instead of a local OS sandbox. That buys real
// containment the local sandboxes cannot give you: network egress is BLOCKED,
// and memory/pid caps are enforced.
//
// Unlike the local-sandbox version this cannot be a single self-mode file: the
// container needs a linux-runnable entry, and the host's node binary and
// darwin-built node_modules (esbuild in particular) will not load there. So the
// boxed side is a separate compiled entry, `boxed-podman.ts` -> `lib/`.
//
// Run it:
//   pnpm build                       # compile both sides to lib/
//   GEMINI_API_KEY=... pnpm genkit:dev:podman
// then call the `runInBox` flow from the Dev UI.

import { box, podmanRunner } from '@genkit-ai/box';
import { googleAI } from '@genkit-ai/google-genai';
import { retry } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import path from 'node:path';

const ai = genkit({ plugins: [googleAI(), retry.plugin()] });

// The workspace root (js/). The runner mounts this at the SAME absolute path
// inside the container, which keeps pnpm's relative symlinks
// (node_modules/genkit -> ../../../genkit) resolvable. Derived from cwd rather
// than import.meta so this compiles to CommonJS like the rest of the testapp.
const workspaceRoot = path.resolve(process.cwd(), '../..');

const myBox = box(ai, {
  runner: podmanRunner({
    image: 'node:22-slim',
    // Compiled JS, not tsx: the host's esbuild binary is darwin-only and
    // cannot run in a linux container.
    cmd: 'node testapps/box/lib/boxed-podman.js',
    projectDir: workspaceRoot,
    // Real resource caps. The local sandbox providers have no equivalent.
    extraArgs: ['--memory=512m', '--pids-limit=256'],
  }),
});

// Declared from a spec rather than handed a local action: the implementation
// lives in the other entry (boxed-podman.ts), so there is nothing local to
// hand over. Shapes must match the boxed definition.
const boxedRunShell = myBox.defineTool({
  name: 'runShell',
  description:
    'Run a shell command inside the container and return its stdout.',
  inputSchema: z.object({ cmd: z.string() }),
  outputSchema: z.object({
    stdout: z.string().optional(),
    error: z.string().optional(),
  }),
});

// The agent and the GEMINI_API_KEY stay in this process. The container gets no
// host env at all, so the key cannot leak into the box.
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

// Exercise the boxed tool directly from the Dev UI, without the model.
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
