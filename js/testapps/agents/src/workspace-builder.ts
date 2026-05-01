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
 * Workspace Builder — artifact production via defineAgent
 *
 * Demonstrates:
 *   • Using `defineAgent` with a tool that emits artifacts
 *   • `ai.currentSession().addArtifacts()` from inside a tool handler
 *   • Artifacts are automatically returned in `result.artifacts`
 *   • No need for `defineCustomAgent` — the standard agent API handles
 *     model calls, tool dispatch, streaming, and message management
 */

import { z } from 'genkit';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Tool — emits a generated code file as an artifact
// ---------------------------------------------------------------------------

const emitArtifact = ai.defineTool(
  {
    name: 'emitArtifact',
    description:
      'Call this tool to emit a generated code file to the user workspace.',
    inputSchema: z.object({ name: z.string(), content: z.string() }),
    outputSchema: z.object({ status: z.string() }),
  },
  async (input) => {
    const artifact = {
      name: input.name,
      parts: [{ text: input.content }],
    };
    ai.currentSession().addArtifacts([artifact]);
    return { status: `Artifact ${input.name} emitted successfully.` };
  }
);

// ---------------------------------------------------------------------------
// Agent — uses defineAgent (the standard shortcut API)
// ---------------------------------------------------------------------------

export const workspaceAgent = ai.defineAgent({
  name: 'workspacePrompt',
  model: 'googleai/gemini-2.5-flash',
  system: `You are a helpful code generation assistant. When the user asks you to create a file, use the emitArtifact tool to produce it.

Rules:
- Use the emitArtifact tool to create files. Pass the filename as "name" and the full file content as "content".
- You can create multiple files in a single turn if requested.
- After emitting artifacts, briefly confirm what you created.
- If the user asks you to modify a previously created file, emit a new artifact with the same name and updated content.`,
  tools: [emitArtifact],
});

// ---------------------------------------------------------------------------
// Test flow
// ---------------------------------------------------------------------------

export const testWorkspaceAgent = ai.defineFlow(
  {
    name: 'testWorkspaceAgent',
    inputSchema: z.string().default('Write poem.txt with a poem about genkit'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const res = await workspaceAgent.run(
      { messages: [{ role: 'user' as const, content: [{ text }] }] },
      { init: {}, onChunk: sendChunk }
    );
    return res.result;
  }
);
