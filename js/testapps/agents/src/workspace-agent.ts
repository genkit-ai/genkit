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
 * Workspace Builder — artifact production via the `artifacts` middleware
 *
 * Demonstrates:
 *   • Using the `artifacts()` middleware to give the model `write_artifact`
 *     and `read_artifact` tools automatically
 *   • No custom tool definition needed — the middleware handles everything
 *   • Artifacts are automatically returned in `result.artifacts` and
 *     streamed via `artifact` chunks
 *   • Artifacts are deduplicated by name (writing the same name replaces it)
 */

import { artifacts } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Agent — uses defineAgent with the artifacts middleware
// ---------------------------------------------------------------------------

export const workspaceAgent = ai.defineAgent({
  name: 'workspaceAgent',
  system: `You are a helpful code generation assistant. When the user asks you to create a file, use the write_artifact tool to produce it.

Rules:
- Use the write_artifact tool to create files. Pass the filename as "name" and the full file content as "content".
- You can create multiple files in a single turn if requested.
- After writing artifacts, briefly confirm what you created.
- If the user asks you to modify a previously created file, use read_artifact to view the current content, then write_artifact with the same name and updated content.
- You can use read_artifact to review any previously created files.`,
  use: [artifacts()],
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
    const chat = workspaceAgent.chat();
    const turn = chat.sendStream(text);
    for await (const chunk of turn.stream) {
      sendChunk(chunk.raw);
    }
    const res = await turn.response;
    return res.raw;
  }
);
