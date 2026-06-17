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

import { z } from 'genkit';
import { InMemorySessionStore } from 'genkit/beta';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Background Agent — demonstrates detached (background) execution
//
// Key concepts:
//   • `detach: true` in the input causes the server to start processing in
//     the background and return a snapshotId immediately.
//   • The client can poll `getSnapshotDataAction` to check the status
//     (pending → completed/failed/aborted).
//   • The client can call `abortAgentAction` to cancel background work.
//   • A persistent store is REQUIRED for detach to work — the server needs
//     somewhere to write the result when the background work completes.
// ---------------------------------------------------------------------------

const store = new InMemorySessionStore();

/**
 * The background agent. Using defineAgent so the prompt and agent are
 * defined together — but the key feature here is the `store`, which
 * enables `detach: true`.
 *
 * When the client sends `{ message: ..., detach: true }`, the server:
 * 1. Saves a snapshot with status 'pending' and returns the snapshotId.
 * 2. Continues processing the LLM request in the background.
 * 3. Updates the snapshot to status 'completed' (or 'failed') when complete.
 */
export const backgroundAgent = ai.defineAgent({
  name: 'backgroundAgent',
  system: `You are a senior research analyst. When given a topic, produce a comprehensive research report in markdown format.

Your report must include:
- **Executive Summary** — A concise overview of the topic and key findings.
- **Background & Context** — Historical context and current landscape.
- **Analysis** — Detailed analysis with data points and examples (3–4 subsections).
- **Implications** — What this means going forward.
- **Conclusion & Recommendations** — Actionable takeaways.

Be thorough, analytical, and evidence-based. Use markdown headings, bullet points, and bold text for structure.`,
  store,
});

/**
 * Programmatic test flow for the background agent — submits a detached
 * request, polls for completion, and returns the final status.
 */
export const testBackgroundAgent = ai.defineFlow(
  {
    name: 'testBackgroundAgent',
    inputSchema: z.void(),
    outputSchema: z.any(),
  },
  async () => {
    // `chat.detach(...)` submits a background turn and returns immediately with
    // a DetachedTask handle carrying the snapshotId.
    const chat = backgroundAgent.chat();
    const task = await chat.detach('Write a report on renewable energy trends');

    // `wait()` polls the server store until the task reaches a terminal state
    // (done / failed / aborted) and resolves with the final snapshot.
    const snapshot = await task.wait({ intervalMs: 2000 });

    return {
      snapshotId: task.snapshotId,
      status: snapshot?.status,
      messagePreview: snapshot?.state?.messages
        ?.slice(-1)?.[0]
        ?.content?.[0]?.text?.slice(0, 200),
    };
  }
);
