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
//     (pending → done/failed/aborted).
//   • The client can call `abortSessionFlowAction` to cancel background work.
//   • A persistent store is REQUIRED for detach to work — the server needs
//     somewhere to write the result when the background work completes.
// ---------------------------------------------------------------------------

const store = new InMemorySessionStore();

/**
 * Prompt for generating research reports. The model is given a topic and
 * produces a comprehensive, multi-section markdown report.
 */
const reportPrompt = ai.definePrompt({
  name: 'reportPrompt',
  model: 'googleai/gemini-flash-latest',
  system: `You are a senior research analyst. When given a topic, produce a comprehensive research report in markdown format.

Your report must include:
- **Executive Summary** — A concise overview of the topic and key findings.
- **Background & Context** — Historical context and current landscape.
- **Analysis** — Detailed analysis with data points and examples (3–4 subsections).
- **Implications** — What this means going forward.
- **Conclusion & Recommendations** — Actionable takeaways.

Be thorough, analytical, and evidence-based. Use markdown headings, bullet points, and bold text for structure.`,
});

/**
 * The background agent session flow. Using defineSessionFlowFromPrompt
 * so the LLM interaction is handled automatically — but the key feature
 * here is the `store`, which enables `detach: true`.
 *
 * When the client sends `{ messages: [...], detach: true }`, the server:
 * 1. Saves a snapshot with status 'pending' and returns the snapshotId.
 * 2. Continues processing the LLM request in the background.
 * 3. Updates the snapshot to status 'done' (or 'failed') when complete.
 */
export const backgroundAgent = ai.defineSessionFlowFromPrompt({
  store,
  promptName: 'reportPrompt',
  defaultInput: {},
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
    const session = backgroundAgent.streamBidi({});
    session.send({
      messages: [
        {
          role: 'user',
          content: [{ text: 'Write a report on renewable energy trends' }],
        },
      ],
      detach: true,
    });

    const output = await session.output;
    const snapshotId = output.snapshotId!;

    // Poll until done, failed, or aborted
    let snapshot: any;
    for (let i = 0; i < 60; i++) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      snapshot = await store.getSnapshot(snapshotId);
      if (snapshot?.status && snapshot.status !== 'pending') break;
    }

    return {
      snapshotId,
      status: snapshot?.status,
      messagePreview: snapshot?.state?.messages
        ?.slice(-1)?.[0]
        ?.content?.[0]?.text?.slice(0, 200),
    };
  }
);
