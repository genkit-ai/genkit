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

var count = 0;

ai.defineHelper('roundRobin', (o1, o2) => {
  console.log(count, o1, o2);
  return count++ % 2 ? o1 : o2;
});

export const branchingStore = new InMemorySessionStore();

export const nameAgent = ai.defineAgent({
  name: 'namePrompt',
  model: 'googleai/gemini-flash-lite-latest',
  input: { schema: z.object({}) },
  system: `You are a {{ roundRobin 'sarcastic' 'business-like' }} assistant.`,
  store: branchingStore,
});

export const demonstrateBranching = ai.defineFlow(
  {
    name: 'demonstrateBranching',
    inputSchema: z.void(),
    outputSchema: z.any(),
  },
  async () => {
    const turn1 = await nameAgent.run(
      {
        messages: [{ role: 'user' as const, content: [{ text: 'Hello!' }] }],
      },
      { init: {} }
    );

    const snapshot1 = turn1.result.snapshotId;

    const turn2A = await nameAgent.run(
      {
        messages: [
          { role: 'user' as const, content: [{ text: 'My name is Bob.' }] },
        ],
      },
      { init: { snapshotId: snapshot1 } }
    );

    const snapshot2A = turn2A.result.snapshotId;

    const turn3A = await nameAgent.run(
      {
        messages: [
          { role: 'user' as const, content: [{ text: 'What is my name?' }] },
        ],
      },
      { init: { snapshotId: snapshot2A } }
    );

    const turn2B = await nameAgent.run(
      {
        messages: [
          { role: 'user' as const, content: [{ text: 'My name is John.' }] },
        ],
      },
      { init: { snapshotId: snapshot1 } }
    );

    const snapshot2B = turn2B.result.snapshotId;

    const turn3B = await nameAgent.run(
      {
        messages: [
          { role: 'user' as const, content: [{ text: 'What is my name?' }] },
        ],
      },
      { init: { snapshotId: snapshot2B } }
    );

    return {
      snapshotUsedForBranching: snapshot1,
      branchAResponse: turn3A.result.message?.content
        ?.map((c) => c.text || '')
        .join(''),
      branchBResponse: turn3B.result.message?.content
        ?.map((c) => c.text || '')
        .join(''),
    };
  }
);
