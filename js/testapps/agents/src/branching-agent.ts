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
import { ai, liteModel } from './genkit.js';

var count = 0;

ai.defineHelper('roundRobin', (o1, o2) => {
  console.log(count, o1, o2);
  return count++ % 2 ? o1 : o2;
});

export const branchingStore = new InMemorySessionStore();

export const branchingAgent = ai.defineAgent({
  name: 'branchingAgent',
  model: liteModel,
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
    // A single chat carries state forward automatically. To *branch*, we open
    // a new chat attached to an earlier snapshot via `chat({ snapshotId })`.
    const root = branchingAgent.chat();
    const res1 = await root.send('Hello!');
    const snapshot1 = res1.snapshotId;

    // Branch A: Bob.
    const branchA = branchingAgent.chat({ snapshotId: snapshot1 });
    await branchA.send('My name is Bob.');
    const resA = await branchA.send('What is my name?');

    // Branch B: John — forks from the SAME snapshot1.
    const branchB = branchingAgent.chat({ snapshotId: snapshot1 });
    await branchB.send('My name is John.');
    const resB = await branchB.send('What is my name?');

    return {
      snapshotUsedForBranching: snapshot1,
      branchAResponse: resA.text,
      branchBResponse: resB.text,
    };
  }
);
