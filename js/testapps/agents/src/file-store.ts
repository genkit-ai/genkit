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

import * as fs from 'fs';
import { z } from 'genkit';
import { FileSessionStore } from 'genkit/beta';
import * as path from 'path';
import { ai } from './genkit.js';

export { FileSessionStore };

export const fileStore = new FileSessionStore<any>('./.snapshots');

// defineAgent registers the prompt internally, so pruningAgent can also
// reference it by name via definePromptAgent.
export const fileStoreAgent = ai.defineAgent({
  name: 'fileStorePrompt',
  model: 'googleai/gemini-flash-lite-latest',
  system: `You are a personal logbook assistant.`,
  store: fileStore,
});

export const testFileStoreAgent = ai.defineFlow(
  {
    name: 'testFileStoreAgent',
    inputSchema: z.string().default('Alice'),
    outputSchema: z.any(),
  },
  async (userName) => {
    // Run Turn 1
    const turn1 = await fileStoreAgent.run(
      {
        messages: [
          {
            role: 'user',
            content: [
              {
                text: 'Hello! Please log this note: I started studying Genkit today.',
              },
            ],
          },
        ],
      },
      {}
    );

    const snapshotId1 = turn1.result.snapshotId!;

    // Now simulate Turn 2 by resuming from the written File snapshot
    const turn2 = await fileStoreAgent.run(
      {
        messages: [
          { role: 'user', content: [{ text: 'What did I study today?' }] },
        ],
      },
      { init: { snapshotId: snapshotId1 } }
    );

    return {
      snapshotId1,
      reply1: turn1.result.message?.content?.map((c) => c.text || '').join(''),
      reply2: turn2.result.message?.content?.map((c) => c.text || '').join(''),
    };
  }
);
export const pruningStore = new FileSessionStore<any>('./.snapshots-pruning', {
  maxPersistedChainLength: 3,
});

export const pruningAgent = ai.definePromptAgent({
  promptName: 'fileStorePrompt',
  store: pruningStore,
});

export const testFileStoreChainPruningAgent = ai.defineFlow(
  {
    name: 'testFileStoreChainPruningAgent',
    inputSchema: z.string().default('Alice'),
    outputSchema: z.any(),
  },
  async (userName) => {
    // Run Turn 1
    const turn1 = await pruningAgent.run(
      {
        messages: [{ role: 'user', content: [{ text: 'Turn 1' }] }],
      },
      {}
    );
    const snap1 = turn1.result.snapshotId!;

    // Run Turn 2
    const turn2 = await pruningAgent.run(
      {
        messages: [{ role: 'user', content: [{ text: 'Turn 2' }] }],
      },
      { init: { snapshotId: snap1 } }
    );
    const snap2 = turn2.result.snapshotId!;

    // Run Turn 3
    const turn3 = await pruningAgent.run(
      {
        messages: [{ role: 'user', content: [{ text: 'Turn 3' }] }],
      },
      { init: { snapshotId: snap2 } }
    );
    const snap3 = turn3.result.snapshotId!;

    // Run Turn 4 (Snap 1 should be deleted here since max chain length is 3)
    const turn4 = await pruningAgent.run(
      {
        messages: [{ role: 'user', content: [{ text: 'Turn 4' }] }],
      },
      { init: { snapshotId: snap3 } }
    );

    // Snapshots are stored under <dirPath>/global/<snapshotId>.json
    const snapshotDir = path.join('./.snapshots-pruning', 'global');
    const snap1Exists = fs.existsSync(path.join(snapshotDir, `${snap1}.json`));
    const snap2Exists = fs.existsSync(path.join(snapshotDir, `${snap2}.json`));
    const snap3Exists = fs.existsSync(path.join(snapshotDir, `${snap3}.json`));
    const snap4Exists = fs.existsSync(
      path.join(snapshotDir, `${turn4.result.snapshotId}.json`)
    );

    return {
      snap1Exists,
      snap2Exists,
      snap3Exists,
      snap4Exists,
    };
  }
);
