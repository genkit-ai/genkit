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
import { FileSessionStore, continuationToSnapshotId } from 'genkit/beta';
import * as path from 'path';
import { ai, liteModel } from './genkit.js';

export { FileSessionStore };

export const fileStore = new FileSessionStore<any>('./.snapshots');

export const fileStoreAgent = ai.defineAgent({
  name: 'fileStoreAgent',
  model: liteModel,
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

    const continuation1 = turn1.result.continuationId!;
    const snapshotId1 = continuationToSnapshotId(continuation1);

    // Resume from the previous turn using the opaque continuation token.
    const turn2 = await fileStoreAgent.run(
      {
        messages: [
          { role: 'user', content: [{ text: 'What did I study today?' }] },
        ],
      },
      { init: { continuationId: continuation1 } }
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

export const pruningAgent = ai.defineAgent({
  name: 'pruningAgent',
  model: liteModel,
  system: `You are a personal logbook assistant.`,
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
    const cont1 = turn1.result.continuationId!;
    const snap1 = continuationToSnapshotId(cont1)!;

    const turn2 = await pruningAgent.run(
      {
        messages: [{ role: 'user', content: [{ text: 'Turn 2' }] }],
      },
      { init: { continuationId: cont1 } }
    );
    const cont2 = turn2.result.continuationId!;
    const snap2 = continuationToSnapshotId(cont2)!;

    const turn3 = await pruningAgent.run(
      {
        messages: [{ role: 'user', content: [{ text: 'Turn 3' }] }],
      },
      { init: { continuationId: cont2 } }
    );
    const cont3 = turn3.result.continuationId!;
    const snap3 = continuationToSnapshotId(cont3)!;

    // Run Turn 4 (Snap 1 should be deleted here since max chain length is 3)
    const turn4 = await pruningAgent.run(
      {
        messages: [{ role: 'user', content: [{ text: 'Turn 4' }] }],
      },
      { init: { continuationId: cont3 } }
    );

    // Snapshots are stored under <dirPath>/global/<snapshotId>.json
    const snapshotDir = path.join('./.snapshots-pruning', 'global');
    const snap1Exists = fs.existsSync(path.join(snapshotDir, `${snap1}.json`));
    const snap2Exists = fs.existsSync(path.join(snapshotDir, `${snap2}.json`));
    const snap3Exists = fs.existsSync(path.join(snapshotDir, `${snap3}.json`));
    const snap4 = continuationToSnapshotId(turn4.result.continuationId);
    const snap4Exists = fs.existsSync(path.join(snapshotDir, `${snap4}.json`));

    return {
      snap1Exists,
      snap2Exists,
      snap3Exists,
      snap4Exists,
    };
  }
);
