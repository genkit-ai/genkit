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
    // A single chat persists to the FileSessionStore and carries the snapshot
    // forward across turns automatically.
    const chat = fileStoreAgent.chat();

    // Turn 1
    const res1 = await chat.send(
      'Hello! Please log this note: I started studying Genkit today.'
    );
    const snapshotId1 = res1.snapshotId!;

    // Turn 2 — continues from the persisted snapshot automatically.
    const res2 = await chat.send('What did I study today?');

    return {
      snapshotId1,
      reply1: res1.text,
      reply2: res2.text,
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
    // A single chat builds up a snapshot chain. We capture each turn's
    // snapshotId to assert which ones get pruned.
    const chat = pruningAgent.chat();

    const snap1 = (await chat.send('Turn 1')).snapshotId!;
    const snap2 = (await chat.send('Turn 2')).snapshotId!;
    const snap3 = (await chat.send('Turn 3')).snapshotId!;
    // Turn 4 (Snap 1 should be deleted here since max chain length is 3)
    const snap4 = (await chat.send('Turn 4')).snapshotId!;

    // Snapshots are stored under <dirPath>/global/<snapshotId>.json
    const snapshotDir = path.join('./.snapshots-pruning', 'global');
    const snap1Exists = fs.existsSync(path.join(snapshotDir, `${snap1}.json`));
    const snap2Exists = fs.existsSync(path.join(snapshotDir, `${snap2}.json`));
    const snap3Exists = fs.existsSync(path.join(snapshotDir, `${snap3}.json`));
    const snap4Exists = fs.existsSync(path.join(snapshotDir, `${snap4}.json`));

    return {
      snap1Exists,
      snap2Exists,
      snap3Exists,
      snap4Exists,
    };
  }
);
