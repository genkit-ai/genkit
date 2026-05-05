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
import {
  SessionSnapshot,
  SessionStore,
  SessionStoreOptions,
} from 'genkit/beta';
import * as path from 'path';
import { ai } from './genkit.js';

export class FileSessionStore<S = unknown, I = unknown>
  implements SessionStore<S, I>
{
  private dirPath: string;
  private maxPersistedChainLength?: number;
  private snapshotPathPrefix?: (
    snapshotId: string,
    options?: SessionStoreOptions
  ) => string;

  /**
   * Creates a local file-system based session snapshot store.
   *
   * @param dirPath The local directory path where JSON snapshots will be stored.
   * @param options Configuration options for persistence behaviour.
   * @param options.maxPersistedChainLength Optional limit enforcing the maximum length of a snapshot chain.
   *   Once the history of a snapshot chain exceeds this length, older snapshots will be unlinked.
   */
  constructor(
    dirPath: string,
    options?: {
      maxPersistedChainLength?: number;
      snapshotPathPrefix?: (
        snapshotId: string,
        options?: SessionStoreOptions
      ) => string;
    }
  ) {
    this.dirPath = path.resolve(dirPath);
    if (!fs.existsSync(this.dirPath)) {
      fs.mkdirSync(this.dirPath, { recursive: true });
    }
    this.maxPersistedChainLength = options?.maxPersistedChainLength;
    this.snapshotPathPrefix = options?.snapshotPathPrefix;
  }

  private getFilePath(
    snapshotId: string,
    options?: SessionStoreOptions
  ): string {
    const prefix = this.snapshotPathPrefix
      ? this.snapshotPathPrefix(snapshotId, options)
      : 'global';
    const dir = path.join(this.dirPath, prefix);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    return path.join(dir, `${snapshotId}.json`);
  }

  /**
   * Retrieves a snapshot from the file system by its snapshotId.
   *
   * @param snapshotId The UUID of the snapshot to load.
   * @returns The reconstructed SessionSnapshot, or undefined if not found.
   */
  async getSnapshot(
    snapshotId: string,
    options?: SessionStoreOptions
  ): Promise<SessionSnapshot<S, I> | undefined> {
    const filePath = this.getFilePath(snapshotId, options);
    if (!fs.existsSync(filePath)) {
      return undefined;
    }
    const fileContents = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(fileContents) as SessionSnapshot<S, I>;
  }

  /**
   * Saves a new snapshot to the file system.
   * If a maxPersistedChainLength is specified, this will also walk back the `parentId`
   * chain and automatically prune (delete) older snapshots that exceed the history length.
   *
   * @param snapshot The SessionSnapshot instance to store.
   */
  async saveSnapshot(
    snapshot: SessionSnapshot<S, I>,
    options?: SessionStoreOptions
  ): Promise<void> {
    const filePath = this.getFilePath(snapshot.snapshotId, options);
    fs.writeFileSync(filePath, JSON.stringify(snapshot, null, 2), 'utf-8');

    if (this.maxPersistedChainLength && this.maxPersistedChainLength > 0) {
      let currentSnapshot: SessionSnapshot<S, I> | undefined = snapshot;
      let chainLength = 0;
      const chain: string[] = [];

      while (currentSnapshot) {
        chain.push(currentSnapshot.snapshotId);
        chainLength++;
        if (currentSnapshot.parentId) {
          currentSnapshot = await this.getSnapshot(
            currentSnapshot.parentId,
            options
          );
        } else {
          break;
        }
      }

      if (chainLength > this.maxPersistedChainLength) {
        // delete snapshots beyond the allowed length
        for (let i = this.maxPersistedChainLength; i < chainLength; i++) {
          const snapshotIdToDelete = chain[i];
          const pathToDelete = this.getFilePath(snapshotIdToDelete, options);
          if (fs.existsSync(pathToDelete)) {
            fs.unlinkSync(pathToDelete);
          }
        }
      }
    }
  }
}

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
export const pruningStore = new FileSessionStore<any>(
  './.snapshots-pruning',
  {
    maxPersistedChainLength: 3,
  }
);

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

    // verify that snapshot 1 is deleted but 2, 3, 4 are still there
    const snap1Exists = fs.existsSync(
      path.join('./.snapshots-pruning', `${snap1}.json`)
    );
    const snap2Exists = fs.existsSync(
      path.join('./.snapshots-pruning', `${snap2}.json`)
    );
    const snap3Exists = fs.existsSync(
      path.join('./.snapshots-pruning', `${snap3}.json`)
    );
    const snap4Exists = fs.existsSync(
      path.join('./.snapshots-pruning', `${turn4.result.snapshotId}.json`)
    );

    return {
      snap1Exists,
      snap2Exists,
      snap3Exists,
      snap4Exists,
    };
  }
);
