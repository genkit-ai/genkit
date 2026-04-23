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

const store = new InMemorySessionStore<{ foo: string }>();

export const abortableAgent = ai.defineSessionFlow(
  {
    name: 'abortableAgent',
    store,
  },
  async (sess, { abortSignal }) => {
    let isAborted = false;
    if (abortSignal) {
      abortSignal.onabort = () => {
        isAborted = true;
      };
    }

    await sess.run(async () => {
      for (let i = 0; i < 20; i++) {
        if (isAborted) {
          throw new Error('Operation aborted in background');
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    });

    return {
      message: {
        role: 'model',
        content: [{ text: 'Background execution completed' }],
      },
    };
  }
);

export const testAbortableAgent = ai.defineFlow(
  {
    name: 'testAbortableAgent',
    inputSchema: z.void(),
    outputSchema: z.any(),
  },
  async () => {
    const session = abortableAgent.streamBidi({});
    session.send({
      messages: [{ role: 'user', content: [{ text: 'start' }] }],
      detach: true,
    });

    const output = await session.output;
    const snapshotId = output.snapshotId!;

    await new Promise((resolve) => setTimeout(resolve, 200));

    await abortableAgent.abort(snapshotId);

    await new Promise((resolve) => setTimeout(resolve, 500));

    const snapshot = await store.getSnapshot(snapshotId);

    return {
      snapshotId,
      status: snapshot?.status,
    };
  }
);
