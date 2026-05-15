/**
 * Copyright 2024 Google LLC
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

import { expressHandler } from '@genkit-ai/express';
import {
  FirestoreStreamManager,
  RtdbStreamManager,
} from '@genkit-ai/firebase/beta';
import express from 'express';
import { initializeApp } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { genkit, z } from 'genkit';
import { InMemoryStreamManager } from 'genkit/beta';

const ai = genkit({});

export const streamy = ai.defineFlow(
  {
    name: 'streamy',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: z.object({ count: z.number() }),
  },
  async (count, { sendChunk }) => {
    let i = 0;
    for (; i < count; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      sendChunk({ count: i });
    }
    return `done: ${count}, streamed: ${i} times`;
  }
);

export const streamyThrowy = ai.defineFlow(
  {
    name: 'streamyThrowy',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: z.object({ count: z.number() }),
  },
  async (count, { sendChunk }) => {
    let i = 0;
    for (; i < count; i++) {
      if (i == 3) {
        throw new Error('whoops');
      }
      await new Promise((r) => setTimeout(r, 1000));
      sendChunk({ count: i });
    }
    return `done: ${count}, streamed: ${i} times`;
  }
);

const fApp = initializeApp();
export const rtdb = new RtdbStreamManager({
  firebaseApp: fApp,
  refPrefix: 'streamy',
});
export const firestore = new FirestoreStreamManager({
  firebaseApp: fApp,
  db: getFirestore(fApp),
  collection: 'streamy',
});

const inMemory = new InMemoryStreamManager();
const app = express();
app.use(express.json());

app.post('/streamy', expressHandler(streamy, { streamManager: inMemory }));
app.post(
  '/streamyFirestore',
  expressHandler(streamy, { streamManager: firestore })
);
app.post('/streamyRtdb', expressHandler(streamy, { streamManager: rtdb }));
app.post(
  '/streamyThrowy',
  expressHandler(streamyThrowy, { streamManager: inMemory })
);
app.post(
  '/streamyThrowyFirestore',
  expressHandler(streamyThrowy, { streamManager: firestore })
);
app.post(
  '/streamyThrowyRtdb',
  expressHandler(streamyThrowy, { streamManager: rtdb })
);

app.listen(3500);
