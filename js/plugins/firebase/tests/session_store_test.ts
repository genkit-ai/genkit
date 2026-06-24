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

// The comprehensive session-store test suite lives in the core implementation
// package, `@genkit-ai/google-cloud` (tests/session_store_test.ts). This file
// only smoke-tests the thin Firebase wrapper, verifying it derives a Firestore
// instance from a Firebase app and round-trips a snapshot end to end.

import { afterEach, beforeEach, describe, it } from '@jest/globals';
import * as assert from 'assert';
import { deleteApp, initializeApp, type App } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { FirestoreSessionStore } from '../src/session-store/firestore';

describe('FirestoreSessionStore (Firebase wrapper)', () => {
  let app: App;
  let collection: string;

  beforeEach(() => {
    process.env.FIRESTORE_EMULATOR_HOST = '127.0.0.1:8080';
    app = initializeApp({ projectId: 'genkit-test' }, `app-${Math.random()}`);
    collection = `sessions-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
  });

  afterEach(async () => {
    const db = getFirestore(app);
    for (const name of [
      collection,
      `${collection}-pointers`,
      `${collection}-shards`,
    ]) {
      await db.recursiveDelete(db.collection(name));
    }
    await deleteApp(app);
  });

  it('derives Firestore from a firebaseApp and round-trips a snapshot', async () => {
    const store = new FirestoreSessionStore<{ counter?: number }>({
      firebaseApp: app,
      collection,
    });

    const id = await store.saveSnapshot('snap-1', () => ({
      snapshotId: 'snap-1',
      createdAt: new Date().toISOString(),
      event: 'turnEnd',
      status: 'completed',
      state: { sessionId: 'sess-1', custom: { counter: 1 } },
    }));
    assert.strictEqual(id, 'snap-1');

    const bySnapshot = await store.getSnapshot({ snapshotId: 'snap-1' });
    assert.strictEqual(bySnapshot?.state?.custom?.counter, 1);

    const bySession = await store.getSnapshot({ sessionId: 'sess-1' });
    assert.strictEqual(bySession?.snapshotId, 'snap-1');
  });

  it('accepts an explicit db instance', async () => {
    const store = new FirestoreSessionStore<{ counter?: number }>({
      db: getFirestore(app),
      collection,
    });

    await store.saveSnapshot('snap-2', () => ({
      snapshotId: 'snap-2',
      createdAt: new Date().toISOString(),
      event: 'turnEnd',
      status: 'completed',
      state: { sessionId: 'sess-2', custom: { counter: 2 } },
    }));

    const read = await store.getSnapshot({ snapshotId: 'snap-2' });
    assert.strictEqual(read?.state?.custom?.counter, 2);
  });
});
