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

import { Firestore } from '@google-cloud/firestore';
import { afterEach, beforeEach, describe, it } from '@jest/globals';
import * as assert from 'assert';
import type { SessionSnapshotInput } from 'genkit/beta';
import {
  FirestoreSessionStore,
  type FirestoreSessionStoreOptions,
} from '../src/session-store/firestore';

interface Custom {
  counter?: number;
  notes?: string[];
}

/** Builds a snapshot input with sensible defaults. */
function snapshot(
  overrides: Partial<SessionSnapshotInput<Custom>> & {
    state: SessionSnapshotInput<Custom>['state'];
  }
): SessionSnapshotInput<Custom> {
  return {
    createdAt: new Date().toISOString(),
    event: 'turnEnd',
    status: 'completed',
    ...overrides,
  };
}

describe('FirestoreSessionStore', () => {
  let db: Firestore;
  let store: FirestoreSessionStore<Custom>;
  // Collections created during a test, cleaned up afterwards. We only delete
  // what this file created (not every project collection) so it can run in
  // parallel with the other emulator test files.
  let createdCollections: string[];

  /** Creates a store and registers its collections for cleanup. */
  function makeStore(
    opts?: Omit<FirestoreSessionStoreOptions, 'db' | 'collection'> & {
      collection?: string;
    }
  ): FirestoreSessionStore<Custom> {
    const collection =
      opts?.collection ??
      `sessions-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
    createdCollections.push(
      collection,
      `${collection}-pointers`,
      `${collection}-shards`
    );
    return new FirestoreSessionStore<Custom>({
      ...opts,
      db,
      collection,
    });
  }

  beforeEach(() => {
    process.env.FIRESTORE_EMULATOR_HOST = '127.0.0.1:8080';
    db = new Firestore({ projectId: 'genkit-test' });
    createdCollections = [];
    store = makeStore();
  });

  afterEach(async () => {
    for (const name of createdCollections) {
      await db.recursiveDelete(db.collection(name));
    }
    await db.terminate();
  });

  it('saves a snapshot and reads it back by snapshotId', async () => {
    const id = await store.saveSnapshot(undefined, () =>
      snapshot({
        snapshotId: 'snap-1',
        state: { sessionId: 'sess-1', custom: { counter: 1 } },
      })
    );
    assert.strictEqual(id, 'snap-1');

    const read = await store.getSnapshot({ snapshotId: 'snap-1' });
    assert.ok(read);
    assert.strictEqual(read.snapshotId, 'snap-1');
    assert.strictEqual(read.state.sessionId, 'sess-1');
    assert.deepStrictEqual(read.state.custom, { counter: 1 });
  });

  it('returns undefined for missing snapshot/session', async () => {
    assert.strictEqual(
      await store.getSnapshot({ snapshotId: 'nope' }),
      undefined
    );
    assert.strictEqual(
      await store.getSnapshot({ sessionId: 'nope' }),
      undefined
    );
  });

  it('resolves the latest leaf snapshot by sessionId', async () => {
    await store.saveSnapshot('s1', () =>
      snapshot({
        snapshotId: 's1',
        state: { sessionId: 'sess-2', custom: { counter: 1 } },
      })
    );
    await store.saveSnapshot('s2', () =>
      snapshot({
        snapshotId: 's2',
        parentId: 's1',
        state: { sessionId: 'sess-2', custom: { counter: 2 } },
      })
    );

    const leaf = await store.getSnapshot({ sessionId: 'sess-2' });
    assert.ok(leaf);
    assert.strictEqual(leaf.snapshotId, 's2');
    assert.deepStrictEqual(leaf.state.custom, { counter: 2 });
  });

  it('reconstructs full state from a chain of diffs', async () => {
    await store.saveSnapshot('c1', () =>
      snapshot({
        snapshotId: 'c1',
        state: {
          sessionId: 'sess-3',
          custom: { counter: 1, notes: ['a'] },
        },
      })
    );
    await store.saveSnapshot('c2', () =>
      snapshot({
        snapshotId: 'c2',
        parentId: 'c1',
        state: {
          sessionId: 'sess-3',
          custom: { counter: 2, notes: ['a', 'b'] },
        },
      })
    );
    await store.saveSnapshot('c3', () =>
      snapshot({
        snapshotId: 'c3',
        parentId: 'c2',
        state: {
          sessionId: 'sess-3',
          custom: { counter: 3, notes: ['a', 'b', 'c'] },
        },
      })
    );

    // Middle snapshot reconstructs correctly.
    const mid = await store.getSnapshot({ snapshotId: 'c2' });
    assert.deepStrictEqual(mid?.state.custom, {
      counter: 2,
      notes: ['a', 'b'],
    });

    // Leaf reconstructs correctly via sessionId.
    const leaf = await store.getSnapshot({ sessionId: 'sess-3' });
    assert.deepStrictEqual(leaf?.state.custom, {
      counter: 3,
      notes: ['a', 'b', 'c'],
    });

    // The stored document only persists a diff, not the full state.
    const colName = (store as any).snapshots.id as string;
    const raw = await db.collection(colName).doc('c3').get();
    const data = raw.data();
    assert.ok(Array.isArray(data?.statePatch));
    assert.strictEqual(data?.state, undefined);
  });

  it('passes the current snapshot to the mutator on upsert', async () => {
    await store.saveSnapshot('u1', () =>
      snapshot({
        snapshotId: 'u1',
        status: 'pending',
        state: { sessionId: 'sess-4', custom: { counter: 1 } },
      })
    );

    let seenStatus: string | undefined;
    await store.saveSnapshot('u1', (current) => {
      seenStatus = current?.status;
      return { ...current!, status: 'completed' };
    });
    assert.strictEqual(seenStatus, 'pending');

    const read = await store.getSnapshot({ snapshotId: 'u1' });
    assert.strictEqual(read?.status, 'completed');
  });

  it('skips the write when the mutator returns null', async () => {
    const result = await store.saveSnapshot('missing', (current) =>
      current ? { ...current } : null
    );
    assert.strictEqual(result, null);
    assert.strictEqual(
      await store.getSnapshot({ snapshotId: 'missing' }),
      undefined
    );
  });

  it('handles branching: latest leaf wins by pointer', async () => {
    await store.saveSnapshot('b1', () =>
      snapshot({
        snapshotId: 'b1',
        state: { sessionId: 'sess-5', custom: { counter: 1 } },
      })
    );
    // Two children of b1 (a branch, e.g. regenerate).
    await store.saveSnapshot('b2a', () =>
      snapshot({
        snapshotId: 'b2a',
        parentId: 'b1',
        state: { sessionId: 'sess-5', custom: { counter: 20 } },
      })
    );
    await store.saveSnapshot('b2b', () =>
      snapshot({
        snapshotId: 'b2b',
        parentId: 'b1',
        state: { sessionId: 'sess-5', custom: { counter: 21 } },
      })
    );

    // Pointer tracks the most recently created leaf.
    const leaf = await store.getSnapshot({ sessionId: 'sess-5' });
    assert.strictEqual(leaf?.snapshotId, 'b2b');

    // Both branches remain independently addressable.
    const a = await store.getSnapshot({ snapshotId: 'b2a' });
    assert.deepStrictEqual(a?.state.custom, { counter: 20 });
  });

  it('aborting an existing snapshot does not move the leaf pointer', async () => {
    await store.saveSnapshot('a1', () =>
      snapshot({
        snapshotId: 'a1',
        status: 'pending',
        state: { sessionId: 'sess-6', custom: { counter: 1 } },
      })
    );

    await store.saveSnapshot('a1', (current) => ({
      ...current!,
      status: 'aborted',
    }));

    const leaf = await store.getSnapshot({ sessionId: 'sess-6' });
    assert.strictEqual(leaf?.snapshotId, 'a1');
    assert.strictEqual(leaf?.status, 'aborted');
  });

  it('notifies onSnapshotStateChange listeners on status change', async () => {
    await store.saveSnapshot('w1', () =>
      snapshot({
        snapshotId: 'w1',
        status: 'pending',
        state: { sessionId: 'sess-7', custom: { counter: 1 } },
      })
    );

    const aborted = new Promise<void>((resolve) => {
      const unsubscribe = store.onSnapshotStateChange!('w1', (snap) => {
        if (snap.status === 'aborted') {
          unsubscribe?.();
          resolve();
        }
      });
    });

    await store.saveSnapshot('w1', (current) => ({
      ...current!,
      status: 'aborted',
    }));

    await aborted;
  });

  it('creates periodic checkpoints and reconstructs across them', async () => {
    // A small interval forces several checkpoints over a long linear chain.
    const cpStore = makeStore({ checkpointInterval: 5 });

    const turns = 23;
    let parentId: string | undefined;
    for (let i = 0; i < turns; i++) {
      const id = `t${i}`;
      await cpStore.saveSnapshot(id, () =>
        snapshot({
          snapshotId: id,
          parentId,
          state: {
            sessionId: 'long',
            custom: {
              counter: i,
              notes: Array.from({ length: i + 1 }, (_, j) => `n${j}`),
            },
          },
        })
      );
      parentId = id;
    }

    // Leaf reconstructs the full accumulated state.
    const leaf = await cpStore.getSnapshot({ sessionId: 'long' });
    assert.strictEqual(leaf?.snapshotId, `t${turns - 1}`);
    assert.strictEqual(leaf?.state.custom?.counter, turns - 1);
    assert.strictEqual(leaf?.state.custom?.notes?.length, turns);

    // An arbitrary middle snapshot reconstructs correctly across a checkpoint.
    const mid = await cpStore.getSnapshot({ snapshotId: 't12' });
    assert.strictEqual(mid?.state.custom?.counter, 12);
    assert.strictEqual(mid?.state.custom?.notes?.length, 13);

    // Several documents were promoted to checkpoints (root + every 5 turns).
    const colName = (cpStore as any).snapshots.id as string;
    const all = await db
      .collection(colName)
      .where('kind', '==', 'checkpoint')
      .get();
    assert.ok(
      all.size >= turns / 5,
      `expected multiple checkpoints, got ${all.size}`
    );
  });

  it('shards large checkpoint state across multiple documents', async () => {
    // Tiny shard size to force multi-shard storage of a modest state.
    const shardStore = makeStore({ shardSize: 256 });

    const notes = Array.from({ length: 200 }, (_, i) => `note-number-${i}`);
    await shardStore.saveSnapshot('big', () =>
      snapshot({
        snapshotId: 'big',
        state: { sessionId: 'sess-shard', custom: { counter: 1, notes } },
      })
    );

    // Round-trips correctly despite being split into many shards.
    const read = await shardStore.getSnapshot({ snapshotId: 'big' });
    assert.deepStrictEqual(read?.state.custom?.notes, notes);

    // The state really was sharded across more than one document.
    const colName = (shardStore as any).snapshots.id as string;
    const shardCol = `${colName}-shards`;
    const shards = await db.collection(shardCol).get();
    assert.ok(shards.size > 1, `expected multiple shards, got ${shards.size}`);
  });

  it('does not cache full state in the pointer document', async () => {
    await store.saveSnapshot('p1', () =>
      snapshot({
        snapshotId: 'p1',
        state: { sessionId: 'sess-ptr', custom: { counter: 1 } },
      })
    );

    const colName = (store as any).snapshots.id as string;
    const pointer = await db
      .collection(`${colName}-pointers`)
      .doc('sess-ptr')
      .get();
    const data = pointer.data();
    assert.strictEqual(data?.currentSnapshotId, 'p1');
    assert.strictEqual(data?.currentState, undefined);
    assert.strictEqual(typeof data?.checkpointId, 'string');
  });

  it('rejects getSnapshot with both or neither id', async () => {
    await assert.rejects(
      store.getSnapshot({ snapshotId: 'x', sessionId: 'y' }),
      /exactly one/
    );
    await assert.rejects(store.getSnapshot({}), /exactly one/);
  });

  it('rejects getSnapshot with a blank id', async () => {
    // A whitespace-only id is treated as absent and surfaced as a clear
    // INVALID_ARGUMENT rather than being used as an (unusable) document key.
    await assert.rejects(
      store.getSnapshot({ sessionId: '   ' }),
      /exactly one/
    );
    await assert.rejects(store.getSnapshot({ snapshotId: '' }), /exactly one/);
  });

  it('rejects saveSnapshot when state.sessionId is missing', async () => {
    await assert.rejects(
      store.saveSnapshot('no-sess', () =>
        snapshot({ snapshotId: 'no-sess', state: { custom: { counter: 1 } } })
      ),
      /state\.sessionId/
    );
  });

  it('mints a UUID when neither call nor result supplies a snapshotId', async () => {
    const id = await store.saveSnapshot(undefined, () =>
      snapshot({ state: { sessionId: 'sess-uuid', custom: { counter: 1 } } })
    );
    assert.ok(id, 'expected a generated id');
    // A v4 UUID, not one we supplied.
    assert.match(
      id!,
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
    );

    const read = await store.getSnapshot({ snapshotId: id! });
    assert.strictEqual(read?.state.custom?.counter, 1);
  });

  it('promotes an oversized diff to a sharded checkpoint', async () => {
    // A shard size small enough that a single turn's diff exceeds it, forcing
    // the diff -> checkpoint promotion path. A large interval ensures the turn
    // would otherwise be a plain diff.
    const promoteStore = makeStore({ shardSize: 256, checkpointInterval: 100 });

    await promoteStore.saveSnapshot('d1', () =>
      snapshot({
        snapshotId: 'd1',
        state: { sessionId: 'sess-promote', custom: { counter: 0 } },
      })
    );
    // A child whose new state adds a large blob: the resulting diff is bigger
    // than shardSize and must be stored as a checkpoint instead.
    const bigNotes = Array.from({ length: 300 }, (_, i) => `note-${i}`);
    await promoteStore.saveSnapshot('d2', () =>
      snapshot({
        snapshotId: 'd2',
        parentId: 'd1',
        state: {
          sessionId: 'sess-promote',
          custom: { counter: 1, notes: bigNotes },
        },
      })
    );

    // Round-trips correctly.
    const read = await promoteStore.getSnapshot({ snapshotId: 'd2' });
    assert.deepStrictEqual(read?.state.custom?.notes, bigNotes);

    // The promoted document is a checkpoint with no statePatch.
    const colName = (promoteStore as any).snapshots.id as string;
    const raw = await db.collection(colName).doc('d2').get();
    assert.strictEqual(raw.data()?.kind, 'checkpoint');
    assert.strictEqual(raw.data()?.statePatch, undefined);
  });

  it('deletes stale trailing shards when a re-checkpoint shrinks', async () => {
    const shrinkStore = makeStore({ shardSize: 64 });

    // Initial large state spanning several shards.
    const bigNotes = Array.from({ length: 100 }, (_, i) => `note-${i}`);
    await shrinkStore.saveSnapshot('k1', () =>
      snapshot({
        snapshotId: 'k1',
        state: {
          sessionId: 'sess-shrink',
          custom: { counter: 1, notes: bigNotes },
        },
      })
    );

    const colName = (shrinkStore as any).snapshots.id as string;
    const shardCol = `${colName}-shards`;
    const before = await db.collection(shardCol).get();
    assert.ok(before.size > 1, `expected multiple shards, got ${before.size}`);

    // Upsert the same (leaf, checkpoint) snapshot with a much smaller state.
    await shrinkStore.saveSnapshot('k1', (current) => ({
      ...current!,
      state: { sessionId: 'sess-shrink', custom: { counter: 2 } },
    }));

    const after = await db.collection(shardCol).get();
    assert.ok(
      after.size < before.size,
      `expected fewer shards after shrink (before=${before.size}, after=${after.size})`
    );

    // And it still reads back correctly with no leftover/corrupt shards.
    const read = await shrinkStore.getSnapshot({ snapshotId: 'k1' });
    assert.strictEqual(read?.state.custom?.counter, 2);
    assert.strictEqual(read?.state.custom?.notes, undefined);
  });

  it('treats a snapshot with an orphaned parent as a fresh checkpoint', async () => {
    await store.saveSnapshot('o1', () =>
      snapshot({
        snapshotId: 'o1',
        parentId: 'does-not-exist',
        state: { sessionId: 'sess-orphan', custom: { counter: 7 } },
      })
    );

    const read = await store.getSnapshot({ snapshotId: 'o1' });
    assert.strictEqual(read?.state.custom?.counter, 7);

    // It was stored as a checkpoint (root of a new chain), not a diff.
    const colName = (store as any).snapshots.id as string;
    const raw = await db.collection(colName).doc('o1').get();
    assert.strictEqual(raw.data()?.kind, 'checkpoint');
  });

  it('persists and round-trips finishReason, error, and updatedAt', async () => {
    const createdAt = new Date('2026-01-01T00:00:00.000Z').toISOString();
    await store.saveSnapshot('f1', () => ({
      snapshotId: 'f1',
      createdAt,
      event: 'turnEnd',
      status: 'failed',
      finishReason: 'interrupted',
      error: { status: 'INTERNAL', message: 'boom', details: { code: 42 } },
      state: { sessionId: 'sess-fields', custom: { counter: 1 } },
    }));

    const read = await store.getSnapshot({ snapshotId: 'f1' });
    assert.strictEqual(read?.status, 'failed');
    assert.strictEqual(read?.finishReason, 'interrupted');
    assert.deepStrictEqual(read?.error, {
      status: 'INTERNAL',
      message: 'boom',
      details: { code: 42 },
    });
    // updatedAt defaults to createdAt until the snapshot is rewritten.
    assert.strictEqual(read?.updatedAt, createdAt);

    // Upsert with an explicit, later updatedAt.
    const updatedAt = new Date('2026-01-02T00:00:00.000Z').toISOString();
    await store.saveSnapshot('f1', (current) => ({
      ...current!,
      updatedAt,
      status: 'completed',
    }));
    const read2 = await store.getSnapshot({ snapshotId: 'f1' });
    assert.strictEqual(read2?.updatedAt, updatedAt);
    assert.strictEqual(read2?.createdAt, createdAt);
  });

  it('diffs and reconstructs messages and artifacts across a chain', async () => {
    await store.saveSnapshot('m1', () =>
      snapshot({
        snapshotId: 'm1',
        state: {
          sessionId: 'sess-msgs',
          messages: [{ role: 'user', content: [{ text: 'hi' }] }],
          artifacts: [{ name: 'a', parts: [{ text: 'one' }] }],
        },
      })
    );
    await store.saveSnapshot('m2', () =>
      snapshot({
        snapshotId: 'm2',
        parentId: 'm1',
        state: {
          sessionId: 'sess-msgs',
          messages: [
            { role: 'user', content: [{ text: 'hi' }] },
            { role: 'model', content: [{ text: 'hello' }] },
          ],
          artifacts: [
            { name: 'a', parts: [{ text: 'one' }] },
            { name: 'b', parts: [{ text: 'two' }] },
          ],
        },
      })
    );

    const leaf = await store.getSnapshot({ sessionId: 'sess-msgs' });
    assert.strictEqual(leaf?.state.messages?.length, 2);
    assert.strictEqual(leaf?.state.messages?.[1].role, 'model');
    assert.strictEqual(leaf?.state.artifacts?.length, 2);
    assert.strictEqual(leaf?.state.artifacts?.[1].name, 'b');

    // The child stored only a diff, not the full message array.
    const colName = (store as any).snapshots.id as string;
    const raw = await db.collection(colName).doc('m2').get();
    assert.strictEqual(raw.data()?.kind, 'diff');
    assert.ok(Array.isArray(raw.data()?.statePatch));
  });

  it('throws DATA_LOSS when a checkpoint shard is missing', async () => {
    await store.saveSnapshot('g1', () =>
      snapshot({
        snapshotId: 'g1',
        state: { sessionId: 'sess-dataloss', custom: { counter: 1 } },
      })
    );

    // Manually delete the checkpoint's only shard to simulate corruption.
    const colName = (store as any).snapshots.id as string;
    await db.collection(`${colName}-shards`).doc('g1_0').delete();

    await assert.rejects(
      store.getSnapshot({ snapshotId: 'g1' }),
      /missing checkpoint shard/
    );
  });

  it('treats every snapshot as a checkpoint when checkpointInterval is 1', async () => {
    const cp1Store = makeStore({ checkpointInterval: 1 });

    let parentId: string | undefined;
    for (let i = 0; i < 4; i++) {
      const id = `i${i}`;
      await cp1Store.saveSnapshot(id, () =>
        snapshot({
          snapshotId: id,
          parentId,
          state: { sessionId: 'sess-cp1', custom: { counter: i } },
        })
      );
      parentId = id;
    }

    const leaf = await cp1Store.getSnapshot({ sessionId: 'sess-cp1' });
    assert.strictEqual(leaf?.state.custom?.counter, 3);

    // Every document is a checkpoint; none is a diff.
    const colName = (cp1Store as any).snapshots.id as string;
    const diffs = await db
      .collection(colName)
      .where('kind', '==', 'diff')
      .get();
    assert.strictEqual(diffs.size, 0);
  });
});
