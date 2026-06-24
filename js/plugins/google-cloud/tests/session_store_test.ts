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
import { AgentAPI, genkit, type SessionSnapshotInput } from 'genkit/beta';
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

  /** The base collection name a store was configured with. */
  function baseCollection(s: FirestoreSessionStore<Custom>): string {
    return (s as any).collection as string;
  }

  /** The (per-tenant) snapshots subcollection used by the store. */
  function snapshotsCol(s: FirestoreSessionStore<Custom>, prefix = 'global') {
    return db.collection(baseCollection(s)).doc(prefix).collection('snapshots');
  }

  /** The (per-tenant) shards subcollection used by the store. */
  function shardsCol(s: FirestoreSessionStore<Custom>, prefix = 'global') {
    return db
      .collection(`${baseCollection(s)}-shards`)
      .doc(prefix)
      .collection('shards');
  }

  /** The (per-tenant) pointers subcollection used by the store. */
  function pointersCol(s: FirestoreSessionStore<Custom>, prefix = 'global') {
    return db
      .collection(`${baseCollection(s)}-pointers`)
      .doc(prefix)
      .collection('pointers');
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
    assert.strictEqual(read.state?.sessionId, 'sess-1');
    assert.deepStrictEqual(read.state?.custom, { counter: 1 });
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
    assert.deepStrictEqual(leaf.state?.custom, { counter: 2 });
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
    assert.deepStrictEqual(mid?.state?.custom, {
      counter: 2,
      notes: ['a', 'b'],
    });

    // Leaf reconstructs correctly via sessionId.
    const leaf = await store.getSnapshot({ sessionId: 'sess-3' });
    assert.deepStrictEqual(leaf?.state?.custom, {
      counter: 3,
      notes: ['a', 'b', 'c'],
    });

    // The stored document only persists a diff, not the full state.
    const raw = await snapshotsCol(store).doc('c3').get();
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
    assert.deepStrictEqual(a?.state?.custom, { counter: 20 });
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
    assert.strictEqual(leaf?.state?.custom?.counter, turns - 1);
    assert.strictEqual(leaf?.state?.custom?.notes?.length, turns);

    // An arbitrary middle snapshot reconstructs correctly across a checkpoint.
    const mid = await cpStore.getSnapshot({ snapshotId: 't12' });
    assert.strictEqual(mid?.state?.custom?.counter, 12);
    assert.strictEqual(mid?.state?.custom?.notes?.length, 13);

    // Several documents were promoted to checkpoints (root + every 5 turns).
    const all = await snapshotsCol(cpStore)
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
    assert.deepStrictEqual(read?.state?.custom?.notes, notes);

    // The state really was sharded across more than one document.
    const shards = await shardsCol(shardStore).get();
    assert.ok(shards.size > 1, `expected multiple shards, got ${shards.size}`);
  });

  it('does not cache full state in the pointer document', async () => {
    await store.saveSnapshot('p1', () =>
      snapshot({
        snapshotId: 'p1',
        state: { sessionId: 'sess-ptr', custom: { counter: 1 } },
      })
    );

    const pointer = await pointersCol(store).doc('sess-ptr').get();
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

  it('rejects saveSnapshot when sessionId is missing', async () => {
    await assert.rejects(
      store.saveSnapshot('no-sess', () =>
        snapshot({ snapshotId: 'no-sess', state: { custom: { counter: 1 } } })
      ),
      /sessionId/
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
    assert.strictEqual(read?.state?.custom?.counter, 1);
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
    assert.deepStrictEqual(read?.state?.custom?.notes, bigNotes);

    // The promoted document is a checkpoint with no statePatch.
    const raw = await snapshotsCol(promoteStore).doc('d2').get();
    assert.strictEqual(raw.data()?.kind, 'checkpoint');
    assert.strictEqual(raw.data()?.statePatch, undefined);
  });

  it('promotes an oversized diff upsert to a sharded checkpoint', async () => {
    // A leaf that starts as a small diff can grow past the limit on an
    // in-place rewrite; the upsert path must promote it to a checkpoint rather
    // than write an oversized statePatch.
    const promoteStore = makeStore({ shardSize: 256, checkpointInterval: 100 });

    await promoteStore.saveSnapshot('e1', () =>
      snapshot({
        snapshotId: 'e1',
        state: { sessionId: 'sess-upsert-promote', custom: { counter: 0 } },
      })
    );
    // A small child diff (well under shardSize).
    await promoteStore.saveSnapshot('e2', () =>
      snapshot({
        snapshotId: 'e2',
        parentId: 'e1',
        state: {
          sessionId: 'sess-upsert-promote',
          custom: { counter: 1 },
        },
      })
    );

    // It started life as a diff.
    const before = await snapshotsCol(promoteStore).doc('e2').get();
    assert.strictEqual(before.data()?.kind, 'diff');

    // Rewrite the leaf in place with a much larger state: the diff would now
    // exceed shardSize and must be promoted to a sharded checkpoint.
    const bigNotes = Array.from({ length: 300 }, (_, i) => `note-${i}`);
    await promoteStore.saveSnapshot('e2', (current) => ({
      ...current!,
      state: {
        sessionId: 'sess-upsert-promote',
        custom: { counter: 2, notes: bigNotes },
      },
    }));

    // Round-trips correctly.
    const read = await promoteStore.getSnapshot({ snapshotId: 'e2' });
    assert.deepStrictEqual(read?.state?.custom?.notes, bigNotes);
    assert.strictEqual(read?.state?.custom?.counter, 2);

    // The promoted document is now a checkpoint with no statePatch, and its
    // state was sharded across more than one document.
    const after = await snapshotsCol(promoteStore).doc('e2').get();
    assert.strictEqual(after.data()?.kind, 'checkpoint');
    assert.strictEqual(after.data()?.statePatch, undefined);
    assert.ok(
      after.data()?.checkpointShardCount > 1,
      `expected multiple shards, got ${after.data()?.checkpointShardCount}`
    );
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

    const before = await shardsCol(shrinkStore).get();
    assert.ok(before.size > 1, `expected multiple shards, got ${before.size}`);

    // Upsert the same (leaf, checkpoint) snapshot with a much smaller state.
    await shrinkStore.saveSnapshot('k1', (current) => ({
      ...current!,
      state: { sessionId: 'sess-shrink', custom: { counter: 2 } },
    }));

    const after = await shardsCol(shrinkStore).get();
    assert.ok(
      after.size < before.size,
      `expected fewer shards after shrink (before=${before.size}, after=${after.size})`
    );

    // And it still reads back correctly with no leftover/corrupt shards.
    const read = await shrinkStore.getSnapshot({ snapshotId: 'k1' });
    assert.strictEqual(read?.state?.custom?.counter, 2);
    assert.strictEqual(read?.state?.custom?.notes, undefined);
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
    assert.strictEqual(read?.state?.custom?.counter, 7);

    // It was stored as a checkpoint (root of a new chain), not a diff.
    const raw = await snapshotsCol(store).doc('o1').get();
    assert.strictEqual(raw.data()?.kind, 'checkpoint');
  });

  it('persists and round-trips finishReason, error, heartbeatAt, and updatedAt', async () => {
    const createdAt = new Date('2026-01-01T00:00:00.000Z').toISOString();
    const heartbeatAt = new Date('2026-01-01T00:00:05.000Z').toISOString();
    await store.saveSnapshot('f1', () => ({
      snapshotId: 'f1',
      createdAt,
      status: 'pending',
      finishReason: 'interrupted',
      heartbeatAt,
      error: { status: 'INTERNAL', message: 'boom', details: { code: 42 } },
      state: { sessionId: 'sess-fields', custom: { counter: 1 } },
    }));

    const read = await store.getSnapshot({ snapshotId: 'f1' });
    assert.strictEqual(read?.status, 'pending');
    assert.strictEqual(read?.finishReason, 'interrupted');
    assert.strictEqual(read?.heartbeatAt, heartbeatAt);
    assert.deepStrictEqual(read?.error, {
      status: 'INTERNAL',
      message: 'boom',
      details: { code: 42 },
    });
    // updatedAt defaults to createdAt until the snapshot is rewritten.
    assert.strictEqual(read?.updatedAt, createdAt);

    // Upsert with an explicit, later updatedAt and a refreshed heartbeat.
    const updatedAt = new Date('2026-01-02T00:00:00.000Z').toISOString();
    const heartbeatAt2 = new Date('2026-01-02T00:00:05.000Z').toISOString();
    await store.saveSnapshot('f1', (current) => ({
      ...current!,
      updatedAt,
      heartbeatAt: heartbeatAt2,
      status: 'completed',
    }));
    const read2 = await store.getSnapshot({ snapshotId: 'f1' });
    assert.strictEqual(read2?.updatedAt, updatedAt);
    assert.strictEqual(read2?.createdAt, createdAt);
    assert.strictEqual(read2?.heartbeatAt, heartbeatAt2);
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
    assert.strictEqual(leaf?.state?.messages?.length, 2);
    assert.strictEqual(leaf?.state?.messages?.[1].role, 'model');
    assert.strictEqual(leaf?.state?.artifacts?.length, 2);
    assert.strictEqual(leaf?.state?.artifacts?.[1].name, 'b');

    // The child stored only a diff, not the full message array.
    const raw = await snapshotsCol(store).doc('m2').get();
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
    await shardsCol(store).doc('g1_0').delete();

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
    assert.strictEqual(leaf?.state?.custom?.counter, 3);

    // Every document is a checkpoint; none is a diff.
    const diffs = await snapshotsCol(cp1Store)
      .where('kind', '==', 'diff')
      .get();
    assert.strictEqual(diffs.size, 0);
  });

  describe('snapshotPathPrefix (multi-tenant isolation)', () => {
    // A prefix derived from the call context's user id, mirroring how an app
    // would scope snapshots to the authenticated user.
    const byUser = (options?: { context?: { auth?: { uid?: string } } }) =>
      options?.context?.auth?.uid ?? 'anonymous';

    const ctx = (uid: string) => ({ context: { auth: { uid } } });

    it('isolates snapshots per tenant: a snapshotId is not readable across tenants', async () => {
      const tenantStore = makeStore({ snapshotPathPrefix: byUser });

      await tenantStore.saveSnapshot(
        'shared-id',
        () =>
          snapshot({
            snapshotId: 'shared-id',
            state: { sessionId: 'sess-a', custom: { counter: 1 } },
          }),
        ctx('alice')
      );

      // Alice can read her own snapshot.
      const asAlice = await tenantStore.getSnapshot({
        snapshotId: 'shared-id',
        context: ctx('alice').context,
      });
      assert.strictEqual(asAlice?.state?.custom?.counter, 1);

      // Bob, with the very same snapshotId, sees nothing.
      const asBob = await tenantStore.getSnapshot({
        snapshotId: 'shared-id',
        context: ctx('bob').context,
      });
      assert.strictEqual(asBob, undefined);
    });

    it('isolates sessionId lookups per tenant', async () => {
      const tenantStore = makeStore({ snapshotPathPrefix: byUser });

      await tenantStore.saveSnapshot(
        'a1',
        () =>
          snapshot({
            snapshotId: 'a1',
            state: { sessionId: 'shared-sess', custom: { counter: 11 } },
          }),
        ctx('alice')
      );

      // Same sessionId, different tenant: no leaf.
      const asBob = await tenantStore.getSnapshot({
        sessionId: 'shared-sess',
        context: ctx('bob').context,
      });
      assert.strictEqual(asBob, undefined);

      // Alice resolves her own leaf.
      const asAlice = await tenantStore.getSnapshot({
        sessionId: 'shared-sess',
        context: ctx('alice').context,
      });
      assert.strictEqual(asAlice?.snapshotId, 'a1');
      assert.strictEqual(asAlice?.state?.custom?.counter, 11);
    });

    it('writes documents under the tenant-scoped subcollection', async () => {
      const tenantStore = makeStore({ snapshotPathPrefix: byUser });

      await tenantStore.saveSnapshot(
        'doc-1',
        () =>
          snapshot({
            snapshotId: 'doc-1',
            state: { sessionId: 'sess-scope', custom: { counter: 1 } },
          }),
        ctx('carol')
      );

      // Present under carol's prefix.
      const scoped = await snapshotsCol(tenantStore, 'carol')
        .doc('doc-1')
        .get();
      assert.ok(scoped.exists);

      // Absent under the default 'global' prefix.
      const global = await snapshotsCol(tenantStore, 'global')
        .doc('doc-1')
        .get();
      assert.strictEqual(global.exists, false);
    });
  });

  describe('agent integration (defineAgent + echo model)', () => {
    /**
     * Creates a Firestore-backed store on a fresh collection and registers it
     * for cleanup. Returns the untyped store (State = unknown) suitable for
     * wiring into an agent.
     */
    function makeAgentStore(): FirestoreSessionStore {
      const collection = `agent-sessions-${Date.now()}-${Math.floor(
        Math.random() * 1e6
      )}`;
      createdCollections.push(
        collection,
        `${collection}-pointers`,
        `${collection}-shards`
      );
      return new FirestoreSessionStore({ db, collection });
    }

    /**
     * Defines a genkit/beta instance with a simple echo model that replies
     * with "echo: <last user text>".
     */
    function makeAi() {
      const ai = genkit({});
      ai.defineModel({ name: 'echo' }, async (request) => {
        const lastUser = [...request.messages]
          .reverse()
          .find((m) => m.role === 'user');
        const text =
          lastUser?.content.map((p) => (p as any).text ?? '').join('') ?? '';
        return {
          message: { role: 'model', content: [{ text: `echo: ${text}` }] },
          finishReason: 'stop',
        };
      });
      return ai;
    }

    /** Drives a single agent turn to completion and returns its output. */
    async function runTurn(
      agent: AgentAPI,
      input: { message: { role: 'user'; content: { text: string }[] } },
      init: Record<string, unknown> = {}
    ) {
      const session = agent.chat(init);
      const { stream, response } = session.sendStream(input);
      for await (const _ of stream) {
        // drain the stream
      }
      return {
        response: await response,
        snapshotId: session.snapshotId,
        sessionId: session.sessionId,
      };
    }

    it('persists a multi-turn conversation through Firestore via the agent API', async () => {
      const agentStore = makeAgentStore();
      const ai = makeAi();
      const agent = ai.defineAgent({
        name: 'echoAgent',
        model: 'echo',
        system: 'You are an echo bot.',
        store: agentStore,
      });

      // Turn 1: a fresh, server-managed session.
      const { snapshotId: snapshotId1, sessionId } = await runTurn(agent, {
        message: { role: 'user', content: [{ text: 'hello' }] },
      });
      assert.ok(sessionId, 'turn 1 should mint a sessionId');
      assert.ok(snapshotId1, 'turn 1 should persist a snapshot');

      // The snapshot is readable straight from Firestore and contains the
      // accumulated [user, model] history with the echoed reply.
      const snap1 = await agentStore.getSnapshot({ snapshotId: snapshotId1 });
      assert.ok(snap1, 'turn 1 snapshot should exist in Firestore');
      assert.strictEqual(snap1!.state?.sessionId, sessionId);
      const msgs1 = snap1!.state?.messages ?? [];
      assert.strictEqual(msgs1.length, 2);
      assert.strictEqual(msgs1[0].role, 'user');
      assert.strictEqual(msgs1[0].content[0].text, 'hello');
      assert.strictEqual(msgs1[1].role, 'model');
      assert.strictEqual(msgs1[1].content[0].text, 'echo: hello');

      // Turn 2: resume from the persisted snapshot. History should accumulate.
      const { response: out2 } = await runTurn(
        agent,
        { message: { role: 'user', content: [{ text: 'world' }] } },
        { snapshotId: snapshotId1 }
      );
      assert.strictEqual(
        out2.sessionId,
        sessionId,
        'turn 2 should preserve the sessionId across turns'
      );

      // Resolving the latest leaf by sessionId reflects the full conversation.
      const leaf = await agentStore.getSnapshot({ sessionId });
      assert.ok(leaf, 'a leaf snapshot should be resolvable by sessionId');
      const msgs2 = leaf!.state?.messages ?? [];
      assert.strictEqual(msgs2.length, 4, 'history should accumulate to 4');
      assert.deepStrictEqual(
        msgs2.map((m) => m.content.map((p: any) => p.text).join('')),
        ['hello', 'echo: hello', 'world', 'echo: world']
      );
      // System messages are not persisted into stored history.
      assert.strictEqual(
        msgs2.filter((m) => m.role === 'system').length,
        0,
        'system messages should not be persisted'
      );
    });
  });
});
