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

import { initNodeFeatures } from '@genkit-ai/core/node';
import * as assert from 'assert';
import * as fs from 'fs';
import { describe, it } from 'node:test';
import * as os from 'os';
import * as path from 'path';

import {
  FileSessionStore,
  InMemorySessionStore,
} from '../src/session-stores.js';
import { reserveSnapshotId, type SessionSnapshot } from '../src/session.js';

initNodeFeatures();

describe('InMemorySessionStore', () => {
  it('should save and get snapshots', async () => {
    const store = new InMemorySessionStore<{ foo: string }>();
    const snapshot = {
      snapshotId: 'snap-123',
      createdAt: new Date().toISOString(),
      event: 'turnEnd' as const,
      state: { custom: { foo: 'bar' } },
    };
    await store.saveSnapshot('snap-123', () => snapshot);

    const got = await store.getSnapshot({ snapshotId: 'snap-123' });
    assert.deepStrictEqual(got, snapshot);
  });

  it('should return undefined for missing snapshot', async () => {
    const store = new InMemorySessionStore();
    const got = await store.getSnapshot({ snapshotId: 'missing' });
    assert.strictEqual(got, undefined);
  });

  it('should deep copy on save and get', async () => {
    const store = new InMemorySessionStore<{ foo: string }>();
    const state = { foo: 'bar' };
    const snapshot = {
      snapshotId: 'snap-123',
      createdAt: new Date().toISOString(),
      event: 'turnEnd' as const,
      state: { custom: state },
    };
    await store.saveSnapshot('snap-123', () => snapshot);

    // Mutate local state
    state.foo = 'baz';

    const got = await store.getSnapshot({ snapshotId: 'snap-123' });
    assert.strictEqual(got?.state.custom?.foo, 'bar');
  });

  it('resolves the latest leaf snapshot by sessionId', async () => {
    const store = new InMemorySessionStore();
    const sessionId = globalThis.crypto.randomUUID();

    const first = reserveSnapshotId();
    const second = reserveSnapshotId();
    await store.saveSnapshot(first, () => ({
      snapshotId: first,
      createdAt: new Date().toISOString(),
      event: 'turnEnd' as const,
      status: 'done' as const,
      state: { sessionId },
    }));
    await store.saveSnapshot(second, () => ({
      snapshotId: second,
      parentId: first,
      createdAt: new Date().toISOString(),
      event: 'turnEnd' as const,
      status: 'done' as const,
      state: { sessionId },
    }));

    const leaf = await store.getSnapshot({ sessionId });
    assert.strictEqual(leaf?.snapshotId, second);
  });
});

describe('FileSessionStore', () => {
  function tmpDir(): string {
    return fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-file-store-'));
  }

  function makeSnapshot(
    snapshotId: string,
    sessionId: string,
    parentId?: string
  ): SessionSnapshot {
    return {
      snapshotId,
      parentId,
      createdAt: new Date().toISOString(),
      event: 'turnEnd',
      status: 'done',
      state: { sessionId },
    };
  }

  it('stores each snapshot as a flat <snapshotId>.json file', async () => {
    const dir = tmpDir();
    const store = new FileSessionStore(dir);
    const snapshotId = reserveSnapshotId();
    const sessionId = globalThis.crypto.randomUUID();

    await store.saveSnapshot(snapshotId, () =>
      makeSnapshot(snapshotId, sessionId)
    );

    // The file lives flat under the default "global" prefix dir.
    const filePath = path.join(dir, 'global', `${snapshotId}.json`);
    assert.ok(fs.existsSync(filePath), `expected file at ${filePath}`);

    const got = await store.getSnapshot({ snapshotId });
    assert.strictEqual(got?.snapshotId, snapshotId);
    assert.strictEqual(got?.state.sessionId, sessionId);
  });

  it('resolves the latest leaf snapshot by sessionId', async () => {
    const dir = tmpDir();
    const store = new FileSessionStore(dir);
    const sessionId = globalThis.crypto.randomUUID();

    const first = reserveSnapshotId();
    const second = reserveSnapshotId();
    await store.saveSnapshot(first, () => makeSnapshot(first, sessionId));
    await store.saveSnapshot(second, () =>
      makeSnapshot(second, sessionId, first)
    );

    const leaf = await store.getSnapshot({ sessionId });
    assert.strictEqual(leaf?.snapshotId, second);
  });

  it('returns the latest leaf for a branching history by default', async () => {
    const dir = tmpDir();
    const store = new FileSessionStore(dir);
    const sessionId = globalThis.crypto.randomUUID();

    const root = reserveSnapshotId();
    const branchA = reserveSnapshotId();
    const branchB = reserveSnapshotId();
    await store.saveSnapshot(root, () => ({
      ...makeSnapshot(root, sessionId),
      createdAt: '2026-01-01T00:00:00.000Z',
    }));
    await store.saveSnapshot(branchA, () => ({
      ...makeSnapshot(branchA, sessionId, root),
      createdAt: '2026-01-01T00:00:01.000Z',
    }));
    await store.saveSnapshot(branchB, () => ({
      ...makeSnapshot(branchB, sessionId, root),
      createdAt: '2026-01-01T00:00:02.000Z',
    }));

    // By default a branched lookup resolves to the most-recently created leaf.
    const leaf = await store.getSnapshot({ sessionId });
    assert.strictEqual(leaf?.snapshotId, branchB);
  });

  it('throws on branching history when rejectBranchingSessions is enabled', async () => {
    const dir = tmpDir();
    const store = new FileSessionStore(dir, { rejectBranchingSessions: true });
    const sessionId = globalThis.crypto.randomUUID();

    const root = reserveSnapshotId();
    const branchA = reserveSnapshotId();
    const branchB = reserveSnapshotId();
    await store.saveSnapshot(root, () => makeSnapshot(root, sessionId));
    await store.saveSnapshot(branchA, () =>
      makeSnapshot(branchA, sessionId, root)
    );
    await store.saveSnapshot(branchB, () =>
      makeSnapshot(branchB, sessionId, root)
    );

    await assert.rejects(
      () => store.getSnapshot({ sessionId }),
      /branching snapshots/
    );
  });

  it('isolates snapshots per tenant via snapshotPathPrefix', async () => {
    const dir = tmpDir();
    const store = new FileSessionStore(dir, {
      // Derive the tenant prefix from the auth context (e.g. user id).
      snapshotPathPrefix: (options) =>
        (options?.context?.auth as any)?.uid ?? 'anon',
    });

    const sessionId = globalThis.crypto.randomUUID();
    const aliceSnap = reserveSnapshotId();
    const bobSnap = reserveSnapshotId();

    const aliceCtx = { context: { auth: { uid: 'alice' } } };
    const bobCtx = { context: { auth: { uid: 'bob' } } };

    await store.saveSnapshot(
      aliceSnap,
      () => makeSnapshot(aliceSnap, sessionId),
      aliceCtx
    );
    await store.saveSnapshot(
      bobSnap,
      () => makeSnapshot(bobSnap, sessionId),
      bobCtx
    );

    // Each tenant gets its own sub-directory.
    assert.ok(
      fs.existsSync(path.join(dir, 'alice', `${aliceSnap}.json`)),
      "alice's snapshot should be under her prefix"
    );
    assert.ok(
      fs.existsSync(path.join(dir, 'bob', `${bobSnap}.json`)),
      "bob's snapshot should be under his prefix"
    );

    // A tenant can only see snapshots scoped to their own prefix.
    assert.ok(await store.getSnapshot({ snapshotId: aliceSnap, ...aliceCtx }));
    assert.strictEqual(
      await store.getSnapshot({ snapshotId: aliceSnap, ...bobCtx }),
      undefined
    );

    // sessionId lookups are likewise scoped per tenant.
    assert.strictEqual(
      (await store.getSnapshot({ sessionId, ...aliceCtx }))?.snapshotId,
      aliceSnap
    );
    assert.strictEqual(
      (await store.getSnapshot({ sessionId, ...bobCtx }))?.snapshotId,
      bobSnap
    );
  });

  it('prunes snapshots beyond maxPersistedChainLength', async () => {
    const dir = tmpDir();
    const store = new FileSessionStore(dir, { maxPersistedChainLength: 2 });
    const sessionId = globalThis.crypto.randomUUID();

    const a = reserveSnapshotId();
    const b = reserveSnapshotId();
    const c = reserveSnapshotId();
    await store.saveSnapshot(a, () => makeSnapshot(a, sessionId));
    await store.saveSnapshot(b, () => makeSnapshot(b, sessionId, a));
    await store.saveSnapshot(c, () => makeSnapshot(c, sessionId, b));

    // Only the two most recent snapshots in the chain are retained.
    assert.ok(await store.getSnapshot({ snapshotId: c }));
    assert.ok(await store.getSnapshot({ snapshotId: b }));
    assert.strictEqual(await store.getSnapshot({ snapshotId: a }), undefined);
  });
});
