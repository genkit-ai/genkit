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
      status: 'completed' as const,
      state: { sessionId },
    }));
    await store.saveSnapshot(second, () => ({
      snapshotId: second,
      parentId: first,
      createdAt: new Date().toISOString(),
      status: 'completed' as const,
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
      status: 'completed',
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

  describe('session pointer', () => {
    it('writes a pointer file tracking the current leaf', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir);
      const sessionId = globalThis.crypto.randomUUID();

      const first = reserveSnapshotId();
      const second = reserveSnapshotId();
      await store.saveSnapshot(first, () => makeSnapshot(first, sessionId));
      await store.saveSnapshot(second, () =>
        makeSnapshot(second, sessionId, first)
      );

      const pointerPath = path.join(
        dir,
        'global',
        '.pointers',
        `${sessionId}.json`
      );
      assert.ok(
        fs.existsSync(pointerPath),
        `expected pointer file at ${pointerPath}`
      );
      const pointer = JSON.parse(fs.readFileSync(pointerPath, 'utf-8'));
      assert.strictEqual(pointer.currentSnapshotId, second);
    });

    it('resolves by sessionId via the pointer fast path', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir);
      const sessionId = globalThis.crypto.randomUUID();

      const first = reserveSnapshotId();
      const second = reserveSnapshotId();
      await store.saveSnapshot(first, () => makeSnapshot(first, sessionId));
      await store.saveSnapshot(second, () =>
        makeSnapshot(second, sessionId, first)
      );

      // Corrupt a non-leaf snapshot. The pointer fast path reads only the leaf,
      // so the lookup must still succeed without scanning every file.
      fs.writeFileSync(
        path.join(dir, 'global', `${first}.json`),
        'not json',
        'utf-8'
      );

      const leaf = await store.getSnapshot({ sessionId });
      assert.strictEqual(leaf?.snapshotId, second);
    });

    it('self-heals a missing pointer via the scan and rewrites it', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir);
      const sessionId = globalThis.crypto.randomUUID();

      const first = reserveSnapshotId();
      const second = reserveSnapshotId();
      await store.saveSnapshot(first, () => makeSnapshot(first, sessionId));
      await store.saveSnapshot(second, () =>
        makeSnapshot(second, sessionId, first)
      );

      const pointerPath = path.join(
        dir,
        'global',
        '.pointers',
        `${sessionId}.json`
      );
      fs.rmSync(pointerPath);

      // Lookup still resolves the leaf by scanning...
      const leaf = await store.getSnapshot({ sessionId });
      assert.strictEqual(leaf?.snapshotId, second);
      // ...and the pointer is rewritten for subsequent fast-path lookups.
      assert.ok(fs.existsSync(pointerPath));
      const pointer = JSON.parse(fs.readFileSync(pointerPath, 'utf-8'));
      assert.strictEqual(pointer.currentSnapshotId, second);
    });

    it('falls back to the scan when the pointer is stale', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir);
      const sessionId = globalThis.crypto.randomUUID();

      const first = reserveSnapshotId();
      const second = reserveSnapshotId();
      await store.saveSnapshot(first, () => makeSnapshot(first, sessionId));
      await store.saveSnapshot(second, () =>
        makeSnapshot(second, sessionId, first)
      );

      // Point at a snapshot that no longer exists.
      const pointerPath = path.join(
        dir,
        'global',
        '.pointers',
        `${sessionId}.json`
      );
      fs.writeFileSync(
        pointerPath,
        JSON.stringify({
          currentSnapshotId: 'does-not-exist',
          updatedAt: new Date().toISOString(),
        }),
        'utf-8'
      );

      const leaf = await store.getSnapshot({ sessionId });
      assert.strictEqual(leaf?.snapshotId, second);
      // The stale pointer is refreshed to the real leaf.
      const pointer = JSON.parse(fs.readFileSync(pointerPath, 'utf-8'));
      assert.strictEqual(pointer.currentSnapshotId, second);
    });

    it('does not write a pointer into the snapshot scan space', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir);
      const sessionId = globalThis.crypto.randomUUID();

      const snapshotId = reserveSnapshotId();
      await store.saveSnapshot(snapshotId, () =>
        makeSnapshot(snapshotId, sessionId)
      );

      // The pointer lives in a hidden sub-directory, not alongside snapshots,
      // so the leaf scan (which only reads *.json files) never sees it.
      const files = fs.readdirSync(path.join(dir, 'global'));
      assert.deepStrictEqual(
        files.filter((f) => f.endsWith('.json')),
        [`${snapshotId}.json`]
      );
    });
  });

  describe('onSnapshotStateChange', () => {
    /**
     * Resolves once `callback` observes a snapshot matching `predicate`, or
     * rejects after `timeoutMs`. Always unsubscribes before settling.
     */
    function waitForSnapshot(
      store: FileSessionStore,
      snapshotId: string,
      predicate: (snap: SessionSnapshot) => boolean,
      options?: { context?: any },
      timeoutMs = 5000
    ): Promise<SessionSnapshot> {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          if (typeof unsubscribe === 'function') unsubscribe();
          reject(
            new Error(
              `Timed out waiting for snapshot ${snapshotId} to match predicate`
            )
          );
        }, timeoutMs);
        const unsubscribe = store.onSnapshotStateChange(
          snapshotId,
          (snap) => {
            if (predicate(snap)) {
              clearTimeout(timer);
              if (typeof unsubscribe === 'function') unsubscribe();
              resolve(snap);
            }
          },
          options
        );
      });
    }

    it('fires the callback when the snapshot file changes', async () => {
      const dir = tmpDir();
      // Use a fast poll interval so the test does not depend on fs.watch
      // delivering events on the host filesystem.
      const store = new FileSessionStore(dir, {
        snapshotWatchPollIntervalMs: 25,
      });
      const sessionId = globalThis.crypto.randomUUID();
      const snapshotId = reserveSnapshotId();

      await store.saveSnapshot(snapshotId, () => ({
        ...makeSnapshot(snapshotId, sessionId),
        status: 'pending',
      }));

      const done = waitForSnapshot(
        store,
        snapshotId,
        (snap) => snap.status === 'aborted'
      );

      // Simulate another process flipping the status to aborted.
      await store.saveSnapshot(snapshotId, (current) => ({
        ...current!,
        status: 'aborted',
      }));

      const snap = await done;
      assert.strictEqual(snap.status, 'aborted');
    });

    it('surfaces the current state immediately on subscribe', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir, {
        snapshotWatchPollIntervalMs: 25,
      });
      const sessionId = globalThis.crypto.randomUUID();
      const snapshotId = reserveSnapshotId();

      await store.saveSnapshot(snapshotId, () => ({
        ...makeSnapshot(snapshotId, sessionId),
        status: 'completed',
      }));

      const snap = await waitForSnapshot(
        store,
        snapshotId,
        (s) => s.status === 'completed'
      );
      assert.strictEqual(snap.snapshotId, snapshotId);
    });

    it('stops firing after unsubscribe', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir, {
        snapshotWatchPollIntervalMs: 25,
      });
      const sessionId = globalThis.crypto.randomUUID();
      const snapshotId = reserveSnapshotId();

      await store.saveSnapshot(snapshotId, () =>
        makeSnapshot(snapshotId, sessionId)
      );

      let calls = 0;
      const unsubscribe = store.onSnapshotStateChange(snapshotId, () => {
        calls++;
      });
      // Wait for the initial emit, then unsubscribe.
      await new Promise((r) => setTimeout(r, 100));
      if (typeof unsubscribe === 'function') unsubscribe();
      const callsAfterUnsubscribe = calls;

      // Further writes must not trigger the callback.
      await store.saveSnapshot(snapshotId, (current) => ({
        ...current!,
        status: 'aborted',
      }));
      await new Promise((r) => setTimeout(r, 200));

      assert.strictEqual(calls, callsAfterUnsubscribe);
    });

    it('scopes the watch to the per-tenant prefix', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir, {
        snapshotWatchPollIntervalMs: 25,
        snapshotPathPrefix: (options) =>
          (options?.context?.auth as any)?.uid ?? 'anon',
      });
      const sessionId = globalThis.crypto.randomUUID();
      const snapshotId = reserveSnapshotId();
      const aliceCtx = { context: { auth: { uid: 'alice' } } };

      await store.saveSnapshot(
        snapshotId,
        () => ({
          ...makeSnapshot(snapshotId, sessionId),
          status: 'pending',
        }),
        aliceCtx
      );

      const done = waitForSnapshot(
        store,
        snapshotId,
        (snap) => snap.status === 'aborted',
        aliceCtx
      );

      await store.saveSnapshot(
        snapshotId,
        (current) => ({ ...current!, status: 'aborted' }),
        aliceCtx
      );

      const snap = await done;
      assert.strictEqual(snap.status, 'aborted');
    });
  });

  describe('FileSessionStore snapshotId validation', () => {
    function tmpDir(): string {
      return fs.mkdtempSync(path.join(os.tmpdir(), 'genkit-sessions-'));
    }

    const traversalIds = [
      '../escape',
      '../../escape',
      'foo/bar',
      'foo\\bar',
      '..',
      '.',
    ];

    for (const badId of traversalIds) {
      it(`rejects getSnapshot for unsafe snapshotId ${JSON.stringify(badId)}`, async () => {
        const store = new FileSessionStore(tmpDir());
        await assert.rejects(
          () => store.getSnapshot({ snapshotId: badId }),
          /Invalid snapshotId/
        );
      });

      it(`rejects saveSnapshot for unsafe snapshotId ${JSON.stringify(badId)}`, async () => {
        const store = new FileSessionStore(tmpDir());
        await assert.rejects(
          () =>
            store.saveSnapshot(badId, (current) => ({
              ...current,
              snapshotId: badId,
              createdAt: new Date().toISOString(),
              state: { custom: {} },
              status: 'completed' as const,
            })),
          /Invalid snapshotId/
        );
      });
    }

    it('does not write outside the store directory for a traversal id', async () => {
      const dir = tmpDir();
      const store = new FileSessionStore(dir);
      await assert.rejects(
        () =>
          store.saveSnapshot('../../escaped', (current) => ({
            ...current,
            snapshotId: '../../escaped',
            createdAt: new Date().toISOString(),
            state: { custom: {} },
            status: 'completed' as const,
          })),
        /Invalid snapshotId/
      );
      // Nothing escaped to the parent of the store dir.
      assert.ok(!fs.existsSync(path.join(dir, '..', 'escaped.json')));
    });

    it('accepts a plain basename snapshotId', async () => {
      const store = new FileSessionStore(tmpDir());
      const id = reserveSnapshotId();
      const saved = await store.saveSnapshot(id, () => ({
        snapshotId: id,
        createdAt: new Date().toISOString(),
        state: { custom: { ok: true } },
        status: 'completed' as const,
      }));
      assert.strictEqual(saved, id);
      const snap = await store.getSnapshot({ snapshotId: id });
      assert.deepStrictEqual(snap?.state.custom, { ok: true });
    });
  });
});
