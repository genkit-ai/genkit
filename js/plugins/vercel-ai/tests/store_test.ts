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

import * as assert from 'assert';
import { afterEach, beforeEach, describe, it } from 'node:test';
import {
  InMemorySnapshotStore,
  LocalStorageSnapshotStore,
} from '../src/store.js';

describe('InMemorySnapshotStore', () => {
  it('returns undefined for unknown chats', () => {
    const store = new InMemorySnapshotStore();
    assert.strictEqual(store.get('nope'), undefined);
  });

  it('stores and retrieves a snapshot', () => {
    const store = new InMemorySnapshotStore();
    store.set('chat', {
      snapshotId: 's2',
      previousSnapshotId: 's1',
      interrupted: true,
    });
    assert.deepStrictEqual(store.get('chat'), {
      snapshotId: 's2',
      previousSnapshotId: 's1',
      interrupted: true,
    });
  });

  it('overwrites on subsequent set', () => {
    const store = new InMemorySnapshotStore();
    store.set('chat', { snapshotId: 's1' });
    store.set('chat', { snapshotId: 's2', previousSnapshotId: 's1' });
    assert.deepStrictEqual(store.get('chat'), {
      snapshotId: 's2',
      previousSnapshotId: 's1',
    });
  });

  it('deletes a snapshot', () => {
    const store = new InMemorySnapshotStore();
    store.set('chat', { snapshotId: 's1' });
    store.delete('chat');
    assert.strictEqual(store.get('chat'), undefined);
  });

  it('keeps chats independent', () => {
    const store = new InMemorySnapshotStore();
    store.set('a', { snapshotId: 'sa' });
    store.set('b', { snapshotId: 'sb' });
    assert.strictEqual(store.get('a')?.snapshotId, 'sa');
    assert.strictEqual(store.get('b')?.snapshotId, 'sb');
  });
});

describe('LocalStorageSnapshotStore', () => {
  // Minimal in-memory localStorage stand-in.
  const makeLocalStorage = (): Storage => {
    const map = new Map<string, string>();
    return {
      get length() {
        return map.size;
      },
      clear: () => map.clear(),
      getItem: (k: string) => (map.has(k) ? map.get(k)! : null),
      key: (i: number) => Array.from(map.keys())[i] ?? null,
      removeItem: (k: string) => map.delete(k),
      setItem: (k: string, v: string) => map.set(k, v),
    } as Storage;
  };

  let original: PropertyDescriptor | undefined;

  beforeEach(() => {
    original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    Object.defineProperty(globalThis, 'localStorage', {
      value: makeLocalStorage(),
      configurable: true,
      writable: true,
    });
  });

  afterEach(() => {
    if (original) {
      Object.defineProperty(globalThis, 'localStorage', original);
    } else {
      // @ts-expect-error cleanup test shim
      delete globalThis.localStorage;
    }
  });

  it('persists and retrieves a snapshot via localStorage', () => {
    const store = new LocalStorageSnapshotStore();
    store.set('chat', { snapshotId: 's2', previousSnapshotId: 's1' });
    assert.deepStrictEqual(store.get('chat'), {
      snapshotId: 's2',
      previousSnapshotId: 's1',
    });
    // A fresh instance reads the same persisted value.
    const store2 = new LocalStorageSnapshotStore();
    assert.strictEqual(store2.get('chat')?.snapshotId, 's2');
  });

  it('namespaces keys by prefix', () => {
    const store = new LocalStorageSnapshotStore({ prefix: 'myapp:' });
    store.set('chat', { snapshotId: 's1' });
    assert.ok(globalThis.localStorage.getItem('myapp:chat'));
    assert.strictEqual(
      globalThis.localStorage.getItem('genkit-chat:chat'),
      null
    );
  });

  it('deletes a snapshot', () => {
    const store = new LocalStorageSnapshotStore();
    store.set('chat', { snapshotId: 's1' });
    store.delete('chat');
    assert.strictEqual(store.get('chat'), undefined);
  });

  it('returns undefined for unknown chats', () => {
    const store = new LocalStorageSnapshotStore();
    assert.strictEqual(store.get('missing'), undefined);
  });

  it('degrades to no-ops when localStorage is unavailable', () => {
    // @ts-expect-error simulate SSR / unavailable storage
    delete globalThis.localStorage;
    const store = new LocalStorageSnapshotStore();
    // Should not throw and should behave as empty.
    assert.doesNotThrow(() => store.set('chat', { snapshotId: 's1' }));
    assert.strictEqual(store.get('chat'), undefined);
    assert.doesNotThrow(() => store.delete('chat'));
  });
});
