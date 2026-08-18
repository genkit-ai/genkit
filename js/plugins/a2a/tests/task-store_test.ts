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

import assert from 'node:assert';
import { describe, it } from 'node:test';
import { InMemoryA2ATaskStore } from '../src/task-store.js';

describe('InMemoryA2ATaskStore', () => {
  it('returns undefined for an unknown task', async () => {
    const store = new InMemoryA2ATaskStore();
    assert.strictEqual(await store.get('nope'), undefined);
  });

  it('sets and gets a task record', async () => {
    const store = new InMemoryA2ATaskStore();
    await store.set('t-1', { contextId: 'ctx-1', snapshotId: 'snap-2' });
    assert.deepStrictEqual(await store.get('t-1'), {
      contextId: 'ctx-1',
      snapshotId: 'snap-2',
    });
  });

  it('overwrites an existing record', async () => {
    const store = new InMemoryA2ATaskStore();
    await store.set('t-1', { contextId: 'ctx-1', snapshotId: 'snap-1' });
    await store.set('t-1', { contextId: 'ctx-1', snapshotId: 'snap-2' });
    assert.strictEqual((await store.get('t-1'))?.snapshotId, 'snap-2');
  });

  it('deletes a record', async () => {
    const store = new InMemoryA2ATaskStore();
    await store.set('t-1', { contextId: 'ctx-1', snapshotId: 'snap-1' });
    await store.delete('t-1');
    assert.strictEqual(await store.get('t-1'), undefined);
  });

  it('evicts the oldest entry past the cap', async () => {
    const store = new InMemoryA2ATaskStore(2);
    await store.set('a', { contextId: 'c', snapshotId: 's-a' });
    await store.set('b', { contextId: 'c', snapshotId: 's-b' });
    await store.set('c', { contextId: 'c', snapshotId: 's-c' });
    // 'a' was the oldest and should have been evicted.
    assert.strictEqual(await store.get('a'), undefined);
    assert.ok(await store.get('b'));
    assert.ok(await store.get('c'));
  });
});
