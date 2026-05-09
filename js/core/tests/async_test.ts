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

import * as assert from 'assert';
import { describe, it } from 'node:test';
import { AsyncTaskQueue, LazyPromise } from '../src/async';
import { sleep } from './utils';

describe('AsyncTaskQueue', () => {
  it('should execute tasks in order', async () => {
    const queue = new AsyncTaskQueue();
    const results: number[] = [];

    queue.enqueue(async () => {
      await sleep(10);
      results.push(1);
    });
    queue.enqueue(() => {
      results.push(2);
    });

    await queue.merge();

    assert.deepStrictEqual(results, [1, 2]);
  });

  it('should handle empty queue', async () => {
    const queue = new AsyncTaskQueue();
    await queue.merge();
    // No error should be thrown.
  });

  it('should handle tasks added after merge is called', async () => {
    const queue = new AsyncTaskQueue();
    const results: number[] = [];

    queue.enqueue(async () => {
      await sleep(10);
      results.push(1);
    });

    queue.enqueue(() => {
      results.push(2);
    });

    assert.deepStrictEqual(results, []);

    await queue.merge();

    assert.deepStrictEqual(results, [1, 2]);
  });

  it('should propagate errors', async () => {
    const queue = new AsyncTaskQueue();
    const error = new Error('test error');

    queue.enqueue(() => {
      throw error;
    });

    await assert.rejects(queue.merge(), error);
  });

  it('should execute tasks without calling merge', async () => {
    const queue = new AsyncTaskQueue();
    const results: number[] = [];

    queue.enqueue(async () => {
      await sleep(20);
      results.push(1);
    });
    queue.enqueue(() => {
      results.push(2);
    });

    await sleep(30);

    assert.deepStrictEqual(results, [1, 2]);
  });

  it('should continue execution after an error', async () => {
    const queue = new AsyncTaskQueue();
    const results: number[] = [];
    const error = new Error('test error');

    queue.enqueue(async () => {
      throw error;
    });

    queue.enqueue(() => {
      results.push(2);
    });

    await queue.merge();

    assert.deepStrictEqual(results, [2]);
  });

  it('should stop execution on error when configured', async () => {
    const queue = new AsyncTaskQueue({ stopOnError: true });
    const results: number[] = [];
    const error = new Error('test error');

    queue.enqueue(async () => {
      throw error;
    });

    queue.enqueue(() => {
      results.push(2);
    });

    await assert.rejects(queue.merge(), error);
    assert.deepStrictEqual(results, []);
  });
});

describe('LazyPromise', () => {
  it('call its function lazily', async () => {
    let called = false;
    const lazy = new LazyPromise((resolver) => {
      called = true;
      resolver('foo');
    });

    assert.ok(!called);

    const result = await lazy;

    assert.ok(called);
    assert.equal(result, 'foo');
  });
});
