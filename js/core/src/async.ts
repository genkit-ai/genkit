/**
 * Copyright 2025 Google LLC
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

// NOTE: This file is pulled into client code and cannot have any Node-only
// dependencies.

/**
 * A handle to a promise and its resolvers.
 */
export interface Task<T> {
  resolve: (result: T) => void;
  reject: (err: unknown) => void;
  promise: Promise<T>;
}

/** Utility for creating Tasks. */
export function createTask<T>(): Task<T> {
  let resolve: unknown, reject: unknown;
  const promise = new Promise<T>(
    (res, rej) => ([resolve, reject] = [res, rej])
  );
  return {
    resolve: resolve as Task<T>['resolve'],
    reject: reject as Task<T>['reject'],
    promise,
  };
}

/**
 * A class designed to help turn repeated callbacks into async iterators.
 * Based loosely on a combination of Go channels and Promises.
 */
export class Channel<T> implements AsyncIterable<T> {
  private ready: Task<void> = createTask<void>();
  private buffer: (T | null)[] = [];
  private err: unknown = null;

  send(value: T): void {
    this.buffer.push(value);
    this.ready.resolve();
  }

  close(): void {
    this.buffer.push(null);
    this.ready.resolve();
  }

  error(err: unknown): void {
    this.err = err;
    // Note: we must call this.ready.reject here in case we get an error even before the stream is initiated,
    // however we cannot rely on this.ready.reject because it will be ignored if ready.resolved has already
    // been called, so this.err will be checked in the iterator as well.
    this.ready.reject(err);
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: async (): Promise<IteratorResult<T>> => {
        if (this.err) {
          throw this.err;
        }

        if (!this.buffer.length) {
          await this.ready.promise;
        }
        const value = this.buffer.shift()!;
        if (!this.buffer.length) {
          this.ready = createTask<void>();
        }

        return {
          value,
          done: !value,
        };
      },
    };
  }
}

/**
 * A lazy promise that does not run its executor function until then is called.
 */
export class LazyPromise<T> implements PromiseLike<T> {
  private executor;
  private promise;

  constructor(executor: (resolve?, reject?) => void | Promise<void>) {
    this.executor = executor;
  }

  then<TResult1 = T, TResult2 = never>(
    onfulfilled?:
      | ((value: T) => TResult1 | PromiseLike<TResult1>)
      | undefined
      | null,
    onrejected?:
      | ((reason: any) => TResult2 | PromiseLike<TResult2>)
      | undefined
      | null
  ): PromiseLike<TResult1 | TResult2> {
    this.promise ??= new Promise<T>(this.executor);
    return this.promise.then(onfulfilled, onrejected);
  }
}

/** Lazily call the provided function to resolve the LazyPromise. */
export function lazy<T>(fn: () => T | PromiseLike<T>): PromiseLike<T> {
  return new LazyPromise<T>((resolve, reject) => {
    try {
      resolve(fn());
    } catch (e) {
      reject(e);
    }
  });
}

/**
 * Options for AsyncTaskQueue.
 */
export interface AsyncTaskQueueOptions {
  /**
   * If true, the queue will stop executing subsequent tasks if a task fails.
   * If false (default), the queue will continue executing subsequent tasks even if a task fails.
   */
  stopOnError?: boolean;
}

/**
 * A queue for asynchronous tasks. The queue ensures that only one task runs at a time in order.
 */
export class AsyncTaskQueue {
  private last: Promise<any> = Promise.resolve();
  private options: AsyncTaskQueueOptions;

  constructor(options?: AsyncTaskQueueOptions) {
    this.options = options || {};
  }

  /**
   * Adds a task to the queue.
   * The task will be executed when its turn comes up in the queue.
   * @param task A function that returns a value or a PromiseLike.
   */
  enqueue(task: () => any | PromiseLike<any>) {
    if (this.options.stopOnError) {
      this.last = this.last.then(() => lazy(task)).then((res) => res);
    } else {
      this.last = this.last
        .catch(() => {})
        .then(() => lazy(task))
        .then((res) => res);
    }
    // Prevent unhandled promise rejections.
    this.last.catch(() => {});
  }

  /**
   * Waits for all tasks currently in the queue to complete.
   */
  async merge() {
    await this.last;
  }
}
