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

/**
 * A minimal async-iterable queue: producers call `send`/`close`/`error`,
 * consumers `for await` over it. Buffers values sent before they are read.
 *
 * Local to this package so we don't depend on Genkit-internal packages; we only
 * need this small subset (send / close / error / iterate) to bridge boxed
 * server-streaming chunks into an `AsyncGenerator`.
 */
export class Channel<T> implements AsyncIterable<T> {
  private buffer: T[] = [];
  private waiting: Array<{
    resolve: (r: IteratorResult<T>) => void;
    reject: (e: unknown) => void;
  }> = [];
  private done = false;
  private failure?: unknown;

  send(value: T): void {
    if (this.done) return;
    const next = this.waiting.shift();
    if (next) {
      next.resolve({ value, done: false });
    } else {
      this.buffer.push(value);
    }
  }

  close(): void {
    if (this.done) return;
    this.done = true;
    for (const w of this.waiting) {
      w.resolve({ value: undefined, done: true });
    }
    this.waiting = [];
  }

  error(err: unknown): void {
    if (this.done) return;
    this.done = true;
    this.failure = err;
    for (const w of this.waiting) {
      w.reject(err);
    }
    this.waiting = [];
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: (): Promise<IteratorResult<T>> => {
        if (this.buffer.length > 0) {
          return Promise.resolve({ value: this.buffer.shift()!, done: false });
        }
        if (this.failure !== undefined) {
          return Promise.reject(this.failure);
        }
        if (this.done) {
          return Promise.resolve({ value: undefined, done: true });
        }
        return new Promise((resolve, reject) => {
          this.waiting.push({ resolve, reject });
        });
      },
    };
  }
}
