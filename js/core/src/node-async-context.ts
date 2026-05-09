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

import { AsyncLocalStorage } from 'node:async_hooks';
import { AsyncContext, setAsyncContext } from './async-context.js';

/**
 * Manages AsyncLocalStorage instances in a single place.
 */
export class NodeAsyncContext implements AsyncContext {
  private asls: Record<string, AsyncLocalStorage<any>> = {};

  getStore<T>(key: string): T | undefined {
    return this.asls[key]?.getStore();
  }

  run<T, R>(key: string, store: T, callback: () => R): R {
    if (!this.asls[key]) {
      this.asls[key] = new AsyncLocalStorage<T>();
    }
    return this.asls[key].run(store, callback);
  }
}

export function initNodeAsyncContext() {
  setAsyncContext(new NodeAsyncContext());
}
