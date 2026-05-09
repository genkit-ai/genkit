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

import { GenkitError } from './error.js';

const asyncContextKey = '__genkit_AsyncContext';

/**
 * @hidden
 */
export function getAsyncContext(): AsyncContext {
  if (!global[asyncContextKey]) {
    throw new GenkitError({
      status: 'FAILED_PRECONDITION',
      message: 'Async context is not initialized.',
    });
  }
  return global[asyncContextKey];
}

/**
 * @hidden
 */
export function setAsyncContext(context: AsyncContext) {
  if (global[asyncContextKey]) return;
  global[asyncContextKey] = context;
}

/**
 * Manages AsyncLocalStorage instances in a single place.
 */
export interface AsyncContext {
  /** Retrieves the store/value from async context. */
  getStore<T>(key: string): T | undefined;

  /** Exeuctes the callback using the provided store/value for as the async context. */
  run<T, R>(key: string, store: T, callback: () => R): R;
}
