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

import { afterAll } from '@jest/globals';
import fs from 'fs';

/**
 * Several test files mock filesystem behavior by directly reassigning methods
 * on the built-in `fs` module, for example:
 *
 *   fs.writeFileSync = jest.fn(() => {});
 *   fs.promises.writeFile = jest.fn(async () => undefined);
 *
 * `jest.restoreAllMocks()` only restores spies created via `jest.spyOn`; it does
 * NOT undo a direct property assignment. Worse, Jest does not sandbox Node's
 * built-in modules between test files in the same worker — `require('fs')`
 * returns the same shared object everywhere. So a mock assigned (and never
 * restored) in one test file silently corrupts `fs` for every test file that
 * later runs in that same worker, producing flaky, scheduling-dependent
 * failures (e.g. `fs.writeFileSync` becoming a no-op in an unrelated test).
 *
 * To make tests robust regardless of how they mock `fs`, snapshot the real
 * implementations once at load time (before any test mutates them) and restore
 * them after each test FILE completes (`afterAll`). We deliberately do NOT
 * restore between individual tests: some suites set an `fs` mock in one test and
 * rely on it in the next. The cross-FILE leak (within a shared worker) is the
 * actual bug, and `afterAll` cleans up before the next file runs without
 * disturbing intra-file behavior.
 */
function snapshot(source: object): Record<string | symbol, any> {
  const snap: Record<string | symbol, any> = {};
  for (const key of Reflect.ownKeys(source)) {
    try {
      snap[key as any] = (source as any)[key];
    } catch {
      // Ignore getters that throw.
    }
  }
  return snap;
}

const realFs = snapshot(fs);
const realFsPromises = snapshot(fs.promises);

/**
 * Restores own properties of `target` from `snapshot`: deletes any properties a
 * test added that weren't in the original, and re-assigns any that changed.
 * Skips non-writable members (for example `fs.constants`) and unchanged ones.
 * Uses `Reflect.ownKeys` so symbol and non-enumerable properties are handled.
 */
function restoreFrom(
  target: Record<string | symbol, any>,
  snap: Record<string | symbol, any>
) {
  // Remove properties added by a test that weren't part of the snapshot.
  for (const key of Reflect.ownKeys(target)) {
    if (!(key in snap)) {
      try {
        delete target[key];
      } catch {
        // Best-effort; ignore properties that can't be deleted.
      }
    }
  }
  // Restore changed properties back to their original implementations.
  for (const key of Reflect.ownKeys(snap)) {
    if (target[key] === snap[key]) {
      continue;
    }
    const descriptor = Object.getOwnPropertyDescriptor(target, key);
    if (descriptor && descriptor.writable === false) {
      continue;
    }
    try {
      target[key] = snap[key];
    } catch {
      // Best-effort restore; ignore properties that can't be reassigned.
    }
  }
}

afterAll(() => {
  restoreFrom(fs, realFs);
  restoreFrom(fs.promises, realFsPromises);
});
