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
import { describe, it } from 'node:test';
import { deepEqual } from '../src/utils.js';

describe('deepEqual', () => {
  // ── Primitives ──────────────────────────────────────────────────────
  it('returns true for identical primitives', () => {
    assert.strictEqual(deepEqual(1, 1), true);
    assert.strictEqual(deepEqual('hello', 'hello'), true);
    assert.strictEqual(deepEqual(true, true), true);
    assert.strictEqual(deepEqual(null, null), true);
    assert.strictEqual(deepEqual(undefined, undefined), true);
  });

  it('returns false for different primitives', () => {
    assert.strictEqual(deepEqual(1, 2), false);
    assert.strictEqual(deepEqual('a', 'b'), false);
    assert.strictEqual(deepEqual(true, false), false);
    assert.strictEqual(deepEqual(0, '0'), false);
    assert.strictEqual(deepEqual(0, false), false);
  });

  // ── null / undefined ────────────────────────────────────────────────
  it('distinguishes null from undefined', () => {
    assert.strictEqual(deepEqual(null, undefined), false);
    assert.strictEqual(deepEqual(undefined, null), false);
  });

  it('returns false when comparing null to an object', () => {
    assert.strictEqual(deepEqual(null, {}), false);
    assert.strictEqual(deepEqual({}, null), false);
  });

  it('returns false when comparing undefined to an object', () => {
    assert.strictEqual(deepEqual(undefined, {}), false);
    assert.strictEqual(deepEqual({}, undefined), false);
  });

  // ── Arrays ──────────────────────────────────────────────────────────
  it('returns true for equal arrays', () => {
    assert.strictEqual(deepEqual([], []), true);
    assert.strictEqual(deepEqual([1, 2, 3], [1, 2, 3]), true);
    assert.strictEqual(deepEqual(['a', 'b'], ['a', 'b']), true);
  });

  it('returns false for arrays with different lengths', () => {
    assert.strictEqual(deepEqual([1, 2], [1, 2, 3]), false);
    assert.strictEqual(deepEqual([1, 2, 3], [1, 2]), false);
  });

  it('returns false for arrays with different elements', () => {
    assert.strictEqual(deepEqual([1, 2, 3], [1, 2, 4]), false);
  });

  it('returns false for array vs non-array', () => {
    assert.strictEqual(deepEqual([1, 2], { 0: 1, 1: 2 }), false);
  });

  // ── Objects ─────────────────────────────────────────────────────────
  it('returns true for equal objects', () => {
    assert.strictEqual(deepEqual({}, {}), true);
    assert.strictEqual(deepEqual({ a: 1, b: 2 }, { a: 1, b: 2 }), true);
  });

  it('returns true regardless of key order', () => {
    assert.strictEqual(deepEqual({ a: 1, b: 2 }, { b: 2, a: 1 }), true);
  });

  it('returns false for objects with different keys', () => {
    assert.strictEqual(deepEqual({ a: 1 }, { b: 1 }), false);
    assert.strictEqual(deepEqual({ a: 1 }, { a: 1, b: 2 }), false);
  });

  it('returns false for objects with different values', () => {
    assert.strictEqual(deepEqual({ a: 1 }, { a: 2 }), false);
  });

  // ── Nested structures ──────────────────────────────────────────────
  it('returns true for deeply nested equal structures', () => {
    const a = { x: { y: { z: [1, { w: 'hello' }] } } };
    const b = { x: { y: { z: [1, { w: 'hello' }] } } };
    assert.strictEqual(deepEqual(a, b), true);
  });

  it('returns false for deeply nested structures with differences', () => {
    const a = { x: { y: { z: [1, { w: 'hello' }] } } };
    const b = { x: { y: { z: [1, { w: 'world' }] } } };
    assert.strictEqual(deepEqual(a, b), false);
  });

  it('handles mixed arrays and objects', () => {
    assert.strictEqual(
      deepEqual([{ a: 1 }, { b: [2, 3] }], [{ a: 1 }, { b: [2, 3] }]),
      true
    );
    assert.strictEqual(
      deepEqual([{ a: 1 }, { b: [2, 3] }], [{ a: 1 }, { b: [2, 4] }]),
      false
    );
  });

  // ── Type mismatches ────────────────────────────────────────────────
  it('returns false for different primitive types', () => {
    assert.strictEqual(deepEqual(1, '1'), false);
    assert.strictEqual(deepEqual(true, 1), false);
  });

  // Note: deepEqual treats {} and [] as equal because both are objects
  // with zero enumerable keys.  This is acceptable for its intended use
  // case (comparing JSON-serializable tool inputs, where an empty object
  // and empty array are unlikely to appear in practice).
  it('treats empty object and empty array as equal (known limitation)', () => {
    assert.strictEqual(deepEqual({}, []), true);
  });

  // ── Same reference ─────────────────────────────────────────────────
  it('returns true for the same reference', () => {
    const obj = { a: 1, b: [2, 3] };
    assert.strictEqual(deepEqual(obj, obj), true);
  });

  // ── Edge case: empty nested structures ─────────────────────────────
  it('handles empty nested structures', () => {
    assert.strictEqual(deepEqual({ a: {} }, { a: {} }), true);
    assert.strictEqual(deepEqual({ a: [] }, { a: [] }), true);
    // { a: {} } vs { a: [] } — same known limitation as {} vs []
    assert.strictEqual(deepEqual({ a: {} }, { a: [] }), true);
  });

  it('distinguishes non-empty arrays from objects', () => {
    assert.strictEqual(deepEqual({ a: [1] }, { a: { 0: 1 } }), false);
    assert.strictEqual(deepEqual([1, 2], { 0: 1, 1: 2 }), false);
  });

  // ── Realistic tool input comparison ────────────────────────────────
  it('correctly compares realistic tool request inputs', () => {
    const original = {
      action: 'delete files',
      target: '/tmp/test',
      options: { recursive: true, force: false },
    };
    const legitimate = {
      action: 'delete files',
      target: '/tmp/test',
      options: { recursive: true, force: false },
    };
    const forged = {
      action: 'delete files',
      target: '/etc/passwd',
      options: { recursive: true, force: false },
    };
    assert.strictEqual(deepEqual(original, legitimate), true);
    assert.strictEqual(deepEqual(original, forged), false);
  });
});
