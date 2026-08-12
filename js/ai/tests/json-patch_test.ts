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
import { applyPatch, diff, type JsonPatch } from '../src/json-patch.js';

/** Asserts that applying `diff(a, b)` to `a` yields `b`. */
function assertRoundTrip(a: unknown, b: unknown) {
  const patch = diff(a, b);
  assert.deepStrictEqual(applyPatch(a, patch), b);
}

describe('json-patch', () => {
  describe('diff', () => {
    it('returns an empty patch for equal values', () => {
      assert.deepStrictEqual(diff({ a: 1 }, { a: 1 }), []);
      assert.deepStrictEqual(diff([1, 2], [1, 2]), []);
      assert.deepStrictEqual(diff('x', 'x'), []);
    });

    it('replaces a changed primitive member', () => {
      assert.deepStrictEqual(diff({ a: 1 }, { a: 2 }), [
        { op: 'replace', path: '/a', value: 2 },
      ]);
    });

    it('adds a new member', () => {
      assert.deepStrictEqual(diff({ a: 1 }, { a: 1, b: 2 }), [
        { op: 'add', path: '/b', value: 2 },
      ]);
    });

    it('removes a deleted member', () => {
      assert.deepStrictEqual(diff({ a: 1, b: 2 }, { a: 1 }), [
        { op: 'remove', path: '/b' },
      ]);
    });

    it('treats an undefined-valued member as absent (no op)', () => {
      // JSON.stringify drops undefined members, so { a: 1, b: undefined } and
      // { a: 1 } are equivalent - the diff must be empty in both directions.
      assert.deepStrictEqual(diff({ a: 1 }, { a: 1, b: undefined }), []);
      assert.deepStrictEqual(diff({ a: 1, b: undefined }, { a: 1 }), []);
      assert.deepStrictEqual(
        diff({ a: 1, b: undefined }, { a: 1, b: undefined }),
        []
      );
    });

    it('does not emit valueless ops for nested undefined members', () => {
      // Regression: a reconstructed parent ({}) vs a live object carrying an
      // explicit `metadata: undefined` must not emit a bogus `add` with no
      // value.
      assert.deepStrictEqual(
        diff(
          { messages: [{ role: 'user' }] },
          {
            messages: [{ role: 'user', metadata: undefined }],
          }
        ),
        []
      );
    });

    it('emits a remove when a member becomes undefined', () => {
      assert.deepStrictEqual(diff({ a: 1, b: 2 }, { a: 1, b: undefined }), [
        { op: 'remove', path: '/b' },
      ]);
    });

    it('emits an add when a member changes from undefined to a value', () => {
      assert.deepStrictEqual(diff({ a: 1, b: undefined }, { a: 1, b: 2 }), [
        { op: 'add', path: '/b', value: 2 },
      ]);
    });

    it('diffs nested objects', () => {
      assert.deepStrictEqual(
        diff({ a: { b: { c: 1 } } }, { a: { b: { c: 2 } } }),
        [{ op: 'replace', path: '/a/b/c', value: 2 }]
      );
    });

    it('appends array items using the "-" token', () => {
      assert.deepStrictEqual(diff({ items: [1] }, { items: [1, 2, 3] }), [
        { op: 'add', path: '/items/-', value: 2 },
        { op: 'add', path: '/items/-', value: 3 },
      ]);
    });

    it('removes trailing array items from the tail', () => {
      assert.deepStrictEqual(diff({ items: [1, 2, 3] }, { items: [1] }), [
        { op: 'remove', path: '/items/2' },
        { op: 'remove', path: '/items/1' },
      ]);
    });

    it('emits a whole-document replace when the root type changes', () => {
      assert.deepStrictEqual(diff({ a: 1 }, [1, 2]), [
        { op: 'replace', path: '', value: [1, 2] },
      ]);
      assert.deepStrictEqual(diff('hello', 42), [
        { op: 'replace', path: '', value: 42 },
      ]);
    });

    it('escapes JSON Pointer tokens (~ and /)', () => {
      assert.deepStrictEqual(diff({}, { 'a/b': 1, 'c~d': 2 }), [
        { op: 'add', path: '/a~1b', value: 1 },
        { op: 'add', path: '/c~0d', value: 2 },
      ]);
    });

    it('round-trips a variety of mutations', () => {
      assertRoundTrip(
        { status: 'a', items: [1, 2], nested: { x: 1 } },
        { status: 'b', items: [1, 2, 3], nested: { x: 1, y: 2 } }
      );
      assertRoundTrip({ a: 1, b: 2 }, { b: 2 });
      assertRoundTrip(undefined, { a: 1 });
      assertRoundTrip({ a: 1 }, undefined);
    });
  });

  describe('applyPatch', () => {
    it('does not mutate the input document', () => {
      const doc = { a: 1 };
      applyPatch(doc, [{ op: 'replace', path: '/a', value: 2 }]);
      assert.deepStrictEqual(doc, { a: 1 });
    });

    it('applies a whole-document replace at the root', () => {
      assert.deepStrictEqual(
        applyPatch({ a: 1 }, [
          { op: 'replace', path: '', value: { b: 2 } },
        ] as JsonPatch),
        { b: 2 }
      );
    });

    it('initializes a missing parent when adding (lenient)', () => {
      assert.deepStrictEqual(
        applyPatch(undefined, [{ op: 'add', path: '/status', value: 'x' }]),
        { status: 'x' }
      );
      assert.deepStrictEqual(
        applyPatch({}, [{ op: 'replace', path: '/a/b', value: 1 }]),
        { a: { b: 1 } }
      );
    });

    it('treats removing a missing member as a no-op', () => {
      assert.deepStrictEqual(
        applyPatch({ a: 1 }, [{ op: 'remove', path: '/missing' }]),
        { a: 1 }
      );
    });

    it('honors test operations', () => {
      assert.deepStrictEqual(
        applyPatch({ a: 1 }, [{ op: 'test', path: '/a', value: 1 }]),
        { a: 1 }
      );
      assert.throws(() =>
        applyPatch({ a: 1 }, [{ op: 'test', path: '/a', value: 2 }])
      );
    });

    it('supports move and copy', () => {
      assert.deepStrictEqual(
        applyPatch({ a: 1 }, [{ op: 'move', from: '/a', path: '/b' }]),
        { b: 1 }
      );
      assert.deepStrictEqual(
        applyPatch({ a: 1 }, [{ op: 'copy', from: '/a', path: '/b' }]),
        { a: 1, b: 1 }
      );
    });

    it('appends to arrays via the "-" token', () => {
      assert.deepStrictEqual(
        applyPatch({ items: [1] }, [{ op: 'add', path: '/items/-', value: 2 }]),
        { items: [1, 2] }
      );
    });
  });

  describe('prototype pollution', () => {
    for (const token of ['__proto__', 'prototype', 'constructor']) {
      it(`rejects "${token}" in an add/replace path`, () => {
        assert.throws(
          () => applyPatch({}, [{ op: 'add', path: `/${token}/x`, value: 1 }]),
          /forbidden token/
        );
        // Object.prototype was not polluted.
        assert.strictEqual(({} as any).x, undefined);
      });

      it(`rejects "${token}" as a leaf token`, () => {
        assert.throws(
          () => applyPatch({}, [{ op: 'add', path: `/${token}`, value: 1 }]),
          /forbidden token/
        );
      });

      it(`rejects "${token}" in a move 'from' pointer`, () => {
        assert.throws(
          () =>
            applyPatch({ a: 1 }, [
              { op: 'move', from: `/${token}`, path: '/b' },
            ]),
          /forbidden token/
        );
      });
    }

    it('does not pollute Object.prototype via a /__proto__/polluted patch', () => {
      assert.throws(
        () =>
          applyPatch({}, [
            { op: 'add', path: '/__proto__/polluted', value: 'yes' },
          ]),
        /forbidden token/
      );
      assert.strictEqual(({} as any).polluted, undefined);
    });
  });
});
