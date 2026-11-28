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
import { extractItems, extractJson, parsePartialJson } from '../src/extract';

describe('extract', () => {
  describe('extractItems', () => {
    interface TestStep {
      chunk: string;
      want: unknown[];
    }

    interface TestCase {
      name: string;
      steps: TestStep[];
    }

    const testCases: TestCase[] = [
      {
        name: 'handles simple array in chunks',
        steps: [
          { chunk: '[', want: [] },
          { chunk: '{"a": 1},', want: [{ a: 1 }] },
          { chunk: '{"b": 2}', want: [{ b: 2 }] },
          { chunk: ']', want: [] },
        ],
      },
      {
        name: 'handles nested objects',
        steps: [
          { chunk: '[{"outer": {', want: [] },
          {
            chunk: '"inner": "value"}},',
            want: [{ outer: { inner: 'value' } }],
          },
          { chunk: '{"next": true}]', want: [{ next: true }] },
        ],
      },
      {
        name: 'handles escaped characters',
        steps: [
          { chunk: '[{"text": "line1\\n', want: [] },
          { chunk: 'line2"},', want: [{ text: 'line1\nline2' }] },
          { chunk: '{"text": "tab\\there"}]', want: [{ text: 'tab\there' }] },
        ],
      },
      {
        name: 'ignores content before first array',
        steps: [
          { chunk: 'Here is an array:\n```json\n\n[', want: [] },
          { chunk: '{"a": 1},', want: [{ a: 1 }] },
          { chunk: '{"b": 2}]\n```\nDid you like my array?', want: [{ b: 2 }] },
        ],
      },
      {
        name: 'handles strings',
        steps: [
          { chunk: '["a", "b', want: ['a'] },
          { chunk: ', c\\"d",', want: ['b, c"d'] },
          { chunk: " 'e,]'", want: ['e,]'] },
          { chunk: ']', want: [] },
        ],
      },
      {
        name: 'handles numbers, booleans and null',
        steps: [
          { chunk: '[1, 2', want: [1] },
          { chunk: '3, true,', want: [23, true] },
          { chunk: ' null, -4.5e1]', want: [null, -45] },
        ],
      },
      {
        name: 'handles nested arrays',
        steps: [
          { chunk: '[[1, [2]], ["]"', want: [[1, [2]]] },
          { chunk: '], {"a": [3]}]', want: [[']'], { a: [3] }] },
        ],
      },
      {
        name: 'finds the array once it arrives',
        steps: [
          { chunk: '```json\n', want: [] },
          { chunk: '["a", "b"]', want: ['a', 'b'] },
        ],
      },
      {
        name: 'handles whitespace',
        steps: [
          { chunk: '[\n  ', want: [] },
          { chunk: '{"a": 1},\n  ', want: [{ a: 1 }] },
          { chunk: '{"b": 2}\n]', want: [{ b: 2 }] },
        ],
      },
    ];

    for (const tc of testCases) {
      it(tc.name, () => {
        let text = '';
        let cursor = 0;

        for (const step of tc.steps) {
          text += step.chunk;
          const result = extractItems(text, cursor);
          assert.deepStrictEqual(result.items, step.want);
          cursor = result.cursor;
        }
      });
    }
  });

  describe('extractJson', () => {
    interface TestCase {
      name: string;
      input: {
        text: string;
        throwOnBadJson?: boolean;
      };
      expected?: unknown;
      throws?: boolean;
    }

    const testCases: TestCase[] = [
      {
        name: 'extracts simple object',
        input: {
          text: 'prefix{"a":1}suffix',
        },
        expected: { a: 1 },
      },
      {
        name: 'extracts simple array',
        input: {
          text: 'prefix[1,2,3]suffix',
        },
        expected: [1, 2, 3],
      },
      {
        name: 'handles nested structures',
        input: {
          text: 'text{"a":{"b":[1,2]}}more',
        },
        expected: { a: { b: [1, 2] } },
      },
      {
        name: 'handles strings with braces',
        input: {
          text: '{"text": "not {a} json"}',
        },
        expected: { text: 'not {a} json' },
      },
      {
        name: 'returns null for invalid JSON without throw',
        input: {
          text: 'not json at all',
        },
        expected: null,
      },
      {
        name: 'throws for invalid JSON with throw flag',
        input: {
          text: 'not json at all',
          throwOnBadJson: true,
        },
        throws: true,
      },
    ];

    for (const tc of testCases) {
      it(tc.name, () => {
        if (tc.throws) {
          assert.throws(() => {
            extractJson(tc.input.text, true);
          });
        } else {
          const result = extractJson(
            tc.input.text,
            (tc.input.throwOnBadJson || false) as any
          );
          assert.deepStrictEqual(result, tc.expected);
        }
      });
    }
  });

  describe('parsePartialJson', () => {
    interface TestCase {
      name: string;
      input: string;
      expected: unknown;
    }

    const testCases: TestCase[] = [
      {
        name: 'parses complete object',
        input: '{"a":1,"b":2}',
        expected: { a: 1, b: 2 },
      },
      {
        name: 'parses partial object',
        input: '{"a":1,"b":',
        expected: { a: 1 },
      },
      {
        name: 'parses partial array',
        input: '[1,2,3,',
        expected: [1, 2, 3],
      },
      {
        name: 'parses nested partial structures',
        input: '{"a":{"b":1,"c":]}}',
        expected: { a: { b: 1 } },
      },
    ];

    for (const tc of testCases) {
      it(tc.name, () => {
        const result = parsePartialJson(tc.input);
        assert.deepStrictEqual(result, tc.expected);
      });
    }
  });
});
