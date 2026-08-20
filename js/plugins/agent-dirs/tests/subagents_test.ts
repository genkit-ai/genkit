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

import * as assert from 'node:assert';
import { describe, test } from 'node:test';
import { resolveDelegates } from '../src/compiler.js';

describe('resolveDelegates', () => {
  test('own subagent resolves to self-namespaced name', () => {
    assert.deepStrictEqual(
      resolveDelegates(['refunds'], {
        self: 'support',
        siblings: ['support', 'shipping'],
        subagents: ['refunds'],
      }),
      ['support.refunds']
    );
  });

  test('top-level sibling stays a plain name', () => {
    assert.deepStrictEqual(
      resolveDelegates(['shipping'], {
        self: 'support',
        siblings: ['support', 'shipping'],
        subagents: [],
      }),
      ['shipping']
    );
  });

  test('sibling subagent resolves under the shared parent', () => {
    assert.deepStrictEqual(
      resolveDelegates(['labels'], {
        self: 'support.refunds',
        parent: 'support',
        siblings: ['refunds', 'labels'],
        subagents: [],
      }),
      ['support.labels']
    );
  });

  test('own subagent wins over a same-named sibling', () => {
    assert.deepStrictEqual(
      resolveDelegates(['shipping'], {
        self: 'support',
        siblings: ['support', 'shipping'],
        subagents: ['shipping'],
      }),
      ['support.shipping']
    );
  });

  test('unknown names pass through for runtime resolution', () => {
    assert.deepStrictEqual(
      resolveDelegates(['codeRegistered'], {
        self: 'support',
        siblings: ['support'],
        subagents: [],
      }),
      ['codeRegistered']
    );
  });
});
