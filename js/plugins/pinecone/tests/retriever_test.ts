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
import { toPineconeQuery } from '../src/query.js';

describe('toPineconeQuery', () => {
  it('includes filter when provided', () => {
    const filter = { category: { $eq: 'appliances' } };
    assert.deepStrictEqual(toPineconeQuery({ k: 10, filter }, [0.1, 0.2]), {
      topK: 10,
      vector: [0.1, 0.2],
      includeValues: false,
      includeMetadata: true,
      filter,
    });
  });

  it('omits filter when not provided', () => {
    assert.deepStrictEqual(toPineconeQuery({ k: 5 }, [0.3]), {
      topK: 5,
      vector: [0.3],
      includeValues: false,
      includeMetadata: true,
    });
  });
});
