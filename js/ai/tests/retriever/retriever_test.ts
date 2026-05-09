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

import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import {
  RetrieverAction,
  defineRetriever,
  retriever,
} from '../../src/retriever.js';

initNodeFeatures();

describe('retriever', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
  });

  it('defines a retriever', async () => {
    const ret = defineRetriever(
      registry,
      {
        name: 'test-retriever',
      },
      async (req) => {
        return {
          documents: [{ content: [{ text: 'hello' }, ...req.content] }],
        };
      }
    );

    assert.strictEqual(ret.__action.name, 'test-retriever');

    const lookedUpAction = (await registry.lookupAction(
      '/retriever/test-retriever'
    )) as RetrieverAction;

    assert.strictEqual(lookedUpAction.__action.name, 'test-retriever');

    assert.deepStrictEqual(
      await lookedUpAction({
        query: { content: [{ text: 'in' }] },
      }),
      {
        documents: [{ content: [{ text: 'hello' }, { text: 'in' }] }],
      }
    );
  });
  it('defines a dynamic retriever', async () => {
    const ret = retriever(
      {
        name: 'test-retriever',
      },
      async (req) => {
        return {
          documents: [{ content: [{ text: 'hello' }, ...req.content] }],
        };
      }
    );

    assert.strictEqual(ret.__action.name, 'test-retriever');

    assert.deepStrictEqual(
      await ret({
        query: { content: [{ text: 'in' }] },
      }),
      {
        documents: [{ content: [{ text: 'hello' }, { text: 'in' }] }],
      }
    );
  });
});
