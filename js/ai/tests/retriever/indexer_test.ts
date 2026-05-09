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
  Document,
  IndexerAction,
  defineIndexer,
  indexer,
} from '../../src/retriever.js';

initNodeFeatures();

describe('indexer', () => {
  let registry: Registry;

  beforeEach(() => {
    registry = new Registry();
  });

  it('defines a indexer', async () => {
    let lastIndexerRequest;
    const idx = defineIndexer(
      registry,
      {
        name: 'test-indexer',
      },
      async (req) => {
        lastIndexerRequest = req;
        return;
      }
    );

    assert.strictEqual(idx.__action.name, 'test-indexer');

    const lookedUpAction = (await registry.lookupAction(
      '/indexer/test-indexer'
    )) as IndexerAction;

    assert.strictEqual(lookedUpAction.__action.name, 'test-indexer');

    await lookedUpAction({
      documents: [{ content: [{ text: 'in' }] }],
    });
    assert.deepStrictEqual(lastIndexerRequest, [
      new Document({ content: [{ text: 'in' }] }),
    ]);
  });

  it('defines a dynamic indexer', async () => {
    let lastIndexerRequest;
    const idx = indexer(
      {
        name: 'test-indexer',
      },
      async (req) => {
        lastIndexerRequest = req;
        return;
      }
    );

    assert.strictEqual(idx.__action.name, 'test-indexer');

    await idx({
      documents: [{ content: [{ text: 'in' }] }],
    });
    assert.deepStrictEqual(lastIndexerRequest, [
      new Document({ content: [{ text: 'in' }] }),
    ]);
  });
});
