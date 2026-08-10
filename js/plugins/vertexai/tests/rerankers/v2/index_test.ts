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

import * as assert from 'assert';
import { genkit } from 'genkit';
import { afterEach, describe, it } from 'node:test';
import { __setFakeDerivedParams } from '../../../src/common/index.js';
import { vertexRerankers } from '../../../src/rerankers/v2/index.js';
import * as reranker from '../../../src/rerankers/v2/reranker.js';

describe('vertexRerankersPlugin', () => {
  afterEach(() => {
    __setFakeDerivedParams(undefined);
  });

  it('should initialize and list known rerankers', async () => {
    const ai = genkit({
      plugins: [vertexRerankers({ projectId: 'test-project' })],
    });

    const action = ai.registry.lookupAction(
      'reranker/vertex-rerankers/semantic-ranker-fast-004'
    );
    assert.ok(action);
  });

  it('should resolve a reranker dynamically', async () => {
    __setFakeDerivedParams({
      projectId: 'test-project',
      location: 'us-central1',
    });

    const ai = genkit({
      plugins: [vertexRerankers({})],
    });

    const rerankerRef = reranker.reranker('semantic-ranker-dynamic');
    const action = ai.registry.lookupAction(`reranker/${rerankerRef.name}`);

    assert.ok(action);
  });

  it('should not resolve other action types', async () => {
    const ai = genkit({
      plugins: [vertexRerankers({})],
    });

    const model = await ai.registry.lookupAction(
      'model/vertex-rerankers/semantic-ranker-default@latest'
    );
    assert.strictEqual(model, undefined);
  });
});
