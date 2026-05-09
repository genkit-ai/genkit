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
import { Document, genkit } from 'genkit';
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { vertexRerankers } from '../../../src/rerankers/v2/index.js';
import {
  defineReranker,
  listKnownRerankers,
} from '../../../src/rerankers/v2/reranker.js';

describe('defineReranker', () => {
  let fetchSpy: sinon.SinonStub;
  let authStub: sinon.SinonStub;

  beforeEach(() => {
    fetchSpy = sinon.stub(global, 'fetch');
    authStub = sinon
      .stub(GoogleAuth.prototype, 'getAccessToken')
      .resolves('test-token');
  });

  afterEach(() => {
    sinon.restore();
  });

  it('should define a reranker and process a request', async () => {
    const mockResponse = {
      records: [
        { id: '1', score: 0.9, content: 'doc2' },
        { id: '0', score: 0.8, content: 'doc1' },
      ],
    };
    fetchSpy.resolves(new Response(JSON.stringify(mockResponse)));

    const rerankerAction = defineReranker('test-reranker', {
      projectId: 'test-project',
      authClient: new GoogleAuth(),
    });

    const query = Document.fromText('test query');
    const documents = [Document.fromText('doc1'), Document.fromText('doc2')];

    const result = (await rerankerAction({
      query,
      documents,
      options: {},
    })) as any;

    assert.deepStrictEqual(result.documents.map((d) => d.text).sort(), [
      'doc1',
      'doc2',
    ]);
    assert.deepStrictEqual(
      result.documents.map((d) => d.metadata.score).sort(),
      [0.8, 0.9]
    );
    assert.strictEqual(fetchSpy.callCount, 1);
    const [url, options] = fetchSpy.getCall(0).args;
    assert.strictEqual(
      url,
      'https://discoveryengine.googleapis.com/v1/projects/test-project/locations/us-central1/rankingConfigs/default_ranking_config:rank'
    );
    const request = JSON.parse(options.body as string);
    assert.strictEqual(request.model, 'test-reranker');
    assert.strictEqual(request!.query, 'test query');
    assert.deepStrictEqual(request!.records, [
      { id: '0', content: 'doc1' },
      { id: '1', content: 'doc2' },
    ]);
    assert.ok(authStub.called);
  });
});

describe('listKnownRerankers', () => {
  it('should list all known rerankers', () => {
    const rerankers = listKnownRerankers({
      projectId: 'test-project',
      authClient: new GoogleAuth(),
    });
    assert.ok(Array.isArray(rerankers));
    assert.ok(rerankers.length > 0);
    rerankers.forEach((reranker) => {
      assert.strictEqual(typeof reranker, 'function');
      assert.strictEqual(typeof reranker.__action.name, 'string');
      assert.ok(reranker.__action.name.startsWith('vertex-rerankers/'));
    });
  });
});

describe('v2RerankFlow with vertexRerankers', () => {
  let fetchSpy: sinon.SinonStub;
  let authStub: sinon.SinonStub;

  beforeEach(() => {
    fetchSpy = sinon.stub(global, 'fetch');
    authStub = sinon
      .stub(GoogleAuth.prototype, 'getAccessToken')
      .resolves('test-token');
  });

  afterEach(() => {
    sinon.restore();
  });

  it('should rerank documents using vertexRerankers', async () => {
    const mockResponse = {
      records: [
        { id: '1', score: 0.9 },
        { id: '0', score: 0.8 },
      ],
    };
    fetchSpy.resolves(new Response(JSON.stringify(mockResponse)));

    const ai = genkit({
      plugins: [
        vertexRerankers({
          projectId: 'test-project',
          location: 'us-central1',
        }),
      ],
    });

    const documents = [Document.fromText('doc1'), Document.fromText('doc2')];

    const response = await ai.rerank({
      reranker: 'vertex-rerankers/semantic-ranker-fast-004',
      documents,
      query: 'test query',
      options: {
        topN: 2,
        ignoreRecordDetailsInResponse: true,
      },
    });

    assert.strictEqual(response.length, 2);
    assert.strictEqual(response[0].text, 'doc2');
    assert.strictEqual(response[0].metadata.score, 0.9);
    assert.strictEqual(response[1].text, 'doc1');
    assert.strictEqual(response[1].metadata.score, 0.8);

    assert.strictEqual(fetchSpy.callCount, 1);
    const [url, options] = fetchSpy.getCall(0).args;
    assert.strictEqual(
      url,
      'https://discoveryengine.googleapis.com/v1/projects/test-project/locations/us-central1/rankingConfigs/default_ranking_config:rank'
    );
    const request = JSON.parse(options.body as string);
    assert.strictEqual(request.model, 'semantic-ranker-fast-004');
    assert.strictEqual(request!.query, 'test query');
    assert.deepStrictEqual(request!.records, [
      { id: '0', content: 'doc1' },
      { id: '1', content: 'doc2' },
    ]);
    assert.strictEqual(request!.topN, 2);
    assert.strictEqual(request!.ignoreRecordDetailsInResponse, true);
    assert.ok(authStub.called);
  });
});
