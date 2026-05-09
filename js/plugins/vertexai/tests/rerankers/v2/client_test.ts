/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may not use this file except in compliance with the License.
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
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import {
  getVertexRerankUrl,
  rerankerRank,
} from '../../../src/rerankers/v2/client.js';

describe('getVertexRerankUrl', () => {
  it('should return the correct URL with default location', () => {
    const options = {
      projectId: 'test-project',
      authClient: new GoogleAuth(),
    };
    const url = getVertexRerankUrl(options);
    assert.strictEqual(
      url,
      'https://discoveryengine.googleapis.com/v1/projects/test-project/locations/us-central1/rankingConfigs/default_ranking_config:rank'
    );
  });

  it('should return the correct URL with specified location', () => {
    const options = {
      projectId: 'test-project',
      location: 'europe-west1',
      authClient: new GoogleAuth(),
    };
    const url = getVertexRerankUrl(options);
    assert.strictEqual(
      url,
      'https://discoveryengine.googleapis.com/v1/projects/test-project/locations/europe-west1/rankingConfigs/default_ranking_config:rank'
    );
  });
});

describe('rerankerRank', () => {
  let fetchSpy: sinon.SinonStub;

  beforeEach(() => {
    fetchSpy = sinon.stub(global, 'fetch');
    sinon.stub(GoogleAuth.prototype, 'getAccessToken').resolves('test-token');
  });

  afterEach(() => {
    sinon.restore();
  });

  it('should make a successful request and return the response', async () => {
    const mockResponse = { records: [] };
    fetchSpy.resolves(
      new Response(JSON.stringify(mockResponse), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );

    const response = await rerankerRank(
      'test-model',
      {
        model: 'test-model',
        query: 'test-query',
        records: [],
      },
      {
        projectId: 'test-project',
        authClient: new GoogleAuth(),
      }
    );

    assert.deepStrictEqual(response, mockResponse);
    assert.strictEqual(fetchSpy.callCount, 1);
    const [url, options] = fetchSpy.getCall(0).args;
    assert.strictEqual(
      url,
      'https://discoveryengine.googleapis.com/v1/projects/test-project/locations/us-central1/rankingConfigs/default_ranking_config:rank'
    );
    assert.strictEqual(options.method, 'POST');
    assert.deepStrictEqual(options.headers, {
      Authorization: 'Bearer test-token',
      'x-goog-user-project': 'test-project',
      'Content-Type': 'application/json',
    });
    assert.strictEqual(
      options.body,
      JSON.stringify({
        model: 'test-model',
        query: 'test-query',
        records: [],
      })
    );
  });

  it('should throw an error when the request fails', async () => {
    fetchSpy.resolves(
      new Response('Internal Server Error', {
        status: 500,
        statusText: 'Internal Server Error',
      })
    );

    await assert.rejects(
      rerankerRank(
        'test-model',
        {
          model: 'test-model',
          query: 'test-query',
          records: [],
        },
        {
          projectId: 'test-project',
          authClient: new GoogleAuth(),
        }
      ),
      /Error fetching from .* \[500 Internal Server Error\] Internal Server Error/
    );
  });
});
