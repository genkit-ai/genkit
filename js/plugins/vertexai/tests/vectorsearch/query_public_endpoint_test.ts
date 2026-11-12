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

import assert from 'assert';
import { describe, it, type Mock } from 'node:test';
import { queryPublicEndpoint } from '../../src/vectorsearch/vector_search/query_public_endpoint';

describe('queryPublicEndpoint', () => {
  // FIXME -- t.mock.method is not supported node above 20
  it.skip('sends the correct request and retrieves neighbors', async (t) => {
    t.mock.method(global, 'fetch', async (url: any, options: any) => {
      return {
        ok: true,
        json: async () => ({ neighbors: ['neighbor1', 'neighbor2'] }),
      } as any;
    });

    const params = {
      featureVector: [0.1, 0.2, 0.3],
      neighborCount: 5,
      accessToken: 'test-access-token',
      projectId: 'test-project-id',
      location: 'us-central1',
      indexEndpointId: 'idx123',
      publicDomainName: 'example.com',
      projectNumber: '123456789',
      deployedIndexId: 'deployed-idx123',
    };

    const expectedResponse = { neighbors: ['neighbor1', 'neighbor2'] };

    const response = await queryPublicEndpoint(params);

    const calls = (
      global.fetch as Mock<
        (url: string, options: Record<string, any>) => Promise<Response>
      >
    ).mock.calls;

    assert.strictEqual(calls.length, 1);

    const [url, options] = calls[0].arguments;

    const expectedUrl = `https://example.com/v1/projects/123456789/locations/us-central1/indexEndpoints/idx123:findNeighbors`;

    assert.strictEqual(url.toString(), expectedUrl);

    assert.strictEqual(options.method, 'POST');

    assert.strictEqual(options.headers['Content-Type'], 'application/json');
    assert.strictEqual(
      options.headers['Authorization'],
      'Bearer test-access-token'
    );

    const body = JSON.parse(options.body);
    const expectedBody = {
      deployed_index_id: 'deployed-idx123',
      queries: [
        {
          datapoint: {
            datapoint_id: '0',
            feature_vector: [0.1, 0.2, 0.3],
            restricts: [],
            numeric_restricts: [],
          },
          neighbor_count: 5,
        },
      ],
    };
    assert.deepStrictEqual(body, expectedBody);

    // Verifying the response
    assert.deepStrictEqual(response, expectedResponse);
  });
});
