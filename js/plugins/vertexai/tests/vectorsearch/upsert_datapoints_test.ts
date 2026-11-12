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
import type { GoogleAuth } from 'google-auth-library';
import { describe, it, type Mock } from 'node:test';
import type { IIndexDatapoint } from '../../src/vectorsearch/vector_search/types';
import { upsertDatapoints } from '../../src/vectorsearch/vector_search/upsert_datapoints';

describe('upsertDatapoints', () => {
  // FIXME -- t.mock.method is not supported node above 20
  it.skip('upsertDatapoints sends the correct request and handles response', async (t) => {
    // Mocking the fetch method within the test scope
    t.mock.method(global, 'fetch', async (url: any, options: any) => {
      return {
        ok: true,
        json: async () => ({}),
      } as any;
    });

    // Mocking the GoogleAuth client
    const mockAuthClient = {
      getAccessToken: async () => 'test-access-token',
    } as GoogleAuth;

    const params = {
      datapoints: [
        {
          datapointId: 'dp1',
          featureVector: [0.1, 0.2, 0.3],
          restricts: [
            {
              namespace: 'colour',
              allowList: ['blue'],
              denyList: ['red', 'purple'],
            },
          ],
          numericRestricts: [{ namespace: 'shipping code', valueInt: 24 }],
        },
        { datapointId: 'dp2', featureVector: [0.4, 0.5, 0.6] },
      ] as IIndexDatapoint[],
      authClient: mockAuthClient,
      projectId: 'test-project-id',
      location: 'us-central1',
      indexId: 'idx123',
    };

    await upsertDatapoints(params);

    // Verifying the fetch call
    const calls = (
      global.fetch as Mock<
        (url: string, options: Record<string, any>) => Promise<Response>
      >
    ).mock.calls;

    assert.strictEqual(calls.length, 1);
    const [url, options] = calls[0].arguments;

    assert.strictEqual(
      url.toString(),
      'https://us-central1-aiplatform.googleapis.com/v1/projects/test-project-id/locations/us-central1/indexes/idx123:upsertDatapoints'
    );
    assert.strictEqual(options.method, 'POST');
    assert.strictEqual(options.headers['Content-Type'], 'application/json');
    assert.strictEqual(
      options.headers['Authorization'],
      'Bearer test-access-token'
    );

    const body = JSON.parse(options.body);
    assert.deepStrictEqual(body, {
      datapoints: [
        {
          datapoint_id: 'dp1',
          feature_vector: [0.1, 0.2, 0.3],
          restricts: [
            {
              namespace: 'colour',
              allow_list: ['blue'],
              deny_list: ['red', 'purple'],
            },
          ],
          numeric_restricts: [{ namespace: 'shipping code', value_int: 24 }],
        },
        { datapoint_id: 'dp2', feature_vector: [0.4, 0.5, 0.6] },
      ],
    });
  });
});
