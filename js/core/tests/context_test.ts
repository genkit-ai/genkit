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
import { describe, it } from 'node:test';
import {
  apiKey,
  type ApiKeyContext,
  type RequestData,
} from '../src/context.js';
import { UserFacingError } from '../src/index.js';

function request(key?: string): RequestData {
  let headers: Record<string, string> = {};

  if (key) {
    headers = { authorization: key };
  }
  return {
    method: 'POST',
    headers,
    input: undefined,
  };
}

describe('apiKey', () => {
  it('can merely save api keys', async () => {
    assert.deepEqual(await apiKey()(request()), {
      auth: { apiKey: undefined },
    });
    assert.deepEqual(await apiKey()(request('key')), {
      auth: {
        apiKey: 'key',
      },
    });
  });

  it('can expect specific keys', async () => {
    assert.deepEqual(await apiKey('key')(request('key')), {
      auth: {
        apiKey: 'key',
      },
    });
    await assert.rejects(
      async () => apiKey('key')(request('wrong-key')),
      new UserFacingError('PERMISSION_DENIED', 'Permission Denied')
    );
    await assert.rejects(
      async () => apiKey('key')(request()),
      new UserFacingError('UNAUTHENTICATED', 'Unauthenticated')
    );
  });

  it('can use a policy function', async () => {
    await apiKey((context: ApiKeyContext) => {
      assert.deepEqual(context, {
        auth: {
          apiKey: 'key',
        },
      });
    })(request('key'));
    await apiKey((context: ApiKeyContext) => {
      assert.deepEqual(context, { auth: { apiKey: undefined } });
    })(request());
  });
});
