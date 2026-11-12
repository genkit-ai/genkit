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
import { google } from 'googleapis';
import { describe, it } from 'node:test';
import {
  getAccessToken,
  getProjectNumber,
} from '../../src/vectorsearch/vector_search/utils';

// Mocking the google.auth.getClient method
google.auth.getClient = async () => {
  return {
    getRequestHeaders: async () => ({ Authorization: 'Bearer test-token' }),
  } as any; // Using `any` to bypass type checks for the mock
};

// Mocking the google.cloudresourcemanager method
google.cloudresourcemanager = () => {
  return {
    projects: {
      get: async ({ projectId }: { projectId: string }) => {
        return {
          data: {
            projectNumber: '123456789',
          },
        };
      },
    },
  } as any; // Using `any` to bypass type checks for the mock
};

describe('utils', () => {
  it('getProjectNumber retrieves the project number', async () => {
    const projectId = 'test-project-id';
    const expectedProjectNumber = '123456789';

    const projectNumber = await getProjectNumber(projectId);
    assert.strictEqual(projectNumber, expectedProjectNumber);
  });

  // Mocking the GoogleAuth client
  const mockAuthClient = {
    getAccessToken: async () => ({ token: 'test-access-token' }),
  };

  it('getAccessToken retrieves the access token', async () => {
    // Mocking the GoogleAuth.getClient method to return the mockAuthClient
    const auth = {
      getClient: async () => mockAuthClient,
    } as any; // Using `any` to bypass type checks for the mock

    const accessToken = await getAccessToken(auth);
    assert.strictEqual(accessToken, 'test-access-token');
  });
});
