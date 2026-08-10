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

import type Anthropic from '@anthropic-ai/sdk';
import * as assert from 'assert';
import { genkit } from 'genkit';
import { describe, test } from 'node:test';
import { anthropic } from '../src/index.js';
import { __testClient } from '../src/types.js';
import {
  createMockAnthropicClient,
  createMockAnthropicMessage,
  mockTextChunk,
} from './mocks/anthropic-client.js';

/**
 * Options for creating a plugin with a mock client
 */
interface CreatePluginOptions {
  apiVersion?: 'beta' | 'stable';
  mockClient: Anthropic;
}

/**
 * Creates an Anthropic plugin configured with a mock client for testing
 */
function createPlugin(options: CreatePluginOptions) {
  return anthropic({
    apiVersion: options.apiVersion,
    // @ts-ignore
    [__testClient]: options.mockClient,
  });
}

/**
 * Creates a Genkit instance with the given plugin
 */
function createGenkitInstance(plugin: ReturnType<typeof anthropic>) {
  return genkit({
    plugins: [plugin],
  });
}

/**
 * Helper to get the proper create stub from the mock client for a given API version.
 */
function getCreateStub(mockClient: Anthropic, apiVersion: 'beta' | 'stable') {
  return apiVersion === 'beta'
    ? (mockClient.beta.messages.create as any)
    : (mockClient.messages.create as any);
}

/**
 * Extracts the API request object from the mock for verification
 * @param apiVersion - 'beta' or 'stable' to determine which API endpoint to check
 */
function getApiRequest(
  mockClient: Anthropic,
  apiVersion: 'beta' | 'stable',
  callIndex: number = 0
) {
  const stub = getCreateStub(mockClient, apiVersion);
  return stub.mock.calls[callIndex]?.arguments[0];
}

/**
 * Verifies that the API was called the expected number of times
 * @param apiVersion - 'beta' or 'stable' to determine which API endpoint to verify
 */
function verifyApiCalled(
  mockClient: Anthropic,
  apiVersion: 'beta' | 'stable',
  expectedCalls: number = 1
) {
  const stub = getCreateStub(mockClient, apiVersion);
  assert.strictEqual(
    stub.mock.calls.length,
    expectedCalls,
    `${apiVersion === 'beta' ? 'Beta' : 'Stable'} API should be called ${expectedCalls} time(s)`
  );
}

/**
 * Tests for effort parameter functionality.
 * These tests verify that output_config.effort is correctly passed to the Anthropic API
 * when using the beta API with claude-opus-4-5.
 */
describe('Effort Parameter Tests', () => {
  const OPUS_4_5_MODEL = 'anthropic/claude-opus-4-5';

  test('should pass output_config.effort to API when using beta API with claude-opus-4-5', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Response with high effort',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    await ai.generate({
      model: OPUS_4_5_MODEL,
      prompt: 'Generate a detailed response',
      config: {
        output_config: {
          effort: 'high',
        },
      },
    });

    verifyApiCalled(mockClient, 'beta');
    const apiRequest = getApiRequest(mockClient, 'beta');

    assert.ok(apiRequest.output_config, 'Request should have output_config');
    assert.strictEqual(
      apiRequest.output_config.effort,
      'high',
      'effort should be set to high'
    );
  });

  test('should pass output_config.effort with low value', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Response with low effort',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    await ai.generate({
      model: OPUS_4_5_MODEL,
      prompt: 'Generate a quick response',
      config: {
        output_config: {
          effort: 'low',
        },
      },
    });

    verifyApiCalled(mockClient, 'beta');
    const apiRequest = getApiRequest(mockClient, 'beta');

    assert.ok(apiRequest.output_config, 'Request should have output_config');
    assert.strictEqual(
      apiRequest.output_config.effort,
      'low',
      'effort should be set to low'
    );
  });

  test('should pass output_config.effort with medium value', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Response with medium effort',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    await ai.generate({
      model: OPUS_4_5_MODEL,
      prompt: 'Generate a balanced response',
      config: {
        output_config: {
          effort: 'medium',
        },
      },
    });

    verifyApiCalled(mockClient, 'beta');
    const apiRequest = getApiRequest(mockClient, 'beta');

    assert.ok(apiRequest.output_config, 'Request should have output_config');
    assert.strictEqual(
      apiRequest.output_config.effort,
      'medium',
      'effort should be set to medium'
    );
  });

  test('should pass output_config.effort in streaming requests', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [mockTextChunk('Streaming response')],
      messageResponse: createMockAnthropicMessage({
        text: 'Streaming response',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    await ai.generate({
      model: OPUS_4_5_MODEL,
      prompt: 'Generate a streaming response',
      config: {
        output_config: {
          effort: 'high',
        },
      },
      streamingCallback: () => {},
    });

    const betaStreamStub = mockClient.beta.messages.stream as any;
    assert.strictEqual(betaStreamStub.mock.calls.length, 1);
    const requestBody = betaStreamStub.mock.calls[0]?.arguments[0];

    assert.ok(
      requestBody.output_config,
      'Streaming request should have output_config'
    );
    assert.strictEqual(
      requestBody.output_config.effort,
      'high',
      'effort should be set to high in streaming request'
    );
  });
});
