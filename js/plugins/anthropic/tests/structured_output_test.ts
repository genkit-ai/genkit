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
import { genkit, z } from 'genkit';
import { describe, test } from 'node:test';
import { anthropic } from '../src/index.js';
import { __testClient } from '../src/types.js';
import {
  createMockAnthropicClient,
  createMockAnthropicMessage,
} from './mocks/anthropic-client.js';

/**
 * Test constants for consistent test setup
 */
const TEST_API_KEY = 'test-key';
const SUPPORTING_MODEL = 'anthropic/claude-sonnet-4-5';
const NON_SUPPORTING_MODEL = 'anthropic/claude-sonnet-4';

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
    apiKey: TEST_API_KEY,
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
 * Tests for structured output (constrained generation) functionality.
 * These tests verify that output_format is correctly passed to the Anthropic API
 * when using the beta API with constrained output, and that it's NOT passed
 * in various edge cases (stable API, non-json format, missing schema, etc.)
 */
describe('Structured Output Tests', () => {
  test('should pass output_format to API when using beta API with constrained output', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice","age":30,"city":"New York","isStudent":false,"isEmployee":true,"isRetired":false,"isUnemployed":false,"isDisabled":false}',
      }),
    });

    // Set up plugin with beta API enabled
    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    // Call generate with sonnet 4.5 (supports native constrained output)
    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt:
        'Generate a fictional person with name "Alice", age 30, and city "New York". Return only the JSON.',
      output: {
        schema: z.object({
          name: z.string(),
          age: z.number(),
          city: z.string(),
          isStudent: z.boolean(),
          isEmployee: z.boolean(),
          isRetired: z.boolean(),
          isUnemployed: z.boolean(),
          isDisabled: z.boolean(),
        }),
        format: 'json',
        constrained: true,
      },
    });

    // Verify the beta API was called
    verifyApiCalled(mockClient, 'beta');

    // Verify output_format was included in the API request
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.ok(apiRequest.output_format, 'Request should have output_format');
    assert.strictEqual(
      apiRequest.output_format.type,
      'json_schema',
      'output_format type should be json_schema'
    );
    assert.ok(
      apiRequest.output_format.schema,
      'output_format should have schema'
    );
    // Verify schema transformation: additionalProperties should be false for constrained output
    assert.strictEqual(
      apiRequest.output_format.schema.additionalProperties,
      false,
      'Schema should have additionalProperties: false'
    );
  });

  test('should NOT pass output_format to API when constrained is false and using beta API', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    // Set up plugin with beta API enabled
    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    // Call generate with constrained: false
    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      output: {
        format: 'json',
        constrained: false,
        schema: z.object({
          name: z.string(),
        }),
      },
    });

    // Verify the beta API was called
    verifyApiCalled(mockClient, 'beta');

    // Verify output_format was NOT included when constrained is false
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual(
      apiRequest.output_format,
      undefined,
      'Request should NOT have output_format when constrained is false'
    );
  });

  test('should NOT pass output_format to API when format is not json and using beta API', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: 'Some text response',
      }),
    });

    // Set up plugin with beta API enabled
    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    // Call generate with format: 'text' (not 'json')
    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate text',
      output: {
        format: 'text',
        constrained: true,
      },
    });

    // Verify the beta API was called
    verifyApiCalled(mockClient, 'beta');

    // Verify output_format was NOT included when format is not json
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual(
      apiRequest.output_format,
      undefined,
      'Request should NOT have output_format when format is text'
    );
  });

  test('should NOT pass output_format to API when schema is not provided and using beta API', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"anything": "goes"}',
      }),
    });

    // Set up plugin with beta API enabled
    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    // Call generate with constrained: true but no schema
    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      output: {
        format: 'json',
        constrained: true,
        // No schema provided
      },
    });

    // Verify the beta API was called
    verifyApiCalled(mockClient, 'beta');

    // Verify output_format was NOT included when schema is missing
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual(
      apiRequest.output_format,
      undefined,
      'Request should NOT have output_format when schema is not provided'
    );
  });

  test('should NOT pass output_format to API when model does not support structured output and using beta API', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    // Set up plugin with beta API enabled
    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    // Call generate with model that does not support structured output
    await ai.generate({
      model: NON_SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      output: {
        format: 'json',
        constrained: true,
      },
    });

    // Verify the beta API was called
    verifyApiCalled(mockClient, 'beta');

    // Verify output_format was NOT included when model does not support structured output
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual(
      apiRequest.output_format,
      undefined,
      'Request should NOT have output_format when model does not support structured output'
    );
  });

  test('should throw an error when using stable API with non-text output format', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice","age":30,"city":"New York"}',
      }),
    });

    // Set up plugin with stable API (not beta)
    const plugin = createPlugin({
      apiVersion: 'stable',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    // Call generate with constrained output (would work with beta API)
    // Expect an error to be thrown since only text output is supported for stable API
    await assert.rejects(
      async () => {
        await ai.generate({
          model: SUPPORTING_MODEL,
          prompt: 'Generate JSON',
          output: {
            format: 'json',
            constrained: true,
            schema: z.object({
              name: z.string(),
              age: z.number(),
              city: z.string(),
            }),
          },
        });
      },
      /Only text output format is supported for Claude models currently/,
      'Should throw an error for non-text output on stable API'
    );
  });
});
