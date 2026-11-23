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
  mockTextChunk,
} from './mocks/anthropic-client.js';

/**
 * Test constants for consistent test setup
 */
const TEST_API_KEY = 'test-key';
const SUPPORTING_MODEL = 'anthropic/claude-sonnet-5-5';
// Not in KNOWN_MODELS, so it resolves through the generic fallback.
const UNLISTED_MODEL = 'anthropic/claude-unlisted-6';

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
 * These tests verify that output_config.format is correctly passed to the Anthropic API
 * when using the beta API with constrained output, and that it's NOT passed
 * in various edge cases (stable API, non-json format, missing schema, etc.)
 */
describe('Structured Output Tests', () => {
  test('should pass output_config.format to API when using beta API with constrained output', async () => {
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

    // Call generate with sonnet 5.5 (supports native constrained output)
    const response = await ai.generate({
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

    // Verify output_config.format was included in the API request
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.ok(
      apiRequest.output_config?.format,
      'Request should have output_config.format'
    );
    assert.strictEqual(
      apiRequest.output_config.format.type,
      'json_schema',
      'output_config.format type should be json_schema'
    );
    assert.ok(
      apiRequest.output_config.format.schema,
      'output_config.format should have schema'
    );
    // Verify schema transformation: additionalProperties should be false for constrained output
    assert.strictEqual(
      apiRequest.output_config.format.schema.additionalProperties,
      false,
      'Schema should have additionalProperties: false'
    );
    assert.strictEqual(apiRequest.output_format, undefined);
    assert.ok(
      !apiRequest.betas.includes('structured-outputs-2025-11-13'),
      'Request should not send the structured-outputs beta header'
    );

    // Verify the JSON text response is parsed into Genkit output
    assert.deepStrictEqual(response.output, {
      name: 'Alice',
      age: 30,
      city: 'New York',
      isStudent: false,
      isEmployee: true,
      isRetired: false,
      isUnemployed: false,
      isDisabled: false,
    });
  });

  test('should merge output_config.format with user output_config', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    const response = await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      config: {
        output_config: {
          effort: 'low',
          task_budget: { type: 'tokens', total: 20000 },
        },
      },
      output: {
        format: 'json',
        constrained: true,
        schema: z.object({
          name: z.string(),
        }),
      },
    });

    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.deepStrictEqual(apiRequest.output_config, {
      effort: 'low',
      task_budget: { type: 'tokens', total: 20000 },
      format: {
        type: 'json_schema',
        schema: {
          type: 'object',
          properties: {
            name: { type: 'string' },
          },
          required: ['name'],
          additionalProperties: false,
        },
      },
    });
    assert.deepStrictEqual(response.output, { name: 'Alice' });
  });

  test('should pass output_config.format in streaming requests', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [mockTextChunk('{"name":"Alice"}')],
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    const response = await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      config: {
        output_config: {
          effort: 'low',
        },
      },
      output: {
        format: 'json',
        constrained: true,
        schema: z.object({
          name: z.string(),
        }),
      },
      streamingCallback: () => {},
    });

    const betaStreamStub = mockClient.beta.messages.stream as any;
    assert.strictEqual(betaStreamStub.mock.calls.length, 1);
    const requestBody = betaStreamStub.mock.calls[0]?.arguments[0];

    assert.strictEqual(requestBody.output_config?.effort, 'low');
    assert.strictEqual(requestBody.output_config.format?.type, 'json_schema');
    assert.strictEqual(
      requestBody.output_config.format.schema.additionalProperties,
      false
    );
    assert.strictEqual(requestBody.output_format, undefined);
    assert.ok(!requestBody.betas.includes('structured-outputs-2025-11-13'));
    assert.deepStrictEqual(response.output, { name: 'Alice' });
  });

  test('should prefer the request output schema over a user output_config.format', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      config: {
        output_config: {
          format: {
            type: 'json_schema',
            schema: { type: 'object', properties: {} },
          },
        },
      },
      output: {
        format: 'json',
        constrained: true,
        schema: z.object({
          name: z.string(),
        }),
      },
    });

    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.deepStrictEqual(
      Object.keys(apiRequest.output_config.format.schema.properties),
      ['name']
    );
  });

  test('should forward a passthrough output_format when output is not constrained', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    const outputFormat = {
      type: 'json_schema',
      schema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
        additionalProperties: false,
      },
    };

    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      config: {
        output_format: outputFormat,
      },
    });

    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.deepStrictEqual(apiRequest.output_format, outputFormat);
    assert.strictEqual(apiRequest.output_config, undefined);
    assert.ok(!apiRequest.betas.includes('structured-outputs-2025-11-13'));
  });

  test('should forward a passthrough output_format when output is not constrained and streaming', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [mockTextChunk('{"name":"Alice"}')],
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    const outputFormat = {
      type: 'json_schema',
      schema: {
        type: 'object',
        properties: { name: { type: 'string' } },
        required: ['name'],
        additionalProperties: false,
      },
    };

    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      config: {
        output_format: outputFormat,
      },
      streamingCallback: () => {},
    });

    const betaStreamStub = mockClient.beta.messages.stream as any;
    assert.strictEqual(betaStreamStub.mock.calls.length, 1);
    const requestBody = betaStreamStub.mock.calls[0]?.arguments[0];
    assert.deepStrictEqual(requestBody.output_format, outputFormat);
    assert.strictEqual(requestBody.output_config, undefined);
    assert.ok(!requestBody.betas.includes('structured-outputs-2025-11-13'));
  });

  test('should not send a passthrough output_format with constrained output', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      config: {
        output_format: {
          type: 'json_schema',
          schema: { type: 'object', properties: {} },
        },
      },
      output: {
        format: 'json',
        constrained: true,
        schema: z.object({
          name: z.string(),
        }),
      },
    });

    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual('output_format' in apiRequest, false);
    assert.deepStrictEqual(apiRequest.output_config, {
      format: {
        type: 'json_schema',
        schema: {
          type: 'object',
          properties: {
            name: { type: 'string' },
          },
          required: ['name'],
          additionalProperties: false,
        },
      },
    });
  });

  test('should not send a passthrough output_format with constrained output when streaming', async () => {
    const mockClient = createMockAnthropicClient({
      streamChunks: [mockTextChunk('{"name":"Alice"}')],
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    await ai.generate({
      model: SUPPORTING_MODEL,
      prompt: 'Generate JSON',
      config: {
        output_format: {
          type: 'json_schema',
          schema: { type: 'object', properties: {} },
        },
      },
      output: {
        format: 'json',
        constrained: true,
        schema: z.object({
          name: z.string(),
        }),
      },
      streamingCallback: () => {},
    });

    const betaStreamStub = mockClient.beta.messages.stream as any;
    assert.strictEqual(betaStreamStub.mock.calls.length, 1);
    const requestBody = betaStreamStub.mock.calls[0]?.arguments[0];

    assert.strictEqual('output_format' in requestBody, false);
    assert.deepStrictEqual(requestBody.output_config, {
      format: {
        type: 'json_schema',
        schema: {
          type: 'object',
          properties: {
            name: { type: 'string' },
          },
          required: ['name'],
          additionalProperties: false,
        },
      },
    });
  });

  test('should NOT pass output_config.format to API when constrained is false and using beta API', async () => {
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

    // Verify output_config.format was NOT included when constrained is false
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual(
      apiRequest.output_config?.format,
      undefined,
      'Request should NOT have output_config.format when constrained is false'
    );
    assert.strictEqual(apiRequest.output_format, undefined);
  });

  test('should NOT pass output_config.format to API when format is not json and using beta API', async () => {
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
        schema: z.string(),
      },
    });

    // Verify the beta API was called
    verifyApiCalled(mockClient, 'beta');

    // Verify output_config.format was NOT included when format is not json
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual(
      apiRequest.output_config?.format,
      undefined,
      'Request should NOT have output_config.format when format is text'
    );
    assert.strictEqual(apiRequest.output_format, undefined);
  });

  test('should NOT pass output_config.format to API when schema is not provided and using beta API', async () => {
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

    // Verify output_config.format was NOT included when schema is missing
    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual(
      apiRequest.output_config?.format,
      undefined,
      'Request should NOT have output_config.format when schema is not provided'
    );
    assert.strictEqual(apiRequest.output_format, undefined);
  });

  test('should NOT pass output_config.format to API for a model ID that is not in the known list', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'beta',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    await ai.generate({
      model: UNLISTED_MODEL,
      prompt: 'Generate JSON',
      output: {
        schema: z.object({ name: z.string() }),
        format: 'json',
        constrained: true,
      },
    });

    verifyApiCalled(mockClient, 'beta');

    const apiRequest = getApiRequest(mockClient, 'beta');
    assert.strictEqual(
      apiRequest.output_config?.format,
      undefined,
      'Request should NOT have output_config.format for an unlisted model'
    );
    assert.strictEqual(apiRequest.output_format, undefined);
    assert.ok(
      JSON.stringify(apiRequest.messages).includes(
        'Output should be in JSON format'
      ),
      'Prompt should carry the schema instructions'
    );
  });

  test('should return schema output for a model ID that is not in the known list on the stable API', async () => {
    const mockClient = createMockAnthropicClient({
      messageResponse: createMockAnthropicMessage({
        text: '{"name":"Alice"}',
      }),
    });

    const plugin = createPlugin({
      apiVersion: 'stable',
      mockClient,
    });

    const ai = createGenkitInstance(plugin);

    const response = await ai.generate({
      model: UNLISTED_MODEL,
      prompt: 'Generate JSON',
      output: {
        schema: z.object({ name: z.string() }),
      },
    });

    verifyApiCalled(mockClient, 'stable');

    const apiRequest = getApiRequest(mockClient, 'stable');
    assert.ok(
      JSON.stringify(apiRequest.messages).includes(
        'Output should be in JSON format'
      ),
      'Prompt should carry the schema instructions'
    );
    assert.deepStrictEqual(response.output, { name: 'Alice' });
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
