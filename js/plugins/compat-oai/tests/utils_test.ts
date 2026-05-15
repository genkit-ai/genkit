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

import { describe, expect, it } from '@jest/globals';
import { GenerateRequest } from 'genkit';
import OpenAI from 'openai';
import {
  maybeCreateRequestScopedOpenAIClient,
  toModelName,
} from '../src/utils';

describe('modelName', () => {
  it('should remove standard prefixes', () => {
    expect(toModelName('/model/gpt-4')).toBe('gpt-4');
    expect(toModelName('/models/gpt-4')).toBe('gpt-4');
    expect(toModelName('/background-model/gpt-4')).toBe('gpt-4');
    expect(toModelName('/embedder/gpt-4')).toBe('gpt-4');
    expect(toModelName('/embedders/gpt-4')).toBe('gpt-4');
  });

  it('should remove custom prefix', () => {
    expect(toModelName('custom/gpt-4', 'custom')).toBe('gpt-4');
    expect(toModelName('/model/custom/gpt-4', 'custom')).toBe('gpt-4');
    expect(toModelName('/model/custom/gpt-4/v2', 'custom')).toBe('gpt-4/v2');
  });

  it('should not remove custom prefix if not provided', () => {
    expect(toModelName('custom/gpt-4')).toBe('custom/gpt-4');
    expect(toModelName('/model/custom/gpt-4')).toBe('custom/gpt-4');
  });

  it('should handle model names with slashes', () => {
    // This reproduces the issue: stripping 'openai' from 'openai/hf.co/Menlo/Jan-nano-gguf:Q4_K_M'
    expect(
      toModelName('openai/hf.co/Menlo/Jan-nano-gguf:Q4_K_M', 'openai')
    ).toBe('hf.co/Menlo/Jan-nano-gguf:Q4_K_M');
  });

  it('should not strip prefix if it appears in the middle', () => {
    expect(toModelName('openai/hf.co/openai/another', 'openai')).toBe(
      'hf.co/openai/another'
    );
  });
});

// TODO, actionName

describe('maybeCreateRequestScopedOpenAIClient', () => {
  it('should copy options from defaultClient when pluginOptions is undefined', () => {
    const defaultClient = new OpenAI({
      apiKey: 'default-key',
      baseURL: 'https://example.com/v1',
      timeout: 12345,
    });

    const request = {
      config: { apiKey: 'scoped-key' },
    } as GenerateRequest;

    const newClient = maybeCreateRequestScopedOpenAIClient(
      undefined,
      request,
      defaultClient
    );

    expect(newClient).not.toBe(defaultClient);
    expect(newClient.apiKey).toBe('scoped-key');
    expect(newClient.baseURL).toBe('https://example.com/v1');
    expect(newClient.timeout).toBe(12345);
  });

  it('should prioritize pluginOptions over defaultClient options', () => {
    const defaultClient = new OpenAI({
      apiKey: 'default-key',
      baseURL: 'https://example.com/v1',
    });

    const pluginOptions = {
      name: 'foo',
      baseURL: 'https://plugin-override.com/v1',
    };

    const request = {
      config: { apiKey: 'scoped-key' },
    } as GenerateRequest;

    const newClient = maybeCreateRequestScopedOpenAIClient(
      pluginOptions,
      request,
      defaultClient
    );

    expect(newClient.apiKey).toBe('scoped-key');
    expect(newClient.baseURL).toBe('https://plugin-override.com/v1');
  });

  it('should verify that _options property exists on defaultClient (library compatibility check)', () => {
    // This test ensures that the OpenAI library version being used still has the private '_options' property
    // that we rely on for copying configuration.
    const defaultClient = new OpenAI({ apiKey: 'test' });
    expect(defaultClient).toHaveProperty('_options');
    expect((defaultClient as any)['_options']).toEqual(
      expect.objectContaining({
        apiKey: 'test',
      })
    );
  });
});
