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
import { z } from 'genkit';
import { describe, it } from 'node:test';
import { AnthropicConfigSchema, resolveBetaEnabled } from '../src/types.js';

describe('resolveBetaEnabled', () => {
  it('should return true when config.apiVersion is beta', () => {
    const config: z.infer<typeof AnthropicConfigSchema> = {
      apiVersion: 'beta',
    };
    assert.strictEqual(resolveBetaEnabled(config, 'stable'), true);
  });

  it('should return true when pluginDefaultApiVersion is beta', () => {
    assert.strictEqual(resolveBetaEnabled(undefined, 'beta'), true);
  });

  it('should return false when config.apiVersion is stable', () => {
    const config: z.infer<typeof AnthropicConfigSchema> = {
      apiVersion: 'stable',
    };
    assert.strictEqual(resolveBetaEnabled(config, 'stable'), false);
  });

  it('should return false when both are stable', () => {
    const config: z.infer<typeof AnthropicConfigSchema> = {
      apiVersion: 'stable',
    };
    assert.strictEqual(resolveBetaEnabled(config, 'stable'), false);
  });

  it('should return false when neither is specified', () => {
    assert.strictEqual(resolveBetaEnabled(undefined, undefined), false);
  });

  it('should return false when config is undefined and plugin default is stable', () => {
    assert.strictEqual(resolveBetaEnabled(undefined, 'stable'), false);
  });

  it('should prioritize config.apiVersion over pluginDefaultApiVersion (beta over stable)', () => {
    const config: z.infer<typeof AnthropicConfigSchema> = {
      apiVersion: 'beta',
    };
    // Even though plugin default is stable, request config should override
    assert.strictEqual(resolveBetaEnabled(config, 'stable'), true);
  });

  it('should prioritize config.apiVersion over pluginDefaultApiVersion (stable over beta)', () => {
    const config: z.infer<typeof AnthropicConfigSchema> = {
      apiVersion: 'stable',
    };
    // Request explicitly wants stable, should override plugin default
    assert.strictEqual(resolveBetaEnabled(config, 'beta'), false);
  });

  it('should return false when config is empty object', () => {
    const config: z.infer<typeof AnthropicConfigSchema> = {};
    assert.strictEqual(resolveBetaEnabled(config, undefined), false);
  });

  it('should return true when config is empty but plugin default is beta', () => {
    const config: z.infer<typeof AnthropicConfigSchema> = {};
    assert.strictEqual(resolveBetaEnabled(config, 'beta'), true);
  });

  it('should handle config with other fields but no apiVersion', () => {
    const config: z.infer<typeof AnthropicConfigSchema> = {
      metadata: { user_id: 'test-user' },
    };
    assert.strictEqual(resolveBetaEnabled(config, 'stable'), false);
    assert.strictEqual(resolveBetaEnabled(config, 'beta'), true);
  });
});
