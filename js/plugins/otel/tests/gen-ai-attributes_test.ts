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
  deriveOutputType,
  deriveProviderName,
  mapFinishReason,
  splitModelName,
} from '../src/genai/gen-ai-attributes.js';

describe('splitModelName', () => {
  it('splits a prefixed model name', () => {
    const result = splitModelName('googleai/gemini-flash-latest');
    assert.strictEqual(result.prefix, 'googleai');
    assert.strictEqual(result.model, 'gemini-flash-latest');
  });

  it('keeps only the first slash as separator', () => {
    const result = splitModelName('vertexai/publishers/google/models/x');
    assert.strictEqual(result.prefix, 'vertexai');
    assert.strictEqual(result.model, 'publishers/google/models/x');
  });

  it('returns undefined prefix when no slash', () => {
    const result = splitModelName('some-model');
    assert.strictEqual(result.prefix, undefined);
    assert.strictEqual(result.model, 'some-model');
  });
});

describe('deriveProviderName', () => {
  it('maps known prefixes', () => {
    assert.strictEqual(deriveProviderName('googleai'), 'gcp.gemini');
    assert.strictEqual(deriveProviderName('google-genai'), 'gcp.gemini');
    assert.strictEqual(deriveProviderName('vertexai'), 'gcp.vertex_ai');
    assert.strictEqual(deriveProviderName('openai'), 'openai');
    assert.strictEqual(deriveProviderName('anthropic'), 'anthropic');
  });

  it('is case insensitive', () => {
    assert.strictEqual(deriveProviderName('GoogleAI'), 'gcp.gemini');
  });

  it('passes unknown prefixes through lowercased', () => {
    assert.strictEqual(deriveProviderName('MyPlugin'), 'myplugin');
  });

  it('returns undefined for undefined or empty', () => {
    assert.strictEqual(deriveProviderName(undefined), undefined);
    assert.strictEqual(deriveProviderName(''), undefined);
  });
});

describe('mapFinishReason', () => {
  it('maps known reasons', () => {
    assert.strictEqual(mapFinishReason('stop', false), 'stop');
    assert.strictEqual(mapFinishReason('length', false), 'length');
    assert.strictEqual(mapFinishReason('blocked', false), 'content_filter');
    assert.strictEqual(mapFinishReason('interrupted', false), 'stop');
  });

  it('falls back based on failure for ambiguous reasons', () => {
    assert.strictEqual(mapFinishReason('other', false), 'stop');
    assert.strictEqual(mapFinishReason('other', true), 'error');
    assert.strictEqual(mapFinishReason(undefined, true), 'error');
  });
});

describe('deriveOutputType', () => {
  it('detects json', () => {
    assert.strictEqual(deriveOutputType('json'), 'json');
    assert.strictEqual(deriveOutputType(undefined, 'application/json'), 'json');
  });

  it('detects text', () => {
    assert.strictEqual(deriveOutputType('text'), 'text');
    assert.strictEqual(deriveOutputType(undefined, 'text/plain'), 'text');
  });

  it('returns undefined when unknown', () => {
    assert.strictEqual(deriveOutputType(undefined, undefined), undefined);
    assert.strictEqual(deriveOutputType('media'), undefined);
  });
});
