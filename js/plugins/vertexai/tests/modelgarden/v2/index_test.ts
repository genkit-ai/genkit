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
import { GenkitError } from 'genkit';
import { describe, it } from 'node:test';
import { vertexModelGarden } from '../../../src/modelgarden/v2';
import {
  AnthropicConfigSchema,
  isAnthropicModelName,
} from '../../../src/modelgarden/v2/anthropic';
import {
  LlamaConfigSchema,
  isLlamaModelName,
} from '../../../src/modelgarden/v2/llama';
import {
  MistralConfigSchema,
  isMistralModelName,
} from '../../../src/modelgarden/v2/mistral';
import { modelName as stripPrefix } from '../../../src/modelgarden/v2/utils';

describe('vertexModelGarden.model helper', () => {
  it('should return an Anthropic model reference', () => {
    const modelName = 'claude-haiku-4-5@20251001';
    const model = vertexModelGarden.model(modelName);
    assert.ok(isAnthropicModelName(stripPrefix(model.name)));
    assert.strictEqual(model.name, `vertex-model-garden/${modelName}`);
    assert.deepStrictEqual(model.configSchema, AnthropicConfigSchema);
  });

  it('should return a Mistral model reference', () => {
    const modelName = 'mistral-medium-3';
    const model = vertexModelGarden.model(modelName);
    assert.ok(isMistralModelName(stripPrefix(model.name)));
    assert.strictEqual(model.name, `vertex-model-garden/${modelName}`);
    assert.deepStrictEqual(model.configSchema, MistralConfigSchema);
  });

  it('should return a Llama model reference', () => {
    const modelName = 'meta/llama-4-maverick-17b-128e-instruct-maas';
    const model = vertexModelGarden.model(modelName);
    assert.ok(isLlamaModelName(stripPrefix(model.name)));
    assert.strictEqual(model.name, `vertex-model-garden/${modelName}`);
    assert.deepStrictEqual(model.configSchema, LlamaConfigSchema);
  });

  it('should return an Anthropic model reference for a pattern-matched name', () => {
    const modelName = 'claude-foo';
    const model = vertexModelGarden.model(modelName);
    assert.ok(isAnthropicModelName(stripPrefix(model.name)));
    assert.strictEqual(model.name, `vertex-model-garden/${modelName}`);
    assert.deepStrictEqual(model.configSchema, AnthropicConfigSchema);
  });

  it('should return a Mistral model reference for a pattern-matched name', () => {
    const modelName = 'mistral-foo';
    const model = vertexModelGarden.model(modelName);
    assert.ok(isMistralModelName(stripPrefix(model.name)));
    assert.strictEqual(model.name, `vertex-model-garden/${modelName}`);
    assert.deepStrictEqual(model.configSchema, MistralConfigSchema);
  });

  it('should return a Llama model reference for a pattern-matched name', () => {
    const modelName = 'meta/llama-foo';
    const model = vertexModelGarden.model(modelName);
    assert.ok(isLlamaModelName(stripPrefix(model.name)));
    assert.strictEqual(model.name, `vertex-model-garden/${modelName}`);
    assert.deepStrictEqual(model.configSchema, LlamaConfigSchema);
  });

  it('should throw an error for an unrecognized model name', () => {
    const modelName = 'unrecognized-model';
    assert.throws(
      () => {
        vertexModelGarden.model(modelName);
      },
      (err: GenkitError) => {
        assert.strictEqual(err.status, 'INVALID_ARGUMENT');
        assert.strictEqual(
          err.message,
          `INVALID_ARGUMENT: model '${modelName}' is not a recognized model name`
        );
        return true;
      }
    );
  });
});
