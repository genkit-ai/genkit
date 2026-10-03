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

import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import { enableTelemetry } from '@genkit-ai/core/tracing';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { TestSpanExporter } from '../../../core/tests/utils.js';
import { defineGenerateAction } from '../../src/generate/action.js';
import { defineProgrammableModel, type ProgrammableModel } from '../helpers.js';

initNodeFeatures();

const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessors: [new SimpleSpanProcessor(spanExporter)],
});

const SECRET = 'test-api-key-value';

describe('generate telemetry redaction', () => {
  let registry: Registry;
  let pm: ProgrammableModel;

  beforeEach(() => {
    registry = new Registry();
    defineGenerateAction(registry);
    pm = defineProgrammableModel(registry);
    spanExporter.exportedSpans = [];
  });

  it('never exports config.apiKey in any span attribute', async () => {
    pm.handleResponse = async () =>
      ({
        message: { role: 'model', content: [{ text: 'done' }] },
        finishReason: 'stop',
      }) as any;

    const action = await registry.lookupAction('/util/generate');
    await action({
      model: 'programmableModel',
      messages: [{ role: 'user', content: [{ text: 'hello' }] }],
      config: { apiKey: SECRET, temperature: 0.5 },
    } as any);

    // The model must still receive the real key.
    assert.strictEqual(
      (pm.lastRequest as any)?.config?.apiKey,
      SECRET,
      'model should still receive the real apiKey'
    );

    const offenders: string[] = [];
    for (const span of spanExporter.exportedSpans) {
      for (const [attrKey, attrValue] of Object.entries(span.attributes)) {
        if (typeof attrValue === 'string' && attrValue.includes(SECRET)) {
          offenders.push(`${span.displayName} -> ${attrKey}`);
        }
      }
    }

    assert.deepStrictEqual(
      offenders,
      [],
      `apiKey leaked into span attributes: ${offenders.join(', ')}`
    );
  });

  it('preserves non-sensitive config and the shape of genkit:input', async () => {
    pm.handleResponse = async () =>
      ({
        message: { role: 'model', content: [{ text: 'done' }] },
        finishReason: 'stop',
      }) as any;

    const action = await registry.lookupAction('/util/generate');
    await action({
      model: 'programmableModel',
      messages: [{ role: 'user', content: [{ text: 'hello' }] }],
      config: { apiKey: SECRET, temperature: 0.5 },
    } as any);

    const modelSpan = spanExporter.exportedSpans.find(
      (s) => s.displayName === 'programmableModel'
    );
    assert.ok(modelSpan, 'model span should be exported');

    const input = JSON.parse(modelSpan.attributes['genkit:input'] as string);
    assert.strictEqual(input.config.temperature, 0.5);
    assert.strictEqual(
      input.config.apiKey,
      '<redacted>',
      'apiKey key must be kept with a redacted value, not dropped'
    );
  });
});
