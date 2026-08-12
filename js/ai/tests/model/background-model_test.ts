/**
 * Copyright 2026 Google LLC
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

import { BackgroundActionRunOptions, Operation } from '@genkit-ai/core';
import { initNodeFeatures } from '@genkit-ai/core/node';
import { Registry } from '@genkit-ai/core/registry';
import { enableTelemetry } from '@genkit-ai/core/tracing';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { TestSpanExporter } from '../../../core/tests/utils.js';
import { GenkitAI } from '../../src/genkit-ai.js';
import { GenerateResponseData } from '../../src/model-types.js';
import { defineBackgroundModel } from '../../src/model.js';

initNodeFeatures();
const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessors: [new SimpleSpanProcessor(spanExporter)],
});

describe('background model', () => {
  let registry: Registry;
  let ai: GenkitAI;

  beforeEach(() => {
    registry = new Registry();
    ai = new GenkitAI(registry);
  });

  it('should pass options to check and cancel via ai.checkOperation and ai.cancelOperation', async () => {
    let checkOptionsReceived: any = null;
    let cancelOptionsReceived: any = null;

    defineBackgroundModel(registry, {
      name: 'test-background-model',
      start: async (request) => {
        return {
          id: 'test-op',
          action: '/background-model/test-background-model',
        };
      },
      check: async (op, options) => {
        checkOptionsReceived = options;
        return op;
      },
      cancel: async (op, options) => {
        cancelOptionsReceived = options;
        return op;
      },
    });

    const testOptions: BackgroundActionRunOptions = {
      context: { auth: { customKey: 'secret' } },
      telemetryLabels: { foo: 'bar' },
    };

    const op: Operation<GenerateResponseData> = {
      id: 'test-op',
      action: '/background-model/test-background-model',
    };

    // Test ai.checkOperation passes options down correctly
    await ai.checkOperation(op, testOptions);
    assert.deepStrictEqual(checkOptionsReceived?.context, testOptions.context);
    assert.deepStrictEqual(
      checkOptionsReceived?.telemetryLabels,
      testOptions.telemetryLabels
    );

    // Test ai.cancelOperation passes options down correctly
    await ai.cancelOperation(op, testOptions);
    assert.deepStrictEqual(cancelOptionsReceived?.context, testOptions.context);
    assert.deepStrictEqual(
      cancelOptionsReceived?.telemetryLabels,
      testOptions.telemetryLabels
    );

    // Reset received
    checkOptionsReceived = null;
    cancelOptionsReceived = null;

    // Test omitting options does not throw and works smoothly
    await ai.checkOperation(op);
    assert.ok(
      checkOptionsReceived?.trace?.traceId,
      'check without options should fall back to ambient context and create a trace'
    );

    await ai.cancelOperation(op);
    assert.ok(
      cancelOptionsReceived?.trace?.traceId,
      'cancel without options should fall back to ambient context and create a trace'
    );
  });

  it('should pass context to start via ai.generate', async () => {
    let startOptionsReceived: any = null;

    defineBackgroundModel(registry, {
      name: 'test-background-model-2',
      start: async (request, options) => {
        startOptionsReceived = options;
        return {
          id: 'test-op-2',
          action: '/background-model/test-background-model-2',
        };
      },
      check: async (op) => op,
    });

    const testContext = { auth: { apiKey: 'secret-start-key' } };

    await ai.generate({
      model: 'test-background-model-2',
      prompt: 'hello',
      context: testContext,
    });

    assert.ok(startOptionsReceived, 'start should have received options');
    assert.deepStrictEqual(startOptionsReceived?.context, testContext);
    assert.ok(
      startOptionsReceived?.trace?.traceId,
      'start should receive trace info'
    );
  });
});
