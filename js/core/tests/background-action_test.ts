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

import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { backgroundAction } from '../src/background-action.js';
import { initNodeFeatures } from '../src/node.js';
import { ActionType } from '../src/registry.js';
import { enableTelemetry } from '../src/tracing.js';
import { TestSpanExporter } from './utils.js';

initNodeFeatures();

const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessors: [new SimpleSpanProcessor(spanExporter)],
});

describe('background action', () => {
  beforeEach(() => {
    spanExporter.exportedSpans = [];
  });
  it('should pass options to check and cancel', async () => {
    let checkOptionsReceived: any = null;
    let cancelOptionsReceived: any = null;

    const action = backgroundAction({
      name: 'testAction',
      actionType: 'custom' as ActionType,
      start: async (input, options) => {
        return { id: 'test-op' };
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

    const testOptions = {
      context: {
        customContext: true,
        auth: { token: 'super-secret-auth' },
        secrets: { dbPassword: 'password123' },
      },
    };

    // Test check with explicit options (The new way for passing context/credentials)
    await action.check({ id: 'test-op' }, testOptions);
    assert.deepStrictEqual(checkOptionsReceived?.context, testOptions.context);
    assert.strictEqual(checkOptionsReceived?.context?.customContext, true);
    assert.strictEqual(
      checkOptionsReceived?.context?.auth?.token,
      'super-secret-auth'
    );
    assert.strictEqual(
      checkOptionsReceived?.context?.secrets?.dbPassword,
      'password123'
    );
    assert.ok(checkOptionsReceived?.trace?.traceId);
    assert.ok(checkOptionsReceived?.trace?.spanId);

    // Test cancel with explicit options
    await action.cancel({ id: 'test-op' }, testOptions);
    assert.deepStrictEqual(cancelOptionsReceived?.context, testOptions.context);
    assert.strictEqual(cancelOptionsReceived?.context?.customContext, true);
    assert.strictEqual(
      cancelOptionsReceived?.context?.auth?.token,
      'super-secret-auth'
    );
    assert.strictEqual(
      cancelOptionsReceived?.context?.secrets?.dbPassword,
      'password123'
    );
    assert.ok(cancelOptionsReceived?.trace?.traceId);
    assert.ok(cancelOptionsReceived?.trace?.spanId);

    // Reset received options
    checkOptionsReceived = null;
    cancelOptionsReceived = null;

    // Test check WITHOUT options (ensuring backward compatibility)
    await action.check({ id: 'test-op-2' });
    // When options are omitted, the registry falls back to the ambient AsyncLocalStorage context
    // and generates a new trace ID automatically. We just verify the call succeeds and trace exists.
    assert.ok(checkOptionsReceived?.trace?.traceId);
    assert.ok(checkOptionsReceived?.trace?.spanId);

    // Test cancel WITHOUT options
    await action.cancel({ id: 'test-op-2' });
    assert.ok(cancelOptionsReceived?.trace?.traceId);
    assert.ok(cancelOptionsReceived?.trace?.spanId);
  });
});
