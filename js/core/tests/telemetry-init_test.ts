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

import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { describe, it } from 'node:test';
import { initNodeFeatures } from '../src/node.js';
import type { TelemetryConfig } from '../src/telemetryTypes.js';
import { enableTelemetry, flushTracing, runInNewSpan } from '../src/tracing.js';

// Kept in its own file: enableTelemetry starts a process-global NodeSDK, and
// node:test runs each file in a separate process.
initNodeFeatures();

describe('telemetry init', () => {
  it('spans opened before an async telemetry config resolves are exported', async () => {
    const exporter = new InMemorySpanExporter();
    // Mimics enableFirebaseTelemetry(): config resolves asynchronously and the
    // call is not awaited by the user.
    const config = new Promise<TelemetryConfig>((resolve) =>
      setTimeout(
        () => resolve({ spanProcessors: [new SimpleSpanProcessor(exporter)] }),
        100
      )
    );
    void enableTelemetry(config);

    let traceId = '';
    await runInNewSpan(
      { metadata: { name: 'coldStart' }, labels: { 'genkit:type': 'action' } },
      async (_meta, span) => {
        traceId = span.spanContext().traceId;
      }
    );

    // Ids come from the real SDK tracer, not the no-op one.
    assert.notStrictEqual(traceId, '0'.repeat(32));
    // SimpleSpanProcessor exports asynchronously (it waits on async resource
    // attributes), so flush before asserting.
    await flushTracing();
    const spans = exporter.getFinishedSpans();
    assert.strictEqual(spans.length, 1);
    assert.strictEqual(spans[0].name, 'coldStart');
    assert.strictEqual(spans[0].spanContext().traceId, traceId);
  });
});
