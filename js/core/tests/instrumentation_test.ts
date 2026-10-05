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

import { SpanStatusCode } from '@opentelemetry/api';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { beforeEach, describe, it } from 'node:test';
import { initNodeFeatures } from '../src/node.js';
import { enableTelemetry } from '../src/tracing.js';
import { runInNewSpan } from '../src/tracing/instrumentation.js';
import { TestSpanExporter } from './utils.js';

initNodeFeatures();

const spanExporter = new TestSpanExporter();
enableTelemetry({
  spanProcessors: [new SimpleSpanProcessor(spanExporter)],
});

function exceptionEvents(span: {
  timeEvents?: { timeEvent?: Array<{ annotation?: { description?: string } }> };
}) {
  return (span.timeEvents?.timeEvent || []).filter(
    (e) => e.annotation?.description === 'exception'
  );
}

describe('runInNewSpan exception recording', () => {
  beforeEach(() => {
    spanExporter.exportedSpans = [];
  });

  it('records exception only on the failure-source span when nested spans rethrow', async () => {
    const err = new Error('boom');

    await assert.rejects(
      () =>
        runInNewSpan({ metadata: { name: 'parent' } }, async () =>
          runInNewSpan({ metadata: { name: 'child' } }, async () => {
            throw err;
          })
        ),
      (e: unknown) => e === err
    );

    // Wait a tick for span export.
    await new Promise((r) => setImmediate(r));

    const spans = spanExporter.exportedSpans;
    assert.strictEqual(spans.length, 2);

    const child = spans.find((s) => s.displayName === 'child');
    const parent = spans.find((s) => s.displayName === 'parent');
    assert.ok(child, 'expected child span');
    assert.ok(parent, 'expected parent span');

    // Both spans are ERROR status (failure bubbled).
    assert.strictEqual(child.status.code, SpanStatusCode.ERROR);
    assert.strictEqual(parent.status.code, SpanStatusCode.ERROR);

    // Only the deepest (failure source) span records the exception event.
    assert.strictEqual(exceptionEvents(child).length, 1);
    assert.strictEqual(exceptionEvents(parent).length, 0);

    assert.strictEqual(child.attributes['genkit:isFailureSource'], true);
    assert.notStrictEqual(parent.attributes['genkit:isFailureSource'], true);

    // Error object itself is not polluted with Genkit markers.
    assert.strictEqual(
      Object.prototype.hasOwnProperty.call(err, 'ignoreFailedSpan'),
      false
    );
  });

  it('still records exception once for a single failing span', async () => {
    const err = new Error('solo');

    await assert.rejects(
      () =>
        runInNewSpan({ metadata: { name: 'solo' } }, async () => {
          throw err;
        }),
      (e: unknown) => e === err
    );

    await new Promise((r) => setImmediate(r));

    assert.strictEqual(spanExporter.exportedSpans.length, 1);
    const span = spanExporter.exportedSpans[0];
    assert.strictEqual(exceptionEvents(span).length, 1);
    assert.strictEqual(span.attributes['genkit:isFailureSource'], true);
  });

  it('records shared sentinel errors once per independent trace', async () => {
    const sentinelErr = new Error('sentinel');

    await assert.rejects(
      () =>
        runInNewSpan({ metadata: { name: 'trace1' } }, async () => {
          throw sentinelErr;
        }),
      (e: unknown) => e === sentinelErr
    );

    await assert.rejects(
      () =>
        runInNewSpan({ metadata: { name: 'trace2' } }, async () => {
          throw sentinelErr;
        }),
      (e: unknown) => e === sentinelErr
    );

    await new Promise((r) => setImmediate(r));

    const spans = spanExporter.exportedSpans;
    assert.strictEqual(spans.length, 2);

    const trace1Span = spans.find((s) => s.displayName === 'trace1');
    const trace2Span = spans.find((s) => s.displayName === 'trace2');
    assert.ok(trace1Span);
    assert.ok(trace2Span);

    assert.strictEqual(exceptionEvents(trace1Span).length, 1);
    assert.strictEqual(exceptionEvents(trace2Span).length, 1);
    assert.strictEqual(trace1Span.attributes['genkit:isFailureSource'], true);
    assert.strictEqual(trace2Span.attributes['genkit:isFailureSource'], true);
    assert.notStrictEqual(trace1Span.traceId, trace2Span.traceId);
  });
});
