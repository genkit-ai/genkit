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

import { SpanKind, SpanStatusCode, type SpanContext } from '@opentelemetry/api';
import { ExportResultCode, type ExportResult } from '@opentelemetry/core';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import * as assert from 'assert';
import { afterEach, beforeEach, describe, it } from 'node:test';
import {
  TraceServerExporter,
  setTelemetryServerUrl,
} from '../src/tracing/exporter.js';

const TRACE_A = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const TRACE_B = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';

function fakeSpan(traceId: string, spanId: string): ReadableSpan {
  const spanContext = (): SpanContext => ({
    traceId,
    spanId,
    traceFlags: 1,
    isRemote: false,
  });
  return {
    name: `span-${spanId}`,
    kind: SpanKind.INTERNAL,
    spanContext,
    parentSpanId: undefined,
    startTime: [0, 0],
    endTime: [0, 1],
    status: { code: SpanStatusCode.UNSET },
    attributes: {},
    links: [],
    events: [],
    duration: [0, 1],
    ended: true,
    resource: {} as ReadableSpan['resource'],
    instrumentationLibrary: { name: 'test' },
    droppedAttributesCount: 0,
    droppedEventsCount: 0,
    droppedLinksCount: 0,
  } as ReadableSpan;
}

describe('TraceServerExporter', () => {
  const originalFetch = globalThis.fetch;
  let postedTraceIds: string[];

  beforeEach(() => {
    postedTraceIds = [];
    setTelemetryServerUrl('http://telemetry.test');
    globalThis.fetch = (async (_url: RequestInfo | URL, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? '{}'));
      postedTraceIds.push(body.traceId);
      return new Response('{}', { status: 200 });
    }) as typeof fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('saves every trace in a multi-trace export batch', async () => {
    const exporter = new TraceServerExporter();
    const results: ExportResult[] = [];
    await new Promise<void>((resolve) => {
      exporter.export(
        [
          fakeSpan(TRACE_A, '1111111111111111'),
          fakeSpan(TRACE_B, '2222222222222222'),
        ],
        (result) => {
          results.push(result);
          resolve();
        }
      );
    });
    await new Promise(setImmediate);

    assert.deepStrictEqual(postedTraceIds, [TRACE_A, TRACE_B]);
    assert.deepStrictEqual(results, [{ code: ExportResultCode.SUCCESS }]);
  });

  it('attempts later traces and reports one failure if a save fails', async () => {
    globalThis.fetch = (async (
      _url: RequestInfo | URL,
      init?: RequestInit
    ) => {
      const body = JSON.parse(String(init?.body ?? '{}'));
      postedTraceIds.push(body.traceId);
      if (body.traceId === TRACE_A) {
        throw new Error('save failed');
      }
      return new Response('{}', { status: 200 });
    }) as typeof fetch;
    const exporter = new TraceServerExporter();
    const results: ExportResult[] = [];
    await new Promise<void>((resolve) => {
      exporter.export(
        [
          fakeSpan(TRACE_A, '1111111111111111'),
          fakeSpan(TRACE_B, '2222222222222222'),
        ],
        (result) => {
          results.push(result);
          resolve();
        }
      );
    });
    await new Promise(setImmediate);

    assert.deepStrictEqual(postedTraceIds, [TRACE_A, TRACE_B]);
    assert.deepStrictEqual(results, [{ code: ExportResultCode.FAILED }]);
  });
});
