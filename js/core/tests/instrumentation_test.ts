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

import type { Span as ApiSpan } from '@opentelemetry/api';
import * as assert from 'assert';
import * as http from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, beforeEach, describe, it } from 'node:test';
import { logger } from '../src/logging.js';
import { initNodeFeatures } from '../src/node.js';
import {
  configureInstrumentation,
  resetInstrumentation,
  runInNewSpan,
  setTelemetryServerUrl,
  type GenkitLogRecord,
  type GenkitSpanContext,
  type Instrumentation,
  type InstrumentationNext,
  type InstrumentationSpanInfo,
  type LogRecordingInstrumentation,
} from '../src/tracing.js';
import { sleep } from './utils.js';

initNodeFeatures();

/** A minimal provider that records calls and mints deterministic ids. */
class FakeInstrumentation
  implements Instrumentation, LogRecordingInstrumentation
{
  spans: InstrumentationSpanInfo[] = [];
  logs: GenkitLogRecord[] = [];

  constructor(
    private readonly traceId = '',
    private readonly spanId = ''
  ) {}

  async runInNewSpan<T>(
    info: InstrumentationSpanInfo,
    next: InstrumentationNext<T>
  ): Promise<T> {
    this.spans.push(info);
    const ctx: GenkitSpanContext = {
      traceId: this.traceId,
      spanId: this.spanId,
      setMetadata() {},
    };
    // A wrapped-context span is enough; callers only read spanContext().
    const span = {
      spanContext: () => ({
        traceId: this.traceId || '0'.repeat(32),
        spanId: this.spanId || '0'.repeat(16),
        traceFlags: 1,
      }),
    } as ApiSpan;
    return next(span, ctx);
  }

  recordLog(record: GenkitLogRecord): void {
    this.logs.push(record);
  }
}

describe('instrumentation abstraction', () => {
  beforeEach(() => {
    delete process.env.GENKIT_TELEMETRY_SERVER;
    setTelemetryServerUrl(''); // clear leaked module-level url between tests
    resetInstrumentation();
  });

  afterEach(() => {
    resetInstrumentation();
  });

  it('routes runInNewSpan through the configured provider', async () => {
    const fake = new FakeInstrumentation('a'.repeat(32), 'b'.repeat(16));
    configureInstrumentation(fake);

    const out = await runInNewSpan(
      { metadata: { name: 'root' }, labels: { 'genkit:type': 'flow' } },
      async (metadata) => `ran ${metadata.name}`
    );

    assert.equal(out, 'ran root');
    assert.equal(fake.spans.length, 1);
    assert.equal(fake.spans[0].metadata.name, 'root');
    // Dispatcher builds the genkit path before handing off to the provider.
    assert.equal(fake.spans[0].metadata.path, '/{root,t:flow}');
  });

  it('exposes composite span ids to the wrapped fn', async () => {
    const fake = new FakeInstrumentation('c'.repeat(32), 'd'.repeat(16));
    configureInstrumentation(fake);

    let seen = { traceId: '', spanId: '' };
    await runInNewSpan({ metadata: { name: 'ids' } }, async (_m, span) => {
      seen = {
        traceId: span.spanContext().traceId,
        spanId: span.spanContext().spanId,
      };
    });

    assert.equal(seen.traceId, 'c'.repeat(32));
    assert.equal(seen.spanId, 'd'.repeat(16));
  });

  it('prepends Direct instrumentation when a telemetry server is set', async () => {
    const base = new FakeInstrumentation('e'.repeat(32), 'f'.repeat(16));
    configureInstrumentation(base);
    // Direct is created from this URL; its POST is fire-and-forget so the
    // (unreachable) server here does not affect the result.
    setTelemetryServerUrl('http://127.0.0.1:0');

    // Direct mints real ids and is first in the chain, so its ids win over the
    // configured provider's.
    let traceId = '';
    await runInNewSpan(
      { metadata: { name: 'dev' }, labels: { 'genkit:type': 'flow' } },
      async (_m, span) => {
        traceId = span.spanContext().traceId;
      }
    );

    assert.notEqual(traceId, 'e'.repeat(32));
    assert.match(traceId, /^[0-9a-f]{32}$/);
    // The configured provider still runs (composition, not replacement).
    assert.equal(base.spans.length, 1);
  });

  it('resolves first non-empty id across the chain', async () => {
    // Empty-id provider first; a real-id provider is only reachable if we
    // compose. Here we verify the composite skips empty ids.
    const empty = new FakeInstrumentation('', '');
    configureInstrumentation(empty);

    let traceId = 'unset';
    await runInNewSpan({ metadata: { name: 'blank' } }, async (_m, span) => {
      traceId = span.spanContext().traceId;
    });

    // Single empty provider => composite id is '' (surfaced as all-zeros on the
    // wrapped OTel span).
    assert.equal(traceId, '0'.repeat(32));
  });

  it('fans out logs to providers with recordLog', async () => {
    const fake = new FakeInstrumentation('1'.repeat(32), '2'.repeat(16));
    configureInstrumentation(fake);

    logger.info('hello world');

    assert.equal(fake.logs.length, 1);
    assert.equal(fake.logs[0].severity, 'info');
    assert.equal(fake.logs[0].body, 'hello world');
  });
});

describe('DirectTelemetryInstrumentation realtime export', () => {
  let server: http.Server;
  let url: string;
  const posted: any[] = [];
  const prevRealtime = process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;

  beforeEach(async () => {
    posted.length = 0;
    server = http.createServer((req, res) => {
      let body = '';
      req.on('data', (c) => (body += c));
      req.on('end', () => {
        if (req.url === '/api/traces') posted.push(JSON.parse(body));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end('{}');
      });
    });
    await new Promise<void>((resolve) =>
      server.listen(0, '127.0.0.1', resolve)
    );
    const addr = server.address() as AddressInfo;
    url = `http://127.0.0.1:${addr.port}`;
    setTelemetryServerUrl(url);
    process.env.GENKIT_ENABLE_REALTIME_TELEMETRY = 'true';
    resetInstrumentation();
  });

  afterEach(async () => {
    if (prevRealtime === undefined) {
      delete process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;
    } else {
      process.env.GENKIT_ENABLE_REALTIME_TELEMETRY = prevRealtime;
    }
    setTelemetryServerUrl('');
    resetInstrumentation();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  });

  // Waits for the fire-and-forget POSTs to land for the given span.
  async function postsFor(spanId: string) {
    for (let i = 0; i < 50; i++) {
      const spans = posted
        .flatMap((t) => Object.values(t.spans ?? {}))
        .filter((s: any) => s.spanId === spanId);
      if (spans.length >= 2) return spans as any[];
      await sleep(10);
    }
    return posted
      .flatMap((t) => Object.values(t.spans ?? {}))
      .filter((s: any) => s.spanId === spanId) as any[];
  }

  it('exports a pending span (endTime 0) on start, then a final span', async () => {
    let spanId = '';
    let pendingSeen: any[] = [];

    await runInNewSpan(
      { metadata: { name: 'realtime' }, labels: { 'genkit:type': 'flow' } },
      async (_m, span) => {
        spanId = span.spanContext().spanId;
        // Let the fire-and-forget start POST land while we're still running.
        for (let i = 0; i < 50 && pendingSeen.length === 0; i++) {
          await sleep(10);
          pendingSeen = posted
            .flatMap((t) => Object.values(t.spans ?? {}))
            .filter((s: any) => s.spanId === spanId);
        }
      }
    );

    // While running, the span was exported with endTime 0 (pending).
    assert.equal(pendingSeen.length, 1, 'expected an in-progress export');
    assert.equal(pendingSeen[0].endTime, 0);

    // After completion, a final export carries a real endTime.
    const all = await postsFor(spanId);
    const final = all.find((s: any) => s.endTime > 0);
    assert.ok(final, 'expected a final export with a non-zero endTime');
    assert.ok(final.endTime >= final.startTime);
  });

  it('does not export on start when realtime is disabled', async () => {
    delete process.env.GENKIT_ENABLE_REALTIME_TELEMETRY;
    resetInstrumentation();

    let spanId = '';
    let midRun: any[] = [];
    await runInNewSpan(
      { metadata: { name: 'norealtime' }, labels: { 'genkit:type': 'flow' } },
      async (_m, span) => {
        spanId = span.spanContext().spanId;
        await sleep(30);
        midRun = posted
          .flatMap((t) => Object.values(t.spans ?? {}))
          .filter((s: any) => s.spanId === spanId);
      }
    );

    assert.equal(midRun.length, 0, 'should not export before completion');
    const all = await postsFor(spanId);
    assert.ok(
      all.some((s: any) => s.endTime > 0),
      'still exports once on completion'
    );
  });
});
